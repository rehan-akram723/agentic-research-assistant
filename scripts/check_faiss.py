"""scripts/check_faiss.py - FAISS environment diagnostic.

Probes CPU instruction sets, the installed FAISS variant, its internal
BLAS/SIMD configuration, and prints a concrete remediation command if a
faster build is available.

Usage:
    python scripts/check_faiss.py
"""
import sys
import platform
import subprocess
import struct
import importlib


# ── ANSI colours (no deps) ────────────────────────────────────────────────────
_R = "\033[91m"  # red
_G = "\033[92m"  # green
_Y = "\033[93m"  # gold/yellow
_B = "\033[94m"  # blue
_D = "\033[0m"   # reset
_BOLD = "\033[1m"


def _hdr(text: str) -> None:
    width = 64
    print(f"\n{_BOLD}{'-' * width}")
    print(f"  {text}")
    print(f"{'-' * width}{_D}")


def _ok(msg: str) -> None:
    print(f"  {_G}[OK]{_D}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_Y}[!!]{_D}  {msg}")


def _err(msg: str) -> None:
    print(f"  {_R}[XX]{_D}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_B}[ ]{_D}  {msg}")


# ── 1. Python & platform ──────────────────────────────────────────────────────

def check_python() -> None:
    _hdr("Python Environment")
    _info(f"Executable : {sys.executable}")
    _info(f"Version    : {sys.version.split()[0]}")
    _info(f"Platform   : {platform.platform()}")
    _info(f"Architecture: {platform.machine()} / {8 * struct.calcsize('P')}-bit")

    if sys.version_info < (3, 11):
        _warn("Python < 3.11 detected. Upgrading is recommended.")
    else:
        _ok("Python version is adequate.")


# ── 2. CPU capabilities ───────────────────────────────────────────────────────

def _read_cpu_flags_linux() -> set[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":")[1].lower().split())
    except OSError:
        pass
    return set()


def _read_cpu_flags_windows() -> set[str]:
    """Use the 'cpuid' instruction indirectly via wmic or a small ctypes probe."""
    flags: set[str] = set()
    # Attempt via PowerShell (fast, no extra deps)
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-WmiObject -Class Win32_Processor | Select-Object -ExpandProperty Name"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode(errors="ignore").strip()
        _info(f"CPU Model  : {out}")
    except Exception:
        pass

    # Try py-cpuinfo if available
    try:
        import cpuinfo  # type: ignore[import]
        info = cpuinfo.get_cpu_info()
        flags = set(f.lower() for f in info.get("flags", []))
    except ImportError:
        # Fallback: parse CPUID via ctypes on Windows
        flags = _cpuid_flags_ctypes()
    return flags


def _cpuid_flags_ctypes() -> set[str]:
    """Minimal CPUID leaf-7 probe via ctypes to detect AVX2/AVX512."""
    flags: set[str] = set()
    try:
        import ctypes
        import ctypes.wintypes

        # CPUID is not directly callable from Python without a small shellcode
        # trick on Windows. We fall back to reading the registry / env instead.
        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # IsProcessorFeaturePresent constants
        PF_AVX_INSTRUCTIONS_AVAILABLE = 39
        PF_AVX2_INSTRUCTIONS_AVAILABLE = 40
        # 41 = AVX512 (not universally defined)

        if kernel.IsProcessorFeaturePresent(PF_AVX_INSTRUCTIONS_AVAILABLE):
            flags.add("avx")
        if kernel.IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE):
            flags.add("avx2")

        # AVX-512 probed differently on Windows
        try:
            PF_AVX512F_INSTRUCTIONS_AVAILABLE = 41
            if kernel.IsProcessorFeaturePresent(PF_AVX512F_INSTRUCTIONS_AVAILABLE):
                flags.add("avx512f")
        except Exception:
            pass

    except Exception:
        pass
    return flags


def check_cpu() -> dict[str, bool]:
    _hdr("CPU Instruction Set Support")

    system = platform.system()
    if system == "Linux":
        flags = _read_cpu_flags_linux()
    elif system == "Windows":
        flags = _read_cpu_flags_windows()
    elif system == "Darwin":
        # macOS: sysctl -a | grep machdep.cpu.features
        try:
            out = subprocess.check_output(
                ["sysctl", "-a"], stderr=subprocess.DEVNULL, timeout=5
            ).decode(errors="ignore")
            raw = " ".join(
                line.split(":")[-1].lower()
                for line in out.splitlines()
                if "machdep.cpu" in line and "feature" in line
            )
            flags = set(raw.split())
        except Exception:
            flags = set()
    else:
        flags = set()

    cap = {
        "sse4_1": any(f in flags for f in ("sse4_1", "sse4.1", "sse4", "sse4a")),
        "sse4_2": any(f in flags for f in ("sse4_2", "sse4.2")),
        "avx":    "avx" in flags,
        "avx2":   "avx2" in flags,
        "avx512": any(f in flags for f in ("avx512f", "avx512", "avx512bw", "avx512vl")),
        "fma":    "fma" in flags,
    }

    rows = [
        ("SSE4.1 / SSE4.2", cap["sse4_1"] or cap["sse4_2"]),
        ("AVX",             cap["avx"]),
        ("AVX2 + FMA",      cap["avx2"] and cap["fma"]),
        ("AVX-512",         cap["avx512"]),
    ]
    for label, supported in rows:
        if supported:
            _ok(f"{label:<22} SUPPORTED")
        else:
            _warn(f"{label:<22} not detected")

    if not cap["avx2"]:
        _warn(
            "AVX2 not detected. FAISS-AVX2 wheel will still run on this CPU "
            "but without the vectorised SIMD paths — performance will be similar "
            "to faiss-cpu. This is common on VMs or older CPUs."
        )
    return cap


# ── 3. FAISS probe ────────────────────────────────────────────────────────────

def check_faiss(cpu_caps: dict[str, bool]) -> None:
    _hdr("FAISS Installation Probe")

    # Importability
    try:
        import faiss  # type: ignore[import]
    except ImportError:
        _err("faiss is NOT installed in this environment.")
        _print_install_advice(cpu_caps, installed=False)
        return

    _ok(f"faiss imported successfully from: {faiss.__file__}")

    # Version
    ver = getattr(faiss, "__version__", None) or getattr(faiss, "FAISS_VERSION", "unknown")
    _info(f"Version    : {ver}")

    # Detect which variant (avx2 vs generic)
    faiss_path = faiss.__file__.lower()
    if "avx2" in faiss_path:
        _ok("AVX2-optimised wheel detected.")
    elif "gpu" in faiss_path:
        _ok("GPU-enabled build detected.")
    else:
        _warn(
            "Standard (non-AVX2) build detected. "
            "The AVX2 build offers 2–4× faster index operations on compatible CPUs."
        )

    # Functional smoke-test: build a tiny L2 index
    try:
        import numpy as np
        dim = 64
        idx = faiss.IndexFlatL2(dim)
        dummy = np.random.rand(4, dim).astype("float32")
        idx.add(dummy)
        D, I = idx.search(dummy[:1], 2)
        assert I.shape == (1, 2), "Unexpected search output shape."
        _ok(f"IndexFlatL2 smoke-test passed (dim={dim}, ntotal={idx.ntotal}).")
    except Exception as exc:
        _err(f"IndexFlatL2 smoke-test FAILED: {exc}")

    # BLAS info (available in faiss >= 1.7.4)
    try:
        blas_info = faiss.get_compile_options()
        _info(f"Compile options: {blas_info}")
    except AttributeError:
        _info("faiss.get_compile_options() not available in this version — skipping.")

    # Is faiss-cpu from PyPI vs. faiss-cpu-avx2?
    try:
        import importlib.metadata as meta
        installed_packages = {d.metadata["Name"].lower(): d.version for d in meta.distributions()}
        if "faiss-cpu-avx2" in installed_packages:
            _ok(f"faiss-cpu-avx2 {installed_packages['faiss-cpu-avx2']} is installed.")
        elif "faiss-cpu" in installed_packages:
            _warn(
                f"faiss-cpu {installed_packages['faiss-cpu']} (generic build) is installed. "
                f"See remediation advice below."
            )
        elif "faiss-gpu" in installed_packages:
            _ok(f"faiss-gpu {installed_packages['faiss-gpu']} is installed.")
        else:
            _info("Could not identify a faiss-* distribution in metadata.")
    except Exception:
        pass

    _print_install_advice(cpu_caps, installed=True)


def _print_install_advice(cpu_caps: dict[str, bool], installed: bool) -> None:
    _hdr("Remediation Advice")

    pip = _detect_pip()
    avx2 = cpu_caps.get("avx2", False)
    avx512 = cpu_caps.get("avx512", False)

    if avx512:
        target_pkg = "faiss-cpu-avx512"
        perf_note = "AVX-512 SIMD paths — best possible performance."
    elif avx2:
        target_pkg = "faiss-cpu-avx2"
        perf_note = "AVX2+FMA SIMD paths — 2–4× faster than the generic build."
    else:
        target_pkg = "faiss-cpu"
        perf_note = "Generic build — AVX2 not available, this is your optimal choice."

    if not installed:
        _warn(f"FAISS is missing. Install it with:")
        print(f"\n    {_Y}{pip} install {target_pkg}{_D}\n")
        return

    if avx2 or avx512:
        print(
            f"\n  Your CPU supports {'AVX-512' if avx512 else 'AVX2'}. "
            f"If you see slow FAISS operations,\n"
            f"  switch to the optimised build:\n"
        )
        print(f"    {_Y}{pip} uninstall faiss-cpu -y && {pip} install {target_pkg}{_D}\n")
        _info(f"Benefit: {perf_note}")
    else:
        _ok(f"faiss-cpu (generic) is the correct build for this CPU.")

    # requirements.txt guidance
    print(
        f"\n  To pin the correct build in requirements.txt, replace:\n"
        f"    {_R}faiss-cpu>=1.8.0{_D}\n"
        f"  with:\n"
        f"    {_G}{target_pkg}>=1.8.0{_D}\n"
    )


def _detect_pip() -> str:
    """Return the pip invocation appropriate for this interpreter."""
    return f"{sys.executable} -m pip"


# ── 4. Numpy / sentence-transformers ─────────────────────────────────────────

def check_dependencies() -> None:
    _hdr("Key Dependency Versions")

    deps = [
        ("numpy",               "numpy"),
        ("sentence-transformers", "sentence_transformers"),
        ("torch",               "torch"),
        ("langchain",           "langchain"),
        ("langgraph",           "langgraph"),
        ("fastapi",             "fastapi"),
        ("pydantic",            "pydantic"),
    ]
    for display_name, import_name in deps:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            _ok(f"{display_name:<28} {ver}")
        except ImportError:
            _warn(f"{display_name:<28} NOT INSTALLED")


# ── 5. HF token pre-check ────────────────────────────────────────────────────

def check_hf_token() -> None:
    _hdr("Hugging Face Token (Pre-check for Step 4)")
    import os
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        masked = token[:8] + "..." + token[-4:] if len(token) > 12 else "***"
        _ok(f"HF_TOKEN detected: {masked}")
    else:
        _warn(
            "HF_TOKEN not set. Authenticated model downloads will fail.\n"
            "  Set it in your .env file:\n"
            f"    {_Y}HF_TOKEN=hf_your_token_here{_D}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\n{_BOLD}FAISS & Environment Diagnostic  -  Research Assistant{_D}")
    print(f"{'=' * 64}")

    check_python()
    cpu_caps = check_cpu()
    check_faiss(cpu_caps)
    check_dependencies()
    check_hf_token()

    print(f"\n{'=' * 64}")
    print(f"{_BOLD}Diagnostic complete.{_D}  Address any [XX] or [!!] items above")
    print(f"before proceeding to Steps 2-4.\n")


if __name__ == "__main__":
    main()
