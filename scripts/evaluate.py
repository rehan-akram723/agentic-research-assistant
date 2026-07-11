"""Evaluation benchmark for the Multi-Agent Research Assistant.

Runs a fixed set of benchmark questions against the live FastAPI backend,
measures end-to-end latency, counts retrieved documents, word count of the
final report, and citation verification rate, then saves results to a CSV
in docs/evaluation.md and docs/evaluation.csv.

Usage
-----
    # Start the backend first
    uvicorn src.api.main:app --reload

    # Then in a separate terminal:
    python scripts/evaluate.py

    # Optional flags
    python scripts/evaluate.py --api-url http://127.0.0.1:8000 --max-results 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Benchmark questions (10 representative research questions)
# ---------------------------------------------------------------------------
BENCHMARK_QUESTIONS: list[str] = [
    "What are recent methods for low-resource NLP fine-tuning?",
    "Summarize advances in small language model compression and quantization.",
    "How does retrieval-augmented generation improve factual accuracy in LLMs?",
    "What are state-of-the-art techniques for graph neural networks on heterogeneous data?",
    "Explain recent progress in diffusion models for text-to-image generation.",
    "What are the main approaches for continual learning in deep neural networks?",
    "Summarize research on protein structure prediction with machine learning.",
    "What are the latest methods for efficient transformer self-attention?",
    "How is federated learning applied to preserve data privacy in healthcare AI?",
    "What are current benchmarks and results for multimodal language models?",
]

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS_DIR = _REPO_ROOT / "docs"
_CSV_PATH = _DOCS_DIR / "evaluation.csv"
_MD_PATH = _DOCS_DIR / "evaluation.md"


def _call_research_api(
    question: str,
    api_url: str,
    max_results: int,
    include_citations: bool,
    timeout: int,
) -> dict:
    """POST to /research/ and return the JSON body.

    Args:
        question: The benchmark question.
        api_url: Base URL of the FastAPI server.
        max_results: Maximum papers to retrieve.
        include_citations: Whether to include citation verification.
        timeout: HTTP request timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError: On 4xx/5xx responses.
        requests.exceptions.Timeout: If the server takes too long.
    """
    payload = {
        "question": question,
        "max_results": max_results,
        "include_citations": include_citations,
    }
    resp = requests.post(f"{api_url}/research/", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _word_count(text: str) -> int:
    """Return approximate word count for a string."""
    return len(text.split()) if text else 0


def _citation_rate(citations: list[dict]) -> float:
    """Return fraction of citations that are verified (0.0 – 1.0)."""
    if not citations:
        return 0.0
    verified = sum(1 for c in citations if c.get("is_verified", False))
    return round(verified / len(citations), 3)


def run_evaluation(
    questions: list[str],
    api_url: str,
    max_results: int,
    include_citations: bool,
    request_timeout: int,
) -> list[dict]:
    """Run all benchmark questions and collect metrics.

    Args:
        questions: List of research questions to evaluate.
        api_url: Base URL of the FastAPI server.
        max_results: Max papers per query.
        include_citations: Include citation verification step.
        request_timeout: Per-request HTTP timeout (seconds).

    Returns:
        List of result dicts, one per question.
    """
    results: list[dict] = []
    total = len(questions)

    print(f"\n{'='*64}")
    print(f"  Multi-Agent Research Assistant — Evaluation Benchmark")
    print(f"  Target: {api_url}  |  {total} questions  |  max_results={max_results}")
    print(f"{'='*64}\n")

    for idx, question in enumerate(questions, start=1):
        short_q = question[:60] + ("…" if len(question) > 60 else "")
        print(f"[{idx:02d}/{total}] {short_q}")

        result: dict = {
            "index": idx,
            "question": question,
            "status": "error",
            "latency_seconds": None,
            "docs_retrieved": 0,
            "report_word_count": 0,
            "citations_total": 0,
            "citations_verified": 0,
            "citation_rate": 0.0,
            "queries_used": 0,
            "error_message": None,
        }

        wall_start = time.perf_counter()
        try:
            data = _call_research_api(
                question, api_url, max_results, include_citations, request_timeout
            )
            latency = round(time.perf_counter() - wall_start, 2)

            citations: list[dict] = data.get("citations", [])
            result.update(
                {
                    "status": data.get("status", "unknown"),
                    "latency_seconds": latency,
                    "docs_retrieved": data.get("docs_retrieved", 0),
                    "report_word_count": _word_count(data.get("report", "")),
                    "citations_total": len(citations),
                    "citations_verified": sum(
                        1 for c in citations if c.get("is_verified", False)
                    ),
                    "citation_rate": _citation_rate(citations),
                    "queries_used": len(data.get("queries_used", [])),
                    "error_message": data.get("error_message"),
                }
            )

            status_icon = "✅" if result["status"] == "success" else "⚠️"
            print(
                f"       {status_icon} {latency:.1f}s  |  "
                f"{result['docs_retrieved']} docs  |  "
                f"{result['report_word_count']} words  |  "
                f"citations {result['citations_verified']}/{result['citations_total']}"
            )

        except requests.exceptions.ConnectionError:
            result["error_message"] = "ConnectionError — is the API server running?"
            result["latency_seconds"] = round(time.perf_counter() - wall_start, 2)
            print(f"       ❌ Connection refused — skipping remaining questions.")
            results.append(result)
            break

        except requests.exceptions.Timeout:
            result["error_message"] = f"Timeout after {request_timeout}s"
            result["latency_seconds"] = request_timeout
            print(f"       ⏱ Timed out after {request_timeout}s")

        except Exception as exc:
            result["error_message"] = str(exc)
            result["latency_seconds"] = round(time.perf_counter() - wall_start, 2)
            print(f"       ❌ Error: {exc}")

        results.append(result)

        # Polite delay between questions to avoid hammering Semantic Scholar
        if idx < total:
            time.sleep(2)

    return results


def _save_csv(results: list[dict], path: Path) -> None:
    """Write evaluation results to a CSV file.

    Args:
        results: List of result dicts from ``run_evaluation``.
        path: Destination CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index", "question", "status", "latency_seconds",
        "docs_retrieved", "report_word_count",
        "citations_total", "citations_verified", "citation_rate",
        "queries_used", "error_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n📄 CSV saved → {path}")


def _save_markdown(results: list[dict], path: Path, run_timestamp: str) -> None:
    """Write a human-readable Markdown evaluation report.

    Args:
        results: List of result dicts.
        path: Destination .md file path.
        run_timestamp: ISO-format timestamp string for the report header.
    """
    successes = [r for r in results if r["status"] in ("success", "partial")]
    avg_latency = (
        round(sum(r["latency_seconds"] for r in successes) / len(successes), 2)
        if successes else "N/A"
    )
    avg_docs = (
        round(sum(r["docs_retrieved"] for r in successes) / len(successes), 1)
        if successes else "N/A"
    )
    avg_words = (
        round(sum(r["report_word_count"] for r in successes) / len(successes))
        if successes else "N/A"
    )
    overall_cite_rate = (
        round(
            sum(r["citation_rate"] for r in successes) / len(successes) * 100, 1
        )
        if successes else "N/A"
    )

    lines = [
        "# Evaluation Results",
        "",
        f"> Run timestamp: **{run_timestamp}**  ",
        f"> Questions evaluated: **{len(results)}**  |  "
        f"Succeeded: **{len(successes)}**",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Avg latency | {avg_latency} s |",
        f"| Avg docs retrieved | {avg_docs} |",
        f"| Avg report word count | {avg_words} |",
        f"| Avg citation verification rate | {overall_cite_rate}% |",
        "",
        "## Per-Question Results",
        "",
        "| # | Question (truncated) | Status | Latency (s) | Docs | Words | Citations |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        short_q = r["question"][:55] + ("…" if len(r["question"]) > 55 else "")
        status_icon = {"success": "✅", "partial": "⚠️", "error": "❌"}.get(r["status"], "?")
        latency = f"{r['latency_seconds']:.1f}" if r["latency_seconds"] is not None else "—"
        cite_str = f"{r['citations_verified']}/{r['citations_total']}"
        lines.append(
            f"| {r['index']} | {short_q} | {status_icon} {r['status']} "
            f"| {latency} | {r['docs_retrieved']} | {r['report_word_count']} | {cite_str} |"
        )

    lines += ["", "---", "", "*Generated by `scripts/evaluate.py`*"]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📋 Markdown report saved → {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Research Assistant API on benchmark questions."
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the FastAPI server (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=8,
        help="Max papers to retrieve per question (default: 8).",
    )
    parser.add_argument(
        "--no-citations",
        action="store_true",
        help="Skip citation verification (faster, less thorough).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Per-request timeout in seconds (default: 240).",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=len(BENCHMARK_QUESTIONS),
        help=f"Number of benchmark questions to run (1–{len(BENCHMARK_QUESTIONS)}).",
    )
    args = parser.parse_args()

    questions = BENCHMARK_QUESTIONS[: max(1, min(args.questions, len(BENCHMARK_QUESTIONS)))]
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    results = run_evaluation(
        questions=questions,
        api_url=args.api_url,
        max_results=args.max_results,
        include_citations=not args.no_citations,
        request_timeout=args.timeout,
    )

    _save_csv(results, _CSV_PATH)
    _save_markdown(results, _MD_PATH, run_ts)

    # Print summary to stdout
    successes = sum(1 for r in results if r["status"] in ("success", "partial"))
    print(f"\n{'='*64}")
    print(f"  Evaluation complete — {successes}/{len(results)} questions succeeded.")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
