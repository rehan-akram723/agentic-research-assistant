import React, { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

// --- TYPES ---
interface Citation {
  id: number;
  claim: string;
  source_title: string;
  is_verified: boolean;
  quote: string | null;
}

interface RunMetrics {
  latency_seconds: number;
  llm_calls: number;
  embedding_time_seconds: number;
  estimated_cost_usd: number;
  [key: string]: any;
}

// --- STATIC DATA ---
const PRESET_QUERIES = [
  "What are recent methods for low-resource NLP fine-tuning?",
  "How does DPO compare to PPO for RLHF alignment?",
  "What is the impact of KV-cache quantization on inference speed?",
  "Latest advancements in multimodal vision-language models"
];

const AGENT_STEPS = [
  { id: 'planner', label: 'Planner', desc: 'Analyzing directive...' },
  { id: 'retrieval', label: 'Retrieval', desc: 'Sourcing academic papers...' },
  { id: 'summarizer', label: 'Summarizer', desc: 'Extracting key claims...' },
  { id: 'verifier', label: 'Verifier', desc: 'Cross-referencing citations...' },
  { id: 'generator', label: 'Generator', desc: 'Synthesizing final report...' }
];

// --- CONTEXT ---
const HoverContext = React.createContext<{
  setActiveHover: (data: { id: string, spanRect: DOMRect } | null) => void;
  citations: Citation[];
} | null>(null);

const CitationBadge = ({ id }: { id: string }) => {
  const context = React.useContext(HoverContext);
  const spanRef = useRef<HTMLSpanElement>(null);

  if (!context) return <span className="text-slate-500">[{id}]</span>;
  
  const cit = context.citations.find(c => c?.id?.toString() === id);
  if (!cit) return <span className="text-slate-500">[{id}]</span>;

  return (
    <span 
      ref={spanRef}
      onMouseEnter={() => {
        if (spanRef.current) {
          context.setActiveHover({ id, spanRect: spanRef.current.getBoundingClientRect() });
        }
      }}
      onMouseLeave={() => {
        context.setActiveHover(null);
      }}
      className={`relative inline-flex items-center mx-1 px-2 py-0.5 rounded-md text-sm font-mono cursor-pointer transition-all duration-300 ${cit.is_verified ? 'bg-green-500/10 text-green-400 hover:bg-green-500/30' : 'bg-red-500/10 text-red-400 hover:bg-red-500/30'}`}
    >
      [{id}]
    </span>
  );
};

function App() {
  // Input State
  const [query, setQuery] = useState("");
  const [isFocused, setIsFocused] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  
  // Pipeline Parameters
  const [maxResults, setMaxResults] = useState(8);
  const [includeCitations, setIncludeCitations] = useState(true);
  const [language, setLanguage] = useState("en");

  // App State
  const [appState, setAppState] = useState<'landing' | 'executing' | 'complete'>('landing');
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [apiError, setApiError] = useState<string | null>(null);

  // Result State
  const [reportText, setReportText] = useState("");
  const [citations, setCitations] = useState<Citation[]>([]);
  const [metrics, setMetrics] = useState<RunMetrics | null>(null);
  
  // UI State
  const [showMetricsDrawer, setShowMetricsDrawer] = useState(false);

  // Hover Overlay State
  const proseRef = useRef<HTMLDivElement>(null);
  const [activeHover, setActiveHover] = useState<{ id: string, spanRect: DOMRect } | null>(null);
  const [overlayState, setOverlayState] = useState<{
    show: boolean,
    id: string,
    y: number,
    x: number,
    lineStart: {x: number, y: number},
    lineEnd: {x: number, y: number},
    ctrl1: {x: number, y: number},
    ctrl2: {x: number, y: number}
  }>({ show: false, id: '', y: 0, x: 0, lineStart: {x:0, y:0}, lineEnd: {x:0, y:0}, ctrl1: {x:0,y:0}, ctrl2: {x:0,y:0} });

  useEffect(() => {
    if (activeHover && proseRef.current) {
      const cit = citations.find(c => c?.id?.toString() === activeHover.id);
      if (!cit) return;

      const proseRect = proseRef.current.getBoundingClientRect();
      const spanRect = activeHover.spanRect;

      const CARD_WIDTH = 450;
      const CARD_HEIGHT_ESTIMATE = 250;

      let targetX = proseRect.right + 40; 
      let targetY = spanRect.top - 20;

      if (targetX + CARD_WIDTH > window.innerWidth - 20) {
        targetX = window.innerWidth - 20 - CARD_WIDTH;
      }
      if (targetY + CARD_HEIGHT_ESTIMATE > window.innerHeight - 20) {
        targetY = window.innerHeight - 20 - CARD_HEIGHT_ESTIMATE;
      }
      if (targetY < 20) {
        targetY = 20;
      }

      const startX = spanRect.right + 5;
      const startY = spanRect.top + (spanRect.height / 2);
      const endX = targetX - 5;
      const endY = targetY + 40; 

      const c1x = Math.max(startX + 40, startX + (endX - startX) / 3);
      const c2x = endX - 40;

      setOverlayState({
        show: true,
        id: activeHover.id,
        y: targetY,
        x: targetX,
        lineStart: { x: startX, y: startY },
        lineEnd: { x: endX, y: endY },
        ctrl1: { x: c1x, y: startY },
        ctrl2: { x: c2x, y: endY }
      });
    } else {
      setOverlayState(prev => ({ ...prev, show: false }));
    }
  }, [activeHover, citations]);

  const handleResearch = async () => {
    if (!query.trim()) return;
    setAppState('executing');
    setCurrentStepIndex(0);
    setApiError(null);
    
    // Simulate progress visually since backend is a blocking call right now
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step < AGENT_STEPS.length - 1) {
        setCurrentStepIndex(step);
      }
    }, 4000);

    try {
      const response = await fetch("http://127.0.0.1:8000/research/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: query,
          max_results: maxResults,
          include_citations: includeCitations
        })
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || "An error occurred during research execution.");
      }

      setReportText(data.report);
      setCitations(data.citations || []);
      setMetrics(data.metrics || null);
      
      clearInterval(interval);
      setCurrentStepIndex(AGENT_STEPS.length - 1);
      setTimeout(() => setAppState('complete'), 1000);

    } catch (err: any) {
      clearInterval(interval);
      setApiError(err.message);
      setAppState('landing');
    }
  };

  const processedReport = (reportText || "").replace(/\[(\d+)\]/g, '[$1](#cite-$1)');
  const activeCit = overlayState.show ? citations.find(c => c?.id?.toString() === overlayState.id) : null;

  return (
    <HoverContext.Provider value={{ setActiveHover, citations }}>
      <div className="relative min-h-screen overflow-x-hidden flex flex-col items-center justify-center">
        {/* Decorative background blurs */}
        <div className="absolute top-0 left-1/4 w-[800px] h-[800px] bg-neon-purple/5 rounded-full blur-[150px] pointer-events-none" />
        <div className="absolute bottom-0 right-1/4 w-[800px] h-[800px] bg-neon-gold/5 rounded-full blur-[150px] pointer-events-none" />

        <AnimatePresence mode="wait">
          
          {appState === 'landing' && (
            <motion.div 
              key="landing"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20, filter: "blur(10px)", scale: 0.95 }}
              transition={{ duration: 0.5, ease: "easeInOut" }}
              className="relative z-10 w-full max-w-4xl mx-auto flex flex-col items-center px-4 py-20"
            >
              {/* Header */}
              <div className="text-center mb-12">
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-glass-border bg-glass mb-6">
                  <span className="w-2 h-2 rounded-full bg-neon-purple animate-pulse" />
                  <span className="text-xs font-medium text-slate-300 tracking-wider uppercase">Agentic Research Engine</span>
                </div>
                <h1 className="text-5xl md:text-7xl font-bold mb-6 text-glow-purple">
                  Deep Research, <br/>
                  <span className="text-slate-300">Simplified.</span>
                </h1>
                <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto font-light">
                  Enter a research directive. Our multi-agent pipeline will autonomously retrieve papers, verify claims, and synthesize a comprehensive report.
                </p>
              </div>

              {/* Error Message */}
              {apiError && (
                <motion.div 
                  initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}
                  className="w-full max-w-3xl mb-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 flex items-start gap-3"
                >
                  <svg xmlns="http://www.w3.org/-2000/svg" className="w-5 h-5 flex-shrink-0 mt-0.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                  <p className="text-sm leading-relaxed font-medium">{apiError}</p>
                </motion.div>
              )}

              {/* Input Engine */}
              <div className={`w-full max-w-3xl transition-all duration-500 ${isFocused || showSettings ? 'scale-[1.02]' : 'scale-100'}`}>
                <div className="relative group">
                  <div className="absolute -inset-1 bg-gradient-to-r from-neon-purple to-indigo-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-500"></div>
                  <div className="relative glass-panel rounded-2xl flex flex-col">
                    <div className="flex items-center p-2 relative">
                      <textarea
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onFocus={() => setIsFocused(true)}
                        onBlur={() => setIsFocused(false)}
                        placeholder="What would you like to research today?"
                        className="w-full bg-transparent border-none text-slate-100 text-lg md:text-xl placeholder-slate-500 resize-none outline-none focus:ring-0 p-4 pb-12 h-24 max-h-48"
                        rows={2}
                        onInput={(e) => {
                          const target = e.target as HTMLTextAreaElement;
                          target.style.height = 'auto';
                          target.style.height = `${Math.min(target.scrollHeight, 192)}px`;
                        }}
                      />
                      
                      <div className="absolute left-6 bottom-4 flex items-center gap-4">
                        <button 
                          onClick={() => setShowSettings(!showSettings)}
                          className={`text-sm flex items-center gap-1.5 transition-colors ${showSettings ? 'text-neon-purple' : 'text-slate-400 hover:text-slate-200'}`}
                        >
                          <svg xmlns="http://www.w3.org/-2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                          Parameters
                        </button>
                      </div>
                      
                      <button 
                        disabled={!query.trim()}
                        onClick={handleResearch}
                        className="absolute right-4 bottom-4 px-4 py-2 bg-gradient-to-r from-neon-purple to-indigo-600 text-white rounded-xl hover:shadow-[0_0_15px_rgba(176,38,255,0.4)] disabled:opacity-50 disabled:hover:shadow-none transition-all flex items-center gap-2 font-medium"
                      >
                        Research
                        <svg xmlns="http://www.w3.org/-2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="22" y1="2" x2="11" y2="13"></line>
                          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                        </svg>
                      </button>
                    </div>
                    
                    {showSettings && (
                      <div className="border-t border-glass-border p-5 bg-black/20 rounded-b-2xl flex flex-wrap gap-6 items-center animate-in fade-in slide-in-from-top-2 duration-300">
                        <div className="flex flex-col gap-2 flex-1 min-w-[200px]">
                          <div className="flex justify-between items-center">
                            <label className="text-sm font-medium text-slate-300">Max Results: {maxResults}</label>
                          </div>
                          <input 
                            type="range" 
                            min="3" max="15" 
                            value={maxResults}
                            onChange={(e) => setMaxResults(Number(e.target.value))}
                            className="w-full accent-neon-purple"
                          />
                        </div>
                        
                        <div className="flex items-center gap-3">
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input 
                              type="checkbox" 
                              className="sr-only peer"
                              checked={includeCitations}
                              onChange={() => setIncludeCitations(!includeCitations)}
                            />
                            <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-neon-purple"></div>
                            <span className="ml-3 text-sm font-medium text-slate-300">Verify Citations</span>
                          </label>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Presets */}
              <div className="mt-12 flex flex-wrap justify-center gap-3 max-w-3xl">
                {PRESET_QUERIES.map((preset, idx) => (
                  <button
                    key={idx}
                    onClick={() => setQuery(preset)}
                    className="px-4 py-2 rounded-xl text-sm text-slate-300 border border-glass-border bg-glass hover:bg-glass-hover hover:border-neon-purple/50 transition-all duration-300 text-left"
                  >
                    {preset}
                  </button>
                ))}
              </div>
            </motion.div>
          )}

          {appState === 'executing' && (
            <motion.div
              key="executing"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.05, filter: "blur(10px)" }}
              transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
              className="relative z-10 w-full max-w-2xl mx-auto flex flex-col items-center justify-center min-h-screen"
            >
              <div className="relative glass-panel rounded-3xl p-10 w-full overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-tr from-neon-purple/20 via-transparent to-neon-gold/20 opacity-50 animate-pulse" />
                
                <div className="relative z-10 flex flex-col items-center">
                  <div className="w-16 h-16 relative mb-8">
                    <div className="absolute inset-0 rounded-full border-2 border-slate-700"></div>
                    <div className="absolute inset-0 rounded-full border-t-2 border-neon-purple animate-spin"></div>
                    <div className="absolute inset-2 rounded-full border-r-2 border-neon-gold animate-[spin_1.5s_linear_infinite_reverse]"></div>
                  </div>
                  
                  <h2 className="text-2xl font-bold text-white mb-2">Executing Pipeline</h2>
                  <p className="text-slate-400 text-center mb-10 max-w-sm line-clamp-2">
                    "{query}"
                  </p>

                  <div className="w-full space-y-4">
                    {AGENT_STEPS.map((step, idx) => {
                      const isCompleted = idx < currentStepIndex;
                      const isActive = idx === currentStepIndex;

                      return (
                        <div key={step.id} className={`flex items-center gap-4 transition-all duration-500 ${isActive ? 'opacity-100 scale-105 ml-2' : isCompleted ? 'opacity-50' : 'opacity-30'}`}>
                          <div className="relative flex-shrink-0 w-8 h-8 flex items-center justify-center">
                            {isCompleted ? (
                              <div className="w-6 h-6 rounded-full bg-neon-purple flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/-2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>
                              </div>
                            ) : isActive ? (
                              <div className="w-3 h-3 rounded-full bg-neon-gold animate-ping" />
                            ) : (
                              <div className="w-2 h-2 rounded-full bg-slate-500" />
                            )}
                            {idx !== AGENT_STEPS.length - 1 && (
                              <div className={`absolute top-8 w-px h-4 ${isCompleted ? 'bg-neon-purple' : 'bg-slate-700'}`} />
                            )}
                          </div>
                          <div className="flex flex-col">
                            <span className={`font-semibold ${isActive ? 'text-white' : 'text-slate-300'}`}>{step.label}</span>
                            <span className="text-xs text-slate-500">{step.desc}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {appState === 'complete' && (
            <motion.div
              key="complete"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="relative z-10 w-full max-w-7xl mx-auto flex flex-col min-h-screen px-4 py-8"
            >
              <div className="glass-panel p-8 md:p-16 rounded-[2.5rem] w-full min-h-[85vh] relative shadow-2xl flex justify-center">
                
                <div className="w-full max-w-3xl flex flex-col relative z-0">
                  {/* Header / Actions */}
                  <div className="flex justify-between items-center mb-12 pb-6 border-b border-white/5">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-neon-purple/10 border border-neon-purple/30 text-neon-purple rounded-2xl flex items-center justify-center shadow-[0_0_15px_rgba(176,38,255,0.2)]">
                        <svg xmlns="http://www.w3.org/-2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                      </div>
                      <div>
                        <h2 className="text-2xl text-slate-100 font-display font-semibold tracking-wide">Research Synthesis</h2>
                        <p className="text-sm text-slate-400">Generated by Agentic Engine</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <button 
                        onClick={() => setShowMetricsDrawer(true)}
                        className="px-5 py-2.5 border border-glass-border bg-glass hover:bg-glass-hover text-slate-300 text-sm font-medium rounded-xl transition-all flex items-center gap-2"
                      >
                        <svg xmlns="http://www.w3.org/-2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                        View Metrics
                      </button>
                      <button 
                        onClick={() => {
                          setAppState('landing');
                          setQuery("");
                        }}
                        className="px-5 py-2.5 bg-white text-slate-900 font-medium text-sm rounded-xl hover:bg-slate-200 transition-all shadow-lg shadow-white/10"
                      >
                        New Query
                      </button>
                    </div>
                  </div>
                  
                  {/* Markdown Content */}
                  <div ref={proseRef} className="prose prose-invert prose-slate prose-lg max-w-none w-full prose-headings:text-glow-purple prose-headings:font-display prose-a:text-neon-gold prose-p:leading-loose pb-20">
                    <ReactMarkdown
                      components={{
                        a: ({ node, href, children, ...props }) => {
                          if (href?.startsWith('#cite-')) {
                            const citeId = href.replace('#cite-', '');
                            return <CitationBadge id={citeId} />;
                          }
                          return <a href={href} {...props}>{children}</a>;
                        }
                      }}
                    >
                      {processedReport}
                    </ReactMarkdown>
                  </div>
                </div>
              </div>

              {/* Telemetry Drawer */}
              <AnimatePresence>
                {showMetricsDrawer && metrics && (
                  <div className="fixed inset-0 z-50 flex justify-end">
                    <motion.div 
                      initial={{ opacity: 0 }} 
                      animate={{ opacity: 1 }} 
                      exit={{ opacity: 0 }} 
                      className="absolute inset-0 bg-black/60 backdrop-blur-sm cursor-pointer"
                      onClick={() => setShowMetricsDrawer(false)}
                    />
                    <motion.div
                      initial={{ x: "100%" }}
                      animate={{ x: 0 }}
                      exit={{ x: "100%" }}
                      transition={{ type: "spring", damping: 25, stiffness: 200 }}
                      className="relative w-full max-w-md h-full bg-[#0b0c10] border-l border-white/10 shadow-2xl flex flex-col p-8"
                    >
                      <div className="flex justify-between items-center mb-10">
                        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                          <svg xmlns="http://www.w3.org/-2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-neon-gold"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
                          Run Telemetry
                        </h2>
                        <button onClick={() => setShowMetricsDrawer(false)} className="text-slate-500 hover:text-white transition-colors">
                          <svg xmlns="http://www.w3.org/-2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                        </button>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mb-8">
                        <div className="glass-panel p-5 rounded-2xl flex flex-col">
                          <span className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-2">Total Latency</span>
                          <span className="text-3xl font-light text-white">{metrics.latency_seconds?.toFixed(2)}s</span>
                        </div>
                        <div className="glass-panel p-5 rounded-2xl flex flex-col">
                          <span className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-2">Cost (USD)</span>
                          <span className="text-3xl font-light text-neon-gold">${metrics.estimated_cost_usd?.toFixed(4)}</span>
                        </div>
                        <div className="glass-panel p-5 rounded-2xl flex flex-col">
                          <span className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-2">LLM Calls</span>
                          <span className="text-3xl font-light text-white">{metrics.llm_calls}</span>
                        </div>
                        <div className="glass-panel p-5 rounded-2xl flex flex-col">
                          <span className="text-xs uppercase tracking-wider text-slate-500 font-bold mb-2">Embedding Time</span>
                          <span className="text-3xl font-light text-white">{metrics.embedding_time_seconds?.toFixed(2)}s</span>
                        </div>
                      </div>

                      <div className="mt-auto">
                        <div className="text-xs text-slate-500 text-center font-mono">
                          ID: {Math.random().toString(36).substring(7).toUpperCase()} • AGENTIC ENGINE
                        </div>
                      </div>
                    </motion.div>
                  </div>
                )}
              </AnimatePresence>

              {/* Portal for the Hover Overlay (Fixed Positioning to Viewport) */}
              {typeof document !== 'undefined' && createPortal(
                <div className="fixed inset-0 pointer-events-none z-40 overflow-hidden">
                  <svg className="absolute inset-0 w-full h-full">
                    <AnimatePresence>
                      {overlayState.show && activeCit && (
                        <motion.path
                          initial={{ pathLength: 0, opacity: 0 }}
                          animate={{ pathLength: 1, opacity: 1 }}
                          exit={{ opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          d={`M ${overlayState.lineStart.x} ${overlayState.lineStart.y} C ${overlayState.ctrl1.x} ${overlayState.ctrl1.y}, ${overlayState.ctrl2.x} ${overlayState.ctrl2.y}, ${overlayState.lineEnd.x} ${overlayState.lineEnd.y}`}
                          fill="none"
                          stroke={activeCit.is_verified ? "rgba(34, 197, 94, 0.4)" : "rgba(239, 68, 68, 0.4)"}
                          strokeWidth="3"
                          strokeLinecap="round"
                        />
                      )}
                    </AnimatePresence>
                  </svg>

                  <AnimatePresence>
                    {overlayState.show && activeCit && (
                      <motion.div
                        initial={{ opacity: 0, x: -10, scale: 0.95 }}
                        animate={{ opacity: 1, x: 0, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.95 }}
                        transition={{ duration: 0.2 }}
                        className="absolute w-[450px] pointer-events-auto"
                        style={{ left: overlayState.x, top: overlayState.y }}
                      >
                        <div className={`glass-panel p-6 rounded-2xl border bg-[#0b0c10]/95 backdrop-blur-2xl shadow-[0_30px_60px_-15px_rgba(0,0,0,0.6)] ${activeCit.is_verified ? 'border-green-500/30 shadow-[0_0_40px_rgba(34,197,94,0.15)]' : 'border-red-500/30 shadow-[0_0_40px_rgba(239,68,68,0.15)]'}`}>
                          <div className="flex justify-between items-start mb-4">
                            <span className="text-xs uppercase tracking-wider font-bold text-slate-400 bg-white/5 px-2.5 py-1.5 rounded-md">Citation {overlayState.id}</span>
                            {activeCit.is_verified ? (
                              <span className="flex items-center gap-1.5 text-sm font-medium text-green-400 bg-green-400/10 px-3 py-1.5 rounded-full">
                                <svg xmlns="http://www.w3.org/-2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg> Verified
                              </span>
                            ) : (
                              <span className="flex items-center gap-1.5 text-sm font-medium text-red-400 bg-red-400/10 px-3 py-1.5 rounded-full">
                                <svg xmlns="http://www.w3.org/-2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg> Unverified
                              </span>
                            )}
                          </div>
                          
                          <p className="text-base text-slate-100 mb-5 leading-relaxed font-medium">"{activeCit.claim}"</p>
                          
                          {activeCit.source_title && (
                            <div className="text-sm text-slate-400 border-t border-white/10 pt-4 mt-4">
                              <span className="block font-semibold text-slate-500 mb-2">Source Material:</span>
                              <span className="italic text-slate-300">{activeCit.source_title}</span>
                            </div>
                          )}
                          
                          {activeCit.quote && (
                            <div className="mt-3 text-sm bg-black/60 p-3.5 rounded-xl border border-white/5 text-slate-400 italic leading-relaxed">
                              "{activeCit.quote}"
                            </div>
                          )}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>,
                document.body
              )}

            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </HoverContext.Provider>
  );
}

export default App;
