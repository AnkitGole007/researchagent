import { useState, useEffect, useRef } from "react";
import { streamSearch } from "./api";

// Pipeline stages — labels for the loader and the "How it works" panel.
// (Detail strings are also streamed live from the backend per stage.)
// Re-derived from this repo's actual post-LanceDB pipeline (backend/runner.py
// STAGE_NAMES), not copied from the reference repo's stale 6-stage BM25/FAISS
// loader — see docs/PLAN.md's F-01..F-06 "Critical mismatch" note.
const STEPS = [
  { n:"1", name:"Fetch & Filter", d:"385K+ papers → date window", t:"0.5s" },
  { n:"2", name:"Hybrid Retrieval", d:"QIL + LanceDB FTS/vector RRF", t:"1.5s" },
  { n:"3", name:"Semantic Rerank", d:"SPECTER2 asymmetric retrieval", t:"2.0s" },
  { n:"4", name:"Precision Rerank", d:"CrossEncoder + highlights", t:"2.5s" },
  { n:"5", name:"Classify", d:"Primary / secondary / off-topic", t:"0.5s" },
  { n:"6", name:"Impact Score", d:"Moneyball citation engine", t:"6.0s" },
  { n:"7", name:"Summarize", d:"Plain English for top N", t:"3.0s" },
];

// Chat models offered per provider (mirrors the Streamlit app). The first
// entry is the default. "free" needs no key or model.
const MODELS = {
  openai: ["gpt-5.2", "gpt-5", "gpt-5-mini", "gpt-4.1-mini", "gpt-4o-mini", "o1"],
  gemini: ["gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"],
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
};
const KEY_HELP = {
  openai: "Used in memory for this session only — never stored or sent anywhere but OpenAI.",
  gemini: "Get a key from Google AI Studio. Kept in memory for this session only.",
  groq: "Free at console.groq.com. Kept in memory for this session only.",
};

const Chev = ({open,s=14}) => <svg width={s} height={s} viewBox="0 0 16 16" fill="none" style={{transform:open?"rotate(180deg)":"none",transition:"transform 0.2s"}}><path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>;
const XIcon = () => <svg width="18" height="18" viewBox="0 0 20 20" fill="none"><path d="M5 5L15 15M15 5L5 15" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>;
const MenuIcon = () => <svg width="20" height="20" viewBox="0 0 20 20" fill="none"><path d="M3 6H17M3 10H17M3 14H17" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>;
const Ext = () => <svg width="10" height="10" viewBox="0 0 12 12" fill="none" style={{marginLeft:3,opacity:0.4}}><path d="M4.5 1.5H2C1.72 1.5 1.5 1.72 1.5 2V10C1.5 10.28 1.72 10.5 2 10.5H10C10.28 10.5 10.5 10.28 10.5 10V7.5M7 1.5H10.5V5M10.5 1.5L5.5 6.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round"/></svg>;

function Card({ p, i }) {
  const [open, setOpen] = useState(false);
  return (
    <div onClick={() => setOpen(!open)} style={{
      padding:"18px 20px", marginBottom:10, cursor:"pointer",
      background:"var(--card)", borderRadius:14, border:"1px solid var(--line)",
      opacity:0, animation:`up 0.35s ease ${i*0.06}s forwards`,
      transition:"box-shadow 0.2s",
    }}
    onMouseEnter={e => e.currentTarget.style.boxShadow="0 2px 12px rgba(43,107,96,0.06)"}
    onMouseLeave={e => e.currentTarget.style.boxShadow="none"}>
      <div style={{ display:"flex", gap:14, alignItems:"flex-start" }}>
        <div style={{
          width:32, height:32, borderRadius:8,
          background: p.tooNew ? "var(--warm)" : "var(--teal)",
          color:"#fff", display:"flex", alignItems:"center", justifyContent:"center",
          fontFamily:"var(--mono)", fontSize:13, fontWeight:600, flexShrink:0, marginTop:1,
        }}>{p.rank}</div>
        <div style={{ flex:1, minWidth:0 }}>
          <div style={{ display:"flex", gap:8, marginBottom:4, flexWrap:"wrap", alignItems:"center" }}>
            <span style={{
              fontSize:9, fontWeight:700, textTransform:"uppercase", letterSpacing:"0.08em",
              color: p.focus==="primary" ? "var(--teal)" : "var(--dim)",
              background: p.focus==="primary" ? "var(--teal-soft)" : "var(--line)",
              padding:"2px 7px", borderRadius:4,
            }}>{p.focus}</span>
            {p.venue && <span style={{ fontSize:10, color:"var(--dim)", fontWeight:500 }}>{p.venue}</span>}
          </div>
          <h3 style={{ fontSize:14, fontWeight:600, color:"var(--fg)", margin:"2px 0 5px", lineHeight:1.4, fontFamily:"var(--serif)" }}>{p.title}</h3>
          <div style={{ fontSize:11, color:"var(--dim)" }}>{p.authors.slice(0,3).join(", ")}{p.authors.length>3?" et al.":""}</div>
        </div>
        <div style={{ textAlign:"right", flexShrink:0 }}>
          <div style={{ fontSize:20, fontWeight:700, fontFamily:"var(--mono)", color:p.tooNew?"var(--warm)":"var(--teal)" }}>{p.tooNew?"—":p.score}</div>
          <div style={{ fontSize:8, color:"var(--dim)", textTransform:"uppercase", letterSpacing:"0.06em", marginTop:2 }}>{p.tooNew?"new":"impact"}</div>
        </div>
      </div>
      {open && (
        <div style={{ marginTop:14, paddingTop:14, borderTop:"1px solid var(--line)", animation:"up 0.2s ease" }}>
          {p.summary && (
            <p style={{ fontSize:13, lineHeight:1.7, color:"var(--fg)", margin:"0 0 12px", fontWeight:500 }}>{p.summary}</p>
          )}
          <p style={{ fontSize:13, lineHeight:1.7, color:"var(--fg)", margin:"0 0 12px", opacity:0.85 }}>{p.abstract}</p>
          <div style={{ marginBottom:12 }}>
            {p.why.map((w,j) => <div key={j} style={{ fontSize:11, color:"var(--fg)", marginBottom:3, paddingLeft:10, borderLeft:"2px solid var(--teal-soft)", opacity:0.75, lineHeight:1.5 }}>{w}</div>)}
          </div>
          <a href={p.arxiv} target="_blank" rel="noopener noreferrer" onClick={e => e.stopPropagation()} style={{ fontSize:11, color:"var(--teal)", textDecoration:"none", fontWeight:600, display:"inline-flex", alignItems:"center" }}>Open on arXiv <Ext/></a>
        </div>
      )}
    </div>
  );
}

function Drawer({ open, onClose, title, children, width=300 }) {
  if (!open) return null;
  return <>
    <div onClick={onClose} style={{ position:"fixed", inset:0, background:"rgba(28,48,65,0.18)", zIndex:90, animation:"fadeIn 0.2s ease" }}/>
    <div style={{ position:"fixed", top:0, right:0, bottom:0, width, background:"var(--cream)", zIndex:100, borderLeft:"1px solid var(--line)", animation:"slideIn 0.25s ease", display:"flex", flexDirection:"column", overflowY:"auto" }}>
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"18px 22px", borderBottom:"1px solid var(--line)" }}>
        <span style={{ fontFamily:"var(--serif)", fontSize:17, fontWeight:600, color:"var(--fg)" }}>{title}</span>
        <button onClick={onClose} style={{ background:"none", border:"none", cursor:"pointer", color:"var(--dim)", padding:2 }}><XIcon/></button>
      </div>
      <div style={{ padding:"20px 22px", flex:1 }}>{children}</div>
    </div>
  </>;
}

export default function App() {
  const [view, setView] = useState("home");
  const [q, setQ] = useState("");
  const [menu, setMenu] = useState(false);
  const [about, setAbout] = useState(false);
  const [pipeline, setPipeline] = useState(false);
  const [showExclude, setShowExclude] = useState(false);
  const [step, setStep] = useState(0);
  const [provider, setProvider] = useState("free");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [range, setRange] = useState("Last Month");
  const [exclude, setExclude] = useState("");
  const [papers, setPapers] = useState([]);
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);
  const ref = useRef(null);

  const needsKey = provider !== "free";
  const canSearch = q.trim() && (!needsKey || apiKey.trim());

  const changeProvider = (next) => {
    setProvider(next);
    setModel(next === "free" ? "" : MODELS[next][0]);
  };

  const search = () => {
    if (!canSearch) return;
    setError(null);
    setView("loading");
    setStep(0);
    streamSearch(
      {
        query: q, exclude, date_range: range, provider, top_n: 5,
        api_key: needsKey ? apiKey : null,
        model: needsKey ? (model || MODELS[provider][0]) : null,
      },
      {
        onStage: (ev) => { if (ev.status === "done") setStep(ev.index + 1); },
        onDone: (res) => {
          setPapers(res.papers);
          setMeta(res);
          setTimeout(() => setView("results"), 400);
        },
        onError: (msg) => { setError(msg); setView("home"); },
      }
    );
  };
  const reset = () => { setView("home"); setQ(""); setPipeline(false); setStep(0); setError(null); };

  const primaryCount = meta ? meta.primary_count : papers.filter(p=>p.focus==="primary").length;
  const secondaryCount = meta ? meta.secondary_count : papers.filter(p=>p.focus==="secondary").length;

  const exportResults = () => {
    let md = `# Research Agent — Results\n\n`;
    md += `**Query:** ${q}\n**Date range:** ${range}\n**Provider:** ${provider}\n\n---\n\n`;
    papers.forEach(p => {
      md += `## #${p.rank}: ${p.title}\n\n`;
      md += `- **Authors:** ${p.authors.join(", ")}\n`;
      md += `- **Venue:** ${p.venue || "N/A"}\n`;
      md += `- **Impact score:** ${p.tooNew ? "Too new to rate" : p.score}\n`;
      md += `- **Focus:** ${p.focus}\n`;
      md += `- **Relevance:** ${(p.rel*100).toFixed(0)}%\n`;
      md += `- **arXiv:** ${p.arxiv}\n\n`;
      if (p.summary) md += `**Summary:** ${p.summary}\n\n`;
      md += `**Abstract:** ${p.abstract}\n\n`;
      md += `**Why this ranking:**\n`;
      p.why.forEach(w => { md += `- ${w}\n`; });
      md += `\n---\n\n`;
    });
    const blob = new Blob([md], { type:"text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `research_agent_results.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => { if (view==="home" && ref.current) ref.current.focus(); }, [view]);

  return (
    <div style={{ minHeight:"100vh", background:"var(--cream)", color:"var(--fg)", fontFamily:"var(--sans)" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
        :root {
          --cream: #FAF7F2;
          --card: #FFFFFF;
          --fg: #1C3041;
          --dim: #7A8A95;
          --line: #E8E4DD;
          --teal: #2B6B60;
          --teal-dark: #1E4F47;
          --teal-soft: #E2EEEC;
          --teal-grad-1: #1C3041;
          --teal-grad-2: #2B6B60;
          --teal-grad-3: #4A9B8E;
          --warm: #B8860B;
          --serif: 'Source Serif 4', Georgia, serif;
          --sans: 'DM Sans', system-ui, sans-serif;
          --mono: 'JetBrains Mono', monospace;
          --max: 620px;
        }
        @keyframes up { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes fadeIn { from{opacity:0} to{opacity:1} }
        @keyframes slideIn { from{transform:translateX(100%)} to{transform:translateX(0)} }
        @keyframes pulse { 0%,100%{opacity:.3} 50%{opacity:1} }
        *{box-sizing:border-box;margin:0;padding:0}
        input:focus,textarea:focus,select:focus{outline:none}
        ::selection{background:var(--teal-soft);color:var(--teal-dark)}
        textarea{font-family:var(--sans)}
      `}</style>

      {/* ── NAV ── */}
      <nav style={{
        position:"sticky", top:0, zIndex:50, height:54,
        display:"flex", alignItems:"center", justifyContent:"space-between",
        padding:"0 22px", background:"rgba(250,247,242,0.92)", backdropFilter:"blur(10px)",
        borderBottom:"1px solid var(--line)",
      }}>
        <div onClick={reset} style={{ cursor:"pointer", display:"flex", alignItems:"center", gap:9 }}>
          <div style={{
            width:26, height:26, borderRadius:6,
            background:"var(--teal)", color:"#fff",
            display:"flex", alignItems:"center", justifyContent:"center",
            fontFamily:"var(--serif)", fontSize:12, fontWeight:700,
          }}>b²</div>
          <span style={{ fontFamily:"var(--serif)", fontSize:16, fontWeight:600, color:"var(--fg)" }}>Research Agent</span>
        </div>
        <button onClick={() => setMenu(true)} style={{ background:"none", border:"none", cursor:"pointer", color:"var(--fg)", padding:4 }}><MenuIcon/></button>
      </nav>

      {/* ── MENU DRAWER ── */}
      <Drawer open={menu} onClose={() => setMenu(false)} title="Menu">
        <div style={{ display:"flex", flexDirection:"column", gap:0 }}>
          {[
            { label:"About this project", action:() => { setMenu(false); setAbout(true); } },
          ].map((item,i) => (
            <button key={i} onClick={item.action} style={{
              background:"none", border:"none", padding:"14px 0", fontSize:14, color:"var(--fg)",
              cursor:"pointer", textAlign:"left", fontFamily:"var(--sans)", fontWeight:500,
              borderBottom:"1px solid var(--line)",
            }}>{item.label}</button>
          ))}
          <a href="https://github.com/benevolentbandwidth" target="_blank" rel="noopener noreferrer" style={{
            padding:"14px 0", fontSize:14, color:"var(--fg)", textDecoration:"none", fontWeight:500,
            borderBottom:"1px solid var(--line)", display:"flex", alignItems:"center",
          }}>GitHub <Ext/></a>
          <a href="https://benevolentbandwidth.org" target="_blank" rel="noopener noreferrer" style={{
            padding:"14px 0", fontSize:14, color:"var(--fg)", textDecoration:"none", fontWeight:500,
            borderBottom:"1px solid var(--line)", display:"flex", alignItems:"center",
          }}>Foundation website <Ext/></a>
        </div>
        <div style={{ position:"absolute", bottom:22, left:22, right:22, fontSize:11, color:"var(--dim)", lineHeight:1.6 }}>
          The Benevolent Bandwidth Foundation<br/>AI for Humanity · Open Source · MIT License
        </div>
      </Drawer>

      {/* ── ABOUT DRAWER ── */}
      <Drawer open={about} onClose={() => setAbout(false)} title="About" width={340}>
        <p style={{ fontSize:14, lineHeight:1.7, color:"var(--fg)", marginBottom:20 }}>
          Research Agent searches a pre-built corpus of <strong>385,000+ papers</strong> using a hybrid LanceDB retrieval pipeline, then ranks them by predicted citation impact and explains each one in plain English.
        </p>

        <div style={{ background:"var(--teal-soft)", borderRadius:12, padding:16, marginBottom:22 }}>
          <div style={{ fontSize:10, fontWeight:700, color:"var(--teal)", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:8 }}>How ranking works</div>
          <p style={{ fontSize:12, lineHeight:1.6, color:"var(--fg)", marginBottom:0, opacity:0.85 }}>
            The Moneyball engine uses <strong>84% hard data</strong> (author citation velocity from Semantic Scholar) and <strong>16% soft data</strong> (LLM content analysis) to predict 1-year citation impact — a 6x improvement over LLM-only ranking.
          </p>
        </div>

        <div style={{ fontSize:10, fontWeight:700, color:"var(--dim)", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:10 }}>Search pipeline</div>
        <div style={{ marginBottom:22 }}>
          {STEPS.map((s,i) => (
            <div key={i} style={{ display:"flex", gap:8, padding:"5px 0", fontSize:11, alignItems:"baseline" }}>
              <span style={{ fontFamily:"var(--mono)", fontWeight:600, color:"var(--teal)", width:12 }}>{s.n}</span>
              <span style={{ fontWeight:600, width:88, flexShrink:0, color:"var(--fg)" }}>{s.name}</span>
              <span style={{ color:"var(--dim)" }}>{s.d}</span>
            </div>
          ))}
        </div>

        <div style={{ fontSize:10, fontWeight:700, color:"var(--dim)", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:10 }}>Team</div>
        <p style={{ fontSize:12, color:"var(--fg)", lineHeight:1.7, marginBottom:20, opacity:0.85 }}>
          Aaron Eldring · Ankit Gole · Elena (Wenya Wei) · Ismail Ozberk · Janet Wang · Menelaos · Pavithra Kumar · Shagun Saboo · Shan Ali Shah Sayed · Zulfa Mohamed
        </p>

        <div style={{ fontSize:10, fontWeight:700, color:"var(--dim)", textTransform:"uppercase", letterSpacing:"0.08em", marginBottom:10 }}>Our principles</div>
        <p style={{ fontSize:12, color:"var(--fg)", lineHeight:1.7, opacity:0.85 }}>
          Useful tools · Public benefit · Open by default · Privacy first · Humility
        </p>
      </Drawer>

      {/* ═══════ HOME ═══════ */}
      {view === "home" && (
        <div style={{ animation:"fadeIn 0.5s ease" }}>
          {/* Hero gradient banner */}
          <div style={{
            background:`linear-gradient(135deg, var(--teal-grad-1) 0%, var(--teal-grad-2) 55%, var(--teal-grad-3) 100%)`,
            padding:"64px 24px 56px", textAlign:"center",
          }}>
            <h1 style={{
              fontFamily:"var(--serif)", fontSize:"clamp(28px, 5vw, 42px)",
              fontWeight:700, lineHeight:1.2, color:"#FFFFFF", marginBottom:10,
            }}>
              Find the research that matters
            </h1>
            <p style={{ fontSize:15, color:"rgba(255,255,255,0.75)", maxWidth:480, margin:"0 auto", lineHeight:1.5 }}>
              385,000+ AI papers, ranked by predicted citation impact.
            </p>
          </div>

          {/* Search area */}
          <div style={{ maxWidth:"var(--max)", margin:"-28px auto 0", padding:"0 20px", position:"relative", zIndex:2 }}>
            <div style={{
              background:"var(--card)", borderRadius:16, border:"1px solid var(--line)",
              overflow:"hidden", boxShadow:"0 4px 24px rgba(28,48,65,0.06)",
            }}>
              <textarea
                ref={ref} value={q} onChange={e => setQ(e.target.value)}
                placeholder="Describe what you're looking for..."
                rows={3}
                onKeyDown={e => { if(e.key==="Enter" && !e.shiftKey){e.preventDefault();search();} }}
                style={{
                  width:"100%", border:"none", background:"transparent",
                  padding:"18px 18px 6px", fontSize:15, color:"var(--fg)",
                  fontFamily:"var(--sans)", resize:"none", lineHeight:1.5,
                }}
              />
              {/* Inline settings row */}
              <div style={{ padding:"8px 14px", display:"flex", gap:8, flexWrap:"wrap", alignItems:"center", borderTop:"1px solid var(--line)" }}>
                <select value={range} onChange={e=>setRange(e.target.value)} style={{
                  padding:"6px 10px", borderRadius:8, border:"1px solid var(--line)",
                  fontSize:12, fontFamily:"var(--sans)", background:"var(--cream)", color:"var(--fg)", cursor:"pointer",
                }}><option>Last 3 Days</option><option>Last Week</option><option>Last Month</option></select>
                <select value={provider} onChange={e=>changeProvider(e.target.value)} style={{
                  padding:"6px 10px", borderRadius:8, border:"1px solid var(--line)",
                  fontSize:12, fontFamily:"var(--sans)", background:"var(--cream)", color:"var(--fg)", cursor:"pointer",
                }}><option value="free">Free Local</option><option value="openai">OpenAI</option><option value="gemini">Gemini 3</option><option value="groq">Groq</option></select>
                <button onClick={() => setShowExclude(!showExclude)} style={{
                  background:"none", border:"none", cursor:"pointer",
                  fontSize:11, color:"var(--dim)", fontFamily:"var(--sans)", padding:"6px 2px",
                  display:"flex", alignItems:"center", gap:3,
                }}>Exclude topics <Chev open={showExclude} s={10}/></button>
                <button onClick={search} disabled={!canSearch} style={{
                  marginLeft:"auto",
                  background:canSearch?"var(--teal)":"var(--line)",
                  color:canSearch?"#fff":"var(--dim)",
                  border:"none", borderRadius:8, padding:"8px 20px",
                  fontSize:13, fontWeight:600, cursor:canSearch?"pointer":"default",
                  fontFamily:"var(--sans)", transition:"all 0.2s",
                }}
                onMouseEnter={e => { if(canSearch) e.currentTarget.style.background="var(--teal-dark)"; }}
                onMouseLeave={e => { if(canSearch) e.currentTarget.style.background="var(--teal)"; }}>
                  Search
                </button>
              </div>
              {needsKey && (
                <div style={{ padding:"0 14px 12px", display:"flex", flexDirection:"column", gap:7, animation:"up 0.2s ease" }}>
                  <div style={{ display:"flex", gap:8 }}>
                    <input value={apiKey} onChange={e=>setApiKey(e.target.value)} type="password"
                      placeholder={`${provider==="openai"?"OpenAI":provider==="gemini"?"Gemini":"Groq"} API key`}
                      style={{ flex:1, padding:"8px 10px", borderRadius:8, border:"1px solid var(--line)", fontSize:12, fontFamily:"var(--sans)", background:"var(--cream)", color:"var(--fg)" }}
                    />
                    <select value={model} onChange={e=>setModel(e.target.value)} style={{
                      padding:"8px 10px", borderRadius:8, border:"1px solid var(--line)",
                      fontSize:12, fontFamily:"var(--sans)", background:"var(--cream)", color:"var(--fg)", cursor:"pointer",
                    }}>
                      {MODELS[provider].map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
                  </div>
                  <div style={{ fontSize:10.5, color:"var(--dim)", lineHeight:1.45 }}>🔒 {KEY_HELP[provider]}</div>
                </div>
              )}
              {showExclude && (
                <div style={{ padding:"0 14px 12px", animation:"up 0.2s ease" }}>
                  <input value={exclude} onChange={e=>setExclude(e.target.value)}
                    placeholder="e.g. math reasoning, scaling laws, generic LLM papers..."
                    style={{ width:"100%", padding:"8px 10px", borderRadius:8, border:"1px solid var(--line)", fontSize:12, fontFamily:"var(--sans)", background:"var(--cream)", color:"var(--fg)" }}
                  />
                </div>
              )}
            </div>

            {error && (
              <div style={{ marginTop:16, padding:"10px 14px", borderRadius:10, background:"#FBEAEA", border:"1px solid #E9C4C4", color:"#8A2B2B", fontSize:12.5, lineHeight:1.5 }}>
                {error}
              </div>
            )}

            <div style={{ textAlign:"center", marginTop:36, paddingBottom:48 }}>
              <p style={{ fontSize:12, color:"var(--dim)", marginBottom:10 }}>
                Free · Open source · No sign-up · No tracking
              </p>
              <div style={{ display:"inline-flex", gap:6, flexWrap:"wrap", justifyContent:"center" }}>
                {["arXiv","Semantic Scholar","Moneyball Engine"].map((t,i) => (
                  <span key={i} style={{ fontSize:10, color:"var(--teal)", background:"var(--teal-soft)", padding:"3px 10px", borderRadius:20, fontWeight:500 }}>{t}</span>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══════ LOADING ═══════ */}
      {view === "loading" && (
        <div style={{ maxWidth:"var(--max)", margin:"0 auto", padding:"64px 20px 0", animation:"fadeIn 0.3s ease" }}>
          <div style={{ textAlign:"center", marginBottom:36 }}>
            <div style={{ fontFamily:"var(--serif)", fontSize:20, fontWeight:600, marginBottom:4 }}>Searching...</div>
            <div style={{ fontSize:13, color:"var(--dim)", maxWidth:400, margin:"0 auto" }}>{q}</div>
          </div>
          <div style={{ maxWidth:380, margin:"0 auto" }}>
            {STEPS.map((s,i) => (
              <div key={i} style={{
                display:"flex", alignItems:"center", gap:10, padding:"9px 0",
                opacity:i<step?1:i===step?0.5:0.15, transition:"opacity 0.3s",
              }}>
                <div style={{
                  width:26, height:26, borderRadius:"50%", flexShrink:0,
                  background:i<step?"var(--teal)":"var(--line)",
                  color:i<step?"#fff":"var(--dim)",
                  display:"flex", alignItems:"center", justifyContent:"center",
                  fontSize:11, fontFamily:"var(--mono)", fontWeight:600, transition:"all 0.3s",
                }}>{i<step?"✓":s.n}</div>
                <div style={{ flex:1 }}>
                  <span style={{ fontSize:13, fontWeight:600, color:"var(--fg)" }}>{s.name}</span>
                  <span style={{ fontSize:11, color:"var(--dim)", marginLeft:8 }}>{s.d}</span>
                </div>
                {i===step && <div style={{ width:6, height:6, borderRadius:"50%", background:"var(--teal)", animation:"pulse 1s infinite" }}/>}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ═══════ RESULTS ═══════ */}
      {view === "results" && (
        <div style={{ maxWidth:"var(--max)", margin:"0 auto", padding:"24px 20px 80px", animation:"fadeIn 0.3s ease" }}>
          <div style={{ marginBottom:20, paddingBottom:16, borderBottom:"1px solid var(--line)" }}>
            <div style={{ fontSize:12, color:"var(--dim)", marginBottom:4, textTransform:"uppercase", letterSpacing:"0.06em", fontWeight:600 }}>Results for</div>
            <div style={{ fontSize:17, fontFamily:"var(--serif)", fontWeight:600, lineHeight:1.3, marginBottom:8 }}>{q}</div>
            <div style={{ fontSize:11, color:"var(--dim)", display:"flex", gap:8, flexWrap:"wrap", alignItems:"center" }}>
              <span>{range}</span>
              <span style={{opacity:0.3}}>·</span>
              <span>{primaryCount} primary, {secondaryCount} secondary</span>
              <span style={{opacity:0.3}}>·</span>
              <span>ranked by impact</span>
              <span style={{ marginLeft:"auto" }}>
                <button onClick={exportResults} style={{
                  background:"none", border:"1px solid var(--line)", borderRadius:8,
                  padding:"5px 12px", fontSize:10, color:"var(--dim)", cursor:"pointer",
                  fontFamily:"var(--sans)", fontWeight:600,
                }}>↓ Export</button>
              </span>
            </div>
          </div>

          {papers.map((p,i) => <Card key={i} p={p} i={i}/>)}

          {/* Pipeline dropdown */}
          <div style={{ marginTop:16 }}>
            <button onClick={() => setPipeline(!pipeline)} style={{
              background:"none", border:"none", cursor:"pointer", fontFamily:"var(--sans)",
              fontSize:12, color:"var(--dim)", display:"flex", alignItems:"center", gap:5, padding:"8px 0", fontWeight:500,
            }}>
              How these results were found <Chev open={pipeline} s={12}/>
            </button>
            {pipeline && (
              <div style={{ padding:"12px 0", animation:"up 0.2s ease" }}>
                {STEPS.map((s,i) => (
                  <div key={i} style={{ display:"flex", gap:8, padding:"4px 0", fontSize:11, color:"var(--dim)" }}>
                    <span style={{ fontFamily:"var(--mono)", fontWeight:600, color:"var(--teal)", width:12 }}>{s.n}</span>
                    <span style={{ fontWeight:600, color:"var(--fg)", width:88 }}>{s.name}</span>
                    <span>{s.d}</span>
                    <span style={{ marginLeft:"auto", fontFamily:"var(--mono)" }}>{s.t}</span>
                  </div>
                ))}
                <div style={{ marginTop:8, fontSize:11, color:"var(--teal)", background:"var(--teal-soft)", padding:"8px 12px", borderRadius:8, fontWeight:500 }}>
                  Total: {meta ? meta.total_seconds : "—"}s · {provider==="openai"?"OpenAI":provider==="gemini"?"Gemini":provider==="groq"?"Groq":"Local"}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
