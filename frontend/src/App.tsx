import React, { useState } from "react";
import "./App.css";

type Card = {
  verdict?: string;
  score?: number;
  justification?: string;
  tags?: string[];
  facts?: string[];
  url?: string | null;
  bullets?: string[];
  evidence_quotes?: string[];
  score_note?: string;
};

type EvidenceChunk = {
  content: string;
  display_score?: number | null;
  score: number;
  source?: string;
  row?: number;
  start_index?: number;
};

export type PaperSignals = {
  max_score: number;
  mean_score: number;
  coverage: number;
  over_threshold: number;
  query_overlap_terms: string[];
  author_matched: boolean;
  venue_boost: number;
  recency_boost: number;
};

type Paper = {
  paper_id: string;
  title?: string | null;
  authors?: string[] | null;
  venue?: string | null;
  year?: number | null;
  url?: string | null;
  card?: Card;
  explanation?: string;
  evidence?: EvidenceChunk[];
  signals: PaperSignals;
};

type SearchResponse = {
  query?: string;
  refined_query?: string;
  top_score?: number | null;
  context?: string;
  papers: Paper[];
  hits?: any[];
  results?: unknown;
  analysis?: any;
  model?: string;
  reply_text?: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  papers?: Paper[];
  analysis?: any;
  refined_query?: string;
  model?: string;
  loading?: boolean;
};

const VerdictBand: React.FC<{
  label: string;
  text: string;
  tone?: "perfect" | "relevant" | "some";
}> = ({ label, text, tone = "relevant" }) => {
  const bg =
    tone === "perfect"
      ? "rgba(16,185,129,0.20)"
      : tone === "some"
      ? "rgba(234,179,8,0.18)"
      : "rgba(59,130,246,0.18)";
  const color =
    tone === "perfect" ? "#34d399" : tone === "some" ? "#fbbf24" : "#93c5fd";
  return (
    <div style={{ background: bg, borderRadius: 12, padding: 12, marginTop: 10 }}>
      <strong style={{ color }}>{label}</strong>
      <span> {text}</span>
    </div>
  );
};

const renderPaperCard = (p: Paper, showScores: boolean) => {
  const tone =
    p.card?.verdict === "Perfectly Relevant"
      ? "perfect"
      : p.card?.verdict === "Somewhat Relevant"
      ? "some"
      : "relevant";

  // Fallback: derive arXiv URL from paper_id if backend did not supply one.
  const arxivFromId = (pid?: string | null) => {
    if (!pid) return null;
    const cleaned = pid.replace(/\.0$/, "");
    return /^\d{4}\.\d{4,5}(v\d+)?$/.test(cleaned) ? `https://arxiv.org/abs/${cleaned}` : null;
  };
  const href = p.url || arxivFromId(p.paper_id);

  return (
    <div key={p.paper_id} className="paper-card">
      <h3>
        {href ? (
          <a href={href} target="_blank" rel="noreferrer">
            {p.title ?? "Untitled paper"}
          </a>
        ) : (
          p.title ?? "Untitled paper"
        )}
      </h3>

      <div className="meta">
        {p.authors?.length ? <span>{p.authors.join(", ")}</span> : null}
        {(p.venue || p.year) && (
          <span>
            {p.venue ? p.venue : ""}
            {p.venue && p.year ? " · " : ""}
            {p.year ?? ""}
          </span>
        )}
      </div>

      {p.card ? (
        <>
          {p.card.verdict && p.card.justification && (
            <VerdictBand label={p.card.verdict} text={p.card.justification} tone={tone} />
          )}
          {!p.card.verdict && p.card.justification && (
            <p style={{ marginTop: "0.5rem", marginBottom: "0.75rem", lineHeight: 1.5 }}>
              {p.card.justification}
            </p>
          )}

          {p.card?.bullets && p.card.bullets.length > 0 && (
            <ul style={{ marginTop: "0.35rem", marginLeft: "1.25rem" }}>
              {p.card.bullets.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          )}

          {p.card?.evidence_quotes && p.card.evidence_quotes.length > 0 && (
            <>
              <div style={{ marginTop: "0.35rem", fontSize: "0.9rem", opacity: 0.85 }}>LLM-picked evidence</div>
              <ul style={{ marginTop: "0.15rem", marginLeft: "1.25rem" }}>
                {p.card.evidence_quotes.map((q, i) => (
                  <li key={i}>“{q}”</li>
                ))}
              </ul>
            </>
          )}

          {p.card?.score_note && (
            <div style={{ marginTop: 8, fontSize: 12, opacity: 0.6 }}>
              Note: {p.card.score_note}
            </div>
          )}

          {!!p.card.facts?.length && (
            <details style={{ marginTop: 8 }}>
              <summary>Show Evidence</summary>
              <ul style={{ marginTop: 6, opacity: 0.9 }}>
                {p.card.facts.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      ) : (
        <>
          {p.explanation && <p className="why">{p.explanation}</p>}
          {p.evidence && p.evidence.length > 0 && (
            <details>
              <summary>Evidence</summary>
              <ul>
                {p.evidence.slice(0, 2).map((chunk, i) => (
                  <li key={i}>
                    “
                    {chunk.content.length > 220 ? chunk.content.slice(0, 220) + "…" : chunk.content}
                    ”
                    {showScores && chunk.display_score != null && (
                      <small>
                        {" "}
                        (score {chunk.display_score.toFixed(3)})
                      </small>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </div>
  );
};

function App() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      text: "Hi! I’m an Asta-inspired paper finder. Ask a research question to start, then follow up with clarifications or new angles.",
    },
  ]);
  const [k, setK] = useState<number>(5);
  const [showScores, setShowScores] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentSearchId, setCurrentSearchId] = useState<string | null>(null);
  const [currentAbort, setCurrentAbort] = useState<AbortController | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;

    if (currentAbort) {
      currentAbort.abort();
    }

    const searchId =
      (window.crypto && "randomUUID" in window.crypto
        ? (window.crypto as any).randomUUID()
        : Math.random().toString(36).slice(2)) as string;

    const userMsg: ChatMessage = { id: `user-${searchId}`, role: "user", text: trimmed };
    const pendingMsg: ChatMessage = { id: `asst-${searchId}`, role: "assistant", text: "Thinking…", loading: true };
    const historyPayload = [...messages, userMsg].map((m) => ({ role: m.role, content: m.text }));

    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setQuery("");
    setError(null);
    setLoading(true);
    setCurrentSearchId(searchId);

    const controller = new AbortController();
    setCurrentAbort(controller);

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmed,
          k,
          show_scores: showScores,
          search_id: searchId,
          history: historyPayload,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        if (res.status === 499) {
          throw new Error("cancelled");
        }
        throw new Error(`HTTP ${res.status}`);
      }

      const raw: any = await res.json();

      const isPaperArray = (x: unknown): x is Paper[] =>
        Array.isArray(x) && (x.length === 0 || typeof (x as any)[0]?.paper_id === "string");

      const normalizedPapers: Paper[] = isPaperArray(raw?.papers)
        ? raw.papers
        : isPaperArray(raw?.results?.papers)
        ? raw.results.papers
        : isPaperArray(raw?.results)
        ? raw.results
        : [];

      const normalized: SearchResponse = {
        query: raw?.results?.query ?? raw?.query,
        refined_query: raw?.results?.refined_query ?? raw?.refined_query,
        top_score: raw?.results?.top_score ?? raw?.top_score,
        context: raw?.results?.context ?? raw?.context,
        papers: normalizedPapers,
        hits: raw?.results?.hits ?? raw?.hits,
        results: raw?.results ?? raw,
        analysis: raw?.analysis ?? raw?.results?.analysis,
        model: raw?.model,
        reply_text: raw?.reply_text ?? raw?.results?.reply_text,
      };

      const replyText =
        normalized.reply_text ||
        `Here are ${normalizedPapers.length || "some"} papers for “${normalized.query ?? trimmed}”.`;

      const assistantMsg: ChatMessage = {
        id: pendingMsg.id,
        role: "assistant",
        text: replyText,
        papers: normalizedPapers,
        analysis: normalized.analysis,
        refined_query: normalized.refined_query,
        model: normalized.model,
        loading: false,
      };

      setMessages((prev) => prev.map((m) => (m.id === pendingMsg.id ? assistantMsg : m)));
    } catch (err: any) {
      if (err?.name === "AbortError" || err?.message === "cancelled") {
        console.log("Search cancelled");
        setMessages((prev) => prev.filter((m) => !m.loading));
        setError(null);
        return;
      }
      console.error("Search failed:", err);
      setMessages((prev) => prev.filter((m) => !m.loading));
      setError(err?.message ?? "Search failed. Check backend logs.");
    } finally {
      setLoading(false);
      setCurrentAbort(null);
      setCurrentSearchId(null);
    }
  };

  const handleCancel = async () => {
    if (!loading) return;

    if (currentAbort) {
      currentAbort.abort();
    }

    if (currentSearchId) {
      try {
        await fetch(`/api/search/${currentSearchId}/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
        });
      } catch (e) {
        console.warn("Backend cancel failed (probably already done):", e);
      }
    }

    setMessages((prev) => prev.filter((m) => !m.loading));
    setLoading(false);
  };

  return (
    <div className="App" style={{ maxWidth: 1100, margin: "0 auto", padding: "24px 18px" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", gap: 16 }}>
        <div>
          <h1 style={{ margin: 0 }}>Paper Finder</h1>
          <div style={{ opacity: 0.75 }}>Asta-style conversational search</div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <label htmlFor="k-range">
            Papers: <strong>{k}</strong>
          </label>
          <input
            id="k-range"
            type="range"
            min={1}
            max={20}
            step={1}
            value={k}
            onChange={(e) => setK(parseInt(e.target.value, 10))}
            aria-label="Number of papers"
            style={{ width: "180px" }}
          />
          <label htmlFor="show-scores" style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
            <input
              id="show-scores"
              type="checkbox"
              checked={showScores}
              onChange={(e) => setShowScores(e.target.checked)}
            />
            Show scores
          </label>
          {loading && (
            <button
              type="button"
              onClick={handleCancel}
              style={{
                padding: "8px 12px",
                borderRadius: 8,
                background: "#333",
                color: "#fff",
                border: "1px solid #555",
              }}
            >
              Cancel
            </button>
          )}
        </div>
      </header>

      {error && <div style={{ marginTop: 12, color: "#f66" }}>{error}</div>}

      <div
        className="chat-window"
        style={{
          marginTop: 18,
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 14,
          padding: "16px 14px",
          background: "rgba(255,255,255,0.02)",
          minHeight: "65vh",
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        {messages.map((msg) => (
          <div
            key={msg.id}
            style={{
              display: "flex",
              justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
            }}
          >
            <div
              style={{
                maxWidth: "82%",
                background: msg.role === "user" ? "rgba(59,130,246,0.18)" : "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 12,
                padding: "10px 12px",
              }}
            >
              <div style={{ opacity: 0.7, fontSize: 12, marginBottom: 4 }}>
                {msg.role === "user" ? "You" : "Assistant"}
                {msg.model ? ` · ${msg.model}` : ""}
              </div>
              <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.45 }}>{msg.text}</div>
              {msg.refined_query && msg.refined_query !== msg.text && (
                <div style={{ marginTop: 8, opacity: 0.7, fontSize: 12 }}>
                  Refined query: <code>{msg.refined_query}</code>
                </div>
              )}
              {msg.analysis && (
                <details style={{ marginTop: 10 }}>
                  <summary>Analyzer output</summary>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(msg.analysis, null, 2)}</pre>
                </details>
              )}
              {msg.papers && msg.papers.length > 0 && (
                <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 12 }}>
                  {msg.papers.map((p) => renderPaperCard(p, showScores))}
                </div>
              )}
              {msg.loading && <div style={{ marginTop: 8, opacity: 0.6 }}>Thinking…</div>}
            </div>
          </div>
        ))}
      </div>

      <form
        onSubmit={handleSubmit}
        style={{
          marginTop: 16,
          display: "flex",
          alignItems: "flex-end",
          gap: 10,
        }}
      >
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask for papers, e.g., 'agentic AI for robotics' or follow up with refinements…"
          aria-label="Chat input"
          rows={2}
          style={{
            flex: 1,
            padding: "10px 12px",
            fontSize: "1rem",
            borderRadius: 10,
            border: "1px solid #444",
            background: "rgba(255,255,255,0.04)",
            color: "#fff",
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "12px 18px",
            fontSize: "1rem",
            borderRadius: 10,
            background: loading ? "#333" : "#2563eb",
            color: "#fff",
            border: "none",
            minWidth: 120,
          }}
        >
          {loading ? "Searching…" : "Send"}
        </button>
      </form>
    </div>
  );
}

export default App;
