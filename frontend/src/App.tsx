import React, { useState } from "react"; // <-- import React for React.FC typing
import "./App.css";

type Hit = {
  content: string;
  score: number | null;
  source?: string;
  row?: number;
  start_index?: number;
};

type Card = {
  verdict: string;       // "Perfectly Relevant" | "Relevant" | "Somewhat Relevant"
  score: number;         // 0..1
  justification: string; // one tidy sentence
  tags: string[];        // badges
  facts: string[];       // 0–2 short factual sentences
  url?: string | null;
};

type EvidenceChunk = {
  content: string;
  display_score: number | null;
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

  // NEW preferred block
  card?: Card;

  // Fallbacks
  explanation?: string;
  evidence?: EvidenceChunk[];

  signals: {
    max_score: number;
    mean_score: number;
    coverage: number;
    over_threshold: number;
    query_overlap_terms: string[];
    author_matched: boolean;
    venue_boost: number;
    recency_boost: number;
  };
};

type SearchResponse = {
  query?: string;
  top_score?: number | null;
  context?: string;
  papers: Paper[];
  results?: unknown;
  analysis?: any;
  model?: string;
};

const Tag: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <span
    style={{
      display: "inline-block",
      padding: "2px 8px",
      borderRadius: 8,
      background: "rgba(16,185,129,0.2)", // emerald-ish
      color: "rgb(167,243,208)",
      fontSize: 12,
      marginRight: 6,
      marginTop: 6,
    }}
  >
    {children}
  </span>
);

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

function App() {
  const [query, setQuery] = useState("");
  const [k, setK] = useState<number>(5);
  const [showScores, setShowScores] = useState(true);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults(null);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: query, k, show_scores: showScores }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
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
        top_score: raw?.results?.top_score ?? raw?.top_score,
        context: raw?.results?.context ?? raw?.context,
        papers: normalizedPapers,
        results: raw?.results ?? raw,
        analysis: raw?.analysis,
        model: raw?.model,
      };
      setResults(normalized);
    } catch (err: any) {
      console.error("Search failed:", err);
      setError(err?.message ?? "Search failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Paper Finder</h1>

      <form onSubmit={handleSubmit} className="controls">
        <div className="row" style={{ marginBottom: 10 }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for papers…"
            aria-label="Search query"
            style={{
              width: "480px",
              padding: "10px",
              fontSize: "1rem",
              marginRight: "10px",
              borderRadius: "6px",
              border: "1px solid #ccc",
            }}
          />
          <button
            type="submit"
            disabled={loading}
            style={{
              padding: "10px 20px",
              fontSize: "1rem",
              borderRadius: "6px",
            }}
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </div>

        <div className="row" style={{ alignItems: "center", gap: "16px" }}>
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
            style={{ width: "240px" }}
          />

          <label
            htmlFor="show-scores"
            style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}
          >
            <input
              id="show-scores"
              type="checkbox"
              checked={showScores}
              onChange={(e) => setShowScores(e.target.checked)}
            />
            Show scores
          </label>
        </div>
      </form>

      {results?.model && (
        <div style={{ marginTop: 10, opacity: 0.7, fontSize: 12 }}>
          Model: <code>{results.model}</code>
        </div>
      )}
      {!!results?.analysis && (
        <details style={{ marginTop: 10 }}>
          <summary>Analyzer output</summary>
          <pre style={{ whiteSpace: "pre-wrap" }}>
            {JSON.stringify(results.analysis, null, 2)}
          </pre>
        </details>
      )}

      {error && <div style={{ marginTop: 12, color: "#f66" }}>{error}</div>}

      {/* Results */}
      {results?.papers?.length ? (
        <div className="results" style={{ marginTop: 20 }}>
          {results.papers.map((p) => {
            const tone =
              p.card?.verdict === "Perfectly Relevant"
                ? "perfect"
                : p.card?.verdict === "Somewhat Relevant"
                ? "some"
                : "relevant";

            return (
              <div key={p.paper_id} className="paper-card">
                <h3>
                  {p.url ? (
                    <a href={p.url} target="_blank" rel="noreferrer">
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

                {/* Preferred Asta-style card */}
                {p.card ? (
                  <>
                    <VerdictBand
                      label={p.card.verdict}
                      text={p.card.justification}
                      tone={tone as "perfect" | "relevant" | "some"}
                    />
                    {p.card.tags?.length ? (
                      <div
                        style={{
                          display: "flex",
                          flexWrap: "wrap",
                          gap: 6,
                          marginTop: 6,
                        }}
                      >
                        {p.card.tags.map((t) => (
                          <Tag key={t}>{t}</Tag>
                        ))}
                      </div>
                    ) : null}
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
                  // Fallback to old fields if card not present
                  <>
                    {p.explanation && <p className="why">{p.explanation}</p>}
                    {p.evidence && p.evidence.length > 0 && (
                      <details>
                        <summary>Evidence</summary>
                        <ul>
                          {p.evidence.slice(0, 2).map((chunk, i) => (
                            <li key={i}>
                              “
                              {chunk.content.length > 220
                                ? chunk.content.slice(0, 220) + "…"
                                : chunk.content}
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
          })}
        </div>
      ) : null}
    </div>
  );
}

export default App;
