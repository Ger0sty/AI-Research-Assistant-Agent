import { useState } from 'react'
import './App.css'

type Hit = {
  content: string
  score: number | null
  source?: string
  row?: number
  start_index?: number
}

type EvidenceChunk = {
  content: string;
  display_score: number | null; // or undefined if you didn’t request scores
  score: number;                // internal numeric for sorting if you want
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
  explanation: string;      // human-readable paragraph from backend
  evidence: EvidenceChunk[]; // the chunks for this paper (optional to show)
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
  query: string;
  top_score: number | null;
  papers: Paper[];          // <-- use this
  // hits: EvidenceChunk[]; // <-- ignore or remove from UI
  context: string;
};


function App() {
  const [query, setQuery] = useState('')
  const [k, setK] = useState<number>(5)                // NEW: user-controlled #results
  const [showScores, setShowScores] = useState(true)   // NEW: user-controlled scores toggle
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSearch(query: string, k: number, showScores: boolean) {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, k, show_scores: showScores }), // pass both
      })

      if (!res.ok) throw new Error(`HTTP error! ${res.status}`)
      const result = await res.json()
      setData(result)
    } catch (err) {
      console.error('Search failed:', err)
      setError('Search failed. Check backend logs or network tab.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: query, k, show_scores: showScores }),
      });
      const data: SearchResponse = await res.json();
      setResults(data);
    } catch (err) {
      console.error("Search failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Paper Finder</h1>

      <form onSubmit={handleSubmit} className="controls">
        <div className="row" style={{ marginBottom: 10 }}>
          {/* Larger input box */}
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
          {/* Slider for number of papers */}
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

          {/* Checkbox to show/hide scores */}
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

      {/* Results */}
      {results && (
        <div className="results" style={{ marginTop: 20 }}>
          {results.papers?.map((p) => (
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

              <p className="why">{p.explanation}</p>

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
                          <small> (score {chunk.display_score.toFixed(3)})</small>
                        )}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App
