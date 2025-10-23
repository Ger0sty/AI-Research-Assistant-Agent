import { useState } from 'react'
import './App.css'

type Hit = {
  content: string
  score: number | null
  source?: string
  row?: number
  start_index?: number
}

type SearchResponse = {
  query: string
  top_score: number | null
  hits: Hit[]
  context: string
}

function App() {
  const [query, setQuery] = useState('')
  const [k, setK] = useState<number>(5)                // NEW: user-controlled #results
  const [showScores, setShowScores] = useState(true)   // NEW: user-controlled scores toggle
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

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) handleSearch(query, k, showScores)
  }

  return (
    <div className="app">
      <h1>Paper Finder</h1>

      <form onSubmit={handleSubmit} className="search-form">
        <input
          type="text"
          placeholder="Search for papers..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="search-input"
        />

        {/* NEW: Result count slider + number input */}
        <div className="control-row">
          <label htmlFor="k-range" className="control-label">
            Results: <strong>{k}</strong>
          </label>
          <input
            id="k-range"
            type="range"
            min={1}
            max={25}
            step={1}
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
          />
          <input
            type="number"
            min={1}
            max={100}
            value={k}
            onChange={(e) => setK(Number(e.target.value) || 1)}
            className="k-number"
            aria-label="Results count"
          />
        </div>

        {/* NEW: Show scores checkbox */}
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showScores}
            onChange={(e) => setShowScores(e.target.checked)}
          />
          Show similarity scores
        </label>

        <button type="submit" className="search-button" disabled={loading}>
          {loading ? 'Searching…' : 'Search'}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {data && (
        <div className="results">
          <h2>Results</h2>
          {showScores && (
            <p>
              <strong>Top Score:</strong> {data.top_score}
            </p>
          )}
          <ul>
            {data.hits.map((hit, idx) => (
              <li key={idx}>
                <p>{hit.content}</p>
                <small>
                  {showScores ? (
                    <>
                      score={hit.score} | source={hit.source}
                    </>
                  ) : (
                    <>source={hit.source}</>
                  )}
                </small>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default App
