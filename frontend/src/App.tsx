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
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<SearchResponse | null>(null)

  async function handleSearch(query: string) {
    setLoading(true)
    setError(null)
    setData(null)
    try {
      const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ q: query, k: 5 }),
    });

      if (!res.ok) throw new Error(`HTTP error! ${res.status}`);
      const result = await res.json();
      setData(result);
    } catch (err) {
      console.error('Search failed:', err);
      setError('Search failed. Check backend logs or network tab.')
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (query.trim()) handleSearch(query)
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
        <button type="submit" className="search-button">
          Search
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {data && (
        <div className="results">
          <h2>Results</h2>
          <p><strong>Top Score:</strong> {data.top_score}</p>
          <ul>
            {data.hits.map((hit, idx) => (
              <li key={idx}>
                <p>{hit.content}</p>
                <small>
                  score={hit.score} | source={hit.source}
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
