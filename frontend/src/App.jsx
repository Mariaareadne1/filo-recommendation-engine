import { useState } from 'react'
import SearchBar from './components/SearchBar'
import VenueCard from './components/VenueCard'

const API_URL = 'http://localhost:8000/recommend'

/** Suggestions shown before the first search to hint at what filo understands. */
const SUGGESTIONS = [
  'romantic wine bar with candlelight',
  'late night ramen after drinks',
  'rooftop bar with skyline views',
  'cozy speakeasy for a first date',
]

/**
 * filo root component.
 *
 * Owns the search query, result list, loading and error state. Posts the query
 * to the FastAPI `/recommend` endpoint and renders the ranked venues as cards.
 */
export default function App() {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastQuery, setLastQuery] = useState('')
  const [searched, setSearched] = useState(false)

  /**
   * Fetch recommendations for a query from the backend.
   * @param {string} query
   */
  const runSearch = async (query) => {
    setLoading(true)
    setError('')
    setLastQuery(query)
    try {
      const res = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: 8 }),
      })
      if (!res.ok) throw new Error(`Request failed (${res.status})`)
      const data = await res.json()
      setResults(data)
      setSearched(true)
    } catch (err) {
      setError(
        'Could not reach filo. Make sure the API is running on http://localhost:8000.'
      )
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-full">
      <div className="filo-glow">
        <main className="mx-auto flex max-w-3xl flex-col items-center px-6 pt-20 pb-16">
          <p className="text-xs font-semibold uppercase tracking-[0.3em] text-amber">
            New York City
          </p>
          <h1 className="mt-3 text-center text-5xl font-bold tracking-tight text-white sm:text-6xl">
            filo
          </h1>
          <p className="mt-3 text-center text-lg text-mist">
            Describe the vibe. We'll find the spot.
          </p>

          <div className="mt-10 w-full">
            <SearchBar onSearch={runSearch} loading={loading} />
          </div>

          {!searched && !loading && (
            <div className="mt-6 flex flex-wrap justify-center gap-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => runSearch(s)}
                  className="rounded-full border border-edge bg-ink-soft px-3 py-1.5 text-sm text-mist transition hover:border-amber/40 hover:text-amber-soft"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
        </main>
      </div>

      <section className="mx-auto max-w-3xl px-6 pb-24">
        {error && (
          <div className="rounded-xl border border-ember/40 bg-ember/10 px-4 py-3 text-sm text-ember">
            {error}
          </div>
        )}

        {loading && (
          <div className="space-y-4">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-32 animate-pulse rounded-2xl border border-edge bg-card"
              />
            ))}
          </div>
        )}

        {!loading && searched && (
          <>
            <p className="mb-4 text-sm text-mist">
              {results.length > 0
                ? `${results.length} spots for `
                : 'No spots found for '}
              <span className="text-white">"{lastQuery}"</span>
            </p>
            <div className="grid grid-cols-1 gap-4">
              {results.map((venue, i) => (
                <VenueCard key={venue.id} venue={venue} index={i} />
              ))}
            </div>
          </>
        )}
      </section>
    </div>
  )
}
