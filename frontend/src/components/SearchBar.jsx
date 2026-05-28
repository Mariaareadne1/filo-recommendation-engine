import { useState } from 'react'

/**
 * Search input for filo.
 *
 * Renders a single centered text field with a submit button. Submits on Enter
 * or button click by calling `onSearch` with the trimmed query.
 *
 * @param {Object} props
 * @param {(query: string) => void} props.onSearch - called with the query on submit.
 * @param {boolean} props.loading - whether a request is in flight (disables submit).
 */
export default function SearchBar({ onSearch, loading }) {
  const [value, setValue] = useState('')

  const submit = (e) => {
    e.preventDefault()
    const q = value.trim()
    if (q && !loading) onSearch(q)
  }

  return (
    <form onSubmit={submit} className="w-full">
      <div className="group flex items-center gap-2 rounded-2xl border border-edge bg-ink-soft/80 px-4 py-2 shadow-lg shadow-black/40 backdrop-blur transition-colors focus-within:border-amber/60">
        <svg
          className="h-5 w-5 shrink-0 text-mist"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 11a6 6 0 11-12 0 6 6 0 0112 0z" />
        </svg>
        <input
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Find your NYC spot..."
          autoFocus
          className="flex-1 bg-transparent py-2 text-lg text-white placeholder:text-mist focus:outline-none"
        />
        <button
          type="submit"
          disabled={loading || !value.trim()}
          className="rounded-xl bg-amber px-5 py-2 font-semibold text-ink transition hover:bg-amber-soft disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? 'Searching…' : 'Search'}
        </button>
      </div>
    </form>
  )
}
