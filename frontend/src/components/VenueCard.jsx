/** Human-readable labels for the internal category slugs. */
const CATEGORY_LABELS = {
  restaurant: 'Restaurant',
  bar: 'Bar',
  cafe: 'Cafe',
  wine_bar: 'Wine Bar',
  cocktail_bar: 'Cocktail Bar',
  rooftop_bar: 'Rooftop Bar',
  jazz_club: 'Jazz Club',
  speakeasy: 'Speakeasy',
  ramen_shop: 'Ramen',
  pizza_spot: 'Pizza',
  brunch_spot: 'Brunch',
  date_spot: 'Date Spot',
}

/**
 * Format a category slug into a display label.
 * @param {string} category
 * @returns {string}
 */
function categoryLabel(category) {
  return CATEGORY_LABELS[category] || category.replace(/_/g, ' ')
}

/**
 * A single venue result card.
 *
 * Shows the venue name, neighborhood + category pills, the description, and a
 * subtle match-percentage badge derived from the cosine similarity score.
 *
 * @param {Object} props
 * @param {Object} props.venue - venue object from the API.
 * @param {string} props.venue.name
 * @param {string} props.venue.neighborhood
 * @param {string} props.venue.category
 * @param {string} props.venue.description
 * @param {number} props.venue.score - cosine similarity in [0, 1].
 * @param {number} props.index - position in the result list (for stagger animation).
 */
export default function VenueCard({ venue, index = 0 }) {
  const matchPct = Math.round(Math.max(0, Math.min(1, venue.score)) * 100)

  return (
    <article
      className="filo-rise group rounded-2xl border border-edge bg-card p-5 transition-colors hover:border-amber/40 hover:bg-card-hover"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="flex items-start justify-between gap-4">
        <h3 className="text-2xl font-semibold tracking-tight text-white">{venue.name}</h3>
        <span
          className="shrink-0 rounded-full border border-amber/30 bg-amber/10 px-2.5 py-1 text-xs font-medium text-amber-soft"
          title={`Cosine similarity ${venue.score.toFixed(3)}`}
        >
          {matchPct}% match
        </span>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        <span className="rounded-full bg-ember/15 px-3 py-1 text-xs font-medium text-ember">
          {venue.neighborhood}
        </span>
        <span className="rounded-full bg-white/5 px-3 py-1 text-xs font-medium text-mist">
          {categoryLabel(venue.category)}
        </span>
      </div>

      <p className="mt-4 text-sm leading-relaxed text-mist">{venue.description}</p>
    </article>
  )
}
