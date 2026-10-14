/** Describe observation dates without treating request time as market freshness. */
export function observationRange(dates = {}) {
  const values = Object.values(dates).filter((date) => /^\d{4}-\d{2}-\d{2}$/.test(date ?? '')).sort();
  if (!values.length) return 'Unknown';
  const first = values[0];
  const last = values[values.length - 1];
  return first === last ? first : `${first} – ${last}`;
}

/** Report connection state separately from observation age and source cadence. */
export function connectionStatus({ error, overview, loading }) {
  if (error) return overview ? 'REFRESH FAILED' : 'UNAVAILABLE';
  if (loading) return 'CHECKING';
  return overview ? 'CONNECTED' : 'UNAVAILABLE';
}
