export default function DataState({ title = "No data available", message, onRetry, loading = false, error = false }) {
  return <div className="data-state" role={error ? "alert" : "status"}>
    <strong>{title}</strong>
    {message && <p>{message}</p>}
    {onRetry && <button className="ui-button" onClick={onRetry} disabled={loading}>{loading ? "Checking…" : "Retry"}</button>}
  </div>;
}
