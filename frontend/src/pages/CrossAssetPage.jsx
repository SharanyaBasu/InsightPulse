import useHistory from "../hooks/useHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import DataState from "../components/Terminal/DataState";
import CrossAssetCorrelation from "../components/CrossAsset/CrossAssetCorrelation";
import "./CrossAssetPage.css";
export default function CrossAssetPage() {
  const { history, loading, error, refresh } = useHistory();
  if (loading && !history) return <TerminalLoader message="LOADING CROSS-ASSET HISTORY" />;
  return <div className="page-stack">{error && <DataState error title="History feed unavailable" message={history ? "Showing previously loaded history." : "Try again to load the comparison."} onRetry={refresh} loading={loading} />}{history && <CrossAssetCorrelation history={history} />}</div>;
}
