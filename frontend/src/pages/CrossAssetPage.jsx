import useHistory from "../hooks/useHistory";
import TerminalLoader from "../components/Terminal/TerminalLoader";
import CrossAssetCorrelation from "../components/CrossAsset/CrossAssetCorrelation";
import "./CrossAssetPage.css";

export default function CrossAssetPage() {
  const { history, loading, error } = useHistory();

  if (loading) return <TerminalLoader message="LOADING CROSS-ASSET HISTORY" />;

  if (error) {
    return (
      <div className="cross-asset-state" role="alert">
        <strong>HISTORY FEED UNAVAILABLE</strong>
        <span>Refresh the page to retry the comparison.</span>
      </div>
    );
  }

  return <CrossAssetCorrelation history={history} />;
}
