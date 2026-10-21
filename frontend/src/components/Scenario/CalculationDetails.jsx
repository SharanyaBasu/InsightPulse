const OUTPUT_UNIT_LABELS = {
  percent: "%",
  basis_points: "bps",
};

function formatNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const normalized = Object.is(value, -0) ? 0 : value;
  return normalized.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
}

function formatOutput(value, unit) {
  return `${formatNumber(value)} ${OUTPUT_UNIT_LABELS[unit] || unit}`;
}

function formatInput(value, unit) {
  const unitLabel = unit === "percent" ? "%" : unit;
  return `${formatNumber(value)} ${unitLabel}`;
}

/** Render the server-authored model identity, submitted inputs, and prediction arithmetic. */
export default function CalculationDetails({ details, runAt }) {
  if (!details) return null;

  const inputs = Object.entries(details.submitted_inputs || {});
  const targets = Object.entries(details.targets || {});

  return (
    <section className="calculation-details" aria-labelledby="calculation-details-heading">
      <details>
        <summary id="calculation-details-heading">Calculation details</summary>
        <div className="calculation-details__body">
          <dl className="calculation-details__metadata">
            <div><dt>Model version</dt><dd>{details.model_version}</dd></div>
            <div><dt>Bundle version</dt><dd>{details.bundle_version}</dd></div>
            <div><dt>Baseline date</dt><dd>{details.baseline_date}</dd></div>
            <div><dt>Forecast horizon</dt><dd>{details.horizon_trading_days} trading days</dd></div>
            {runAt && <div><dt>Displayed run</dt><dd>{runAt.toLocaleString()}</dd></div>}
          </dl>

          <h3>Submitted inputs</h3>
          <dl className="calculation-details__inputs">
            {inputs.map(([key, input]) => (
              <div key={key}>
                <dt>{input.label}</dt>
                <dd>{formatInput(input.value, input.unit)}</dd>
              </div>
            ))}
          </dl>

          <h3>Prediction calculation</h3>
          <p className="calculation-details__formula">
            Difference = scenario prediction − baseline prediction. Values below are model outputs before the main cards round them for display.
          </p>
          <div className="calculation-details__table-wrap">
            <table className="calculation-details__table">
              <thead>
                <tr>
                  <th scope="col">Target</th>
                  <th scope="col">Baseline</th>
                  <th scope="col">Scenario</th>
                  <th scope="col">Difference</th>
                </tr>
              </thead>
              <tbody>
                {targets.map(([key, target]) => (
                  <tr key={key}>
                    <th scope="row">{target.label}</th>
                    <td>{formatOutput(target.baseline_prediction, target.unit)}</td>
                    <td>{formatOutput(target.scenario_prediction, target.unit)}</td>
                    <td>{formatOutput(target.difference, target.unit)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </details>
    </section>
  );
}
