"""Read only helpers for loading calculated metrics from database."""

from sqlalchemy import func
from sqlalchemy.orm import Session

from models import CalculatedMetric


def _row_to_dict(row: CalculatedMetric) -> dict:
    return {
        "metric_name": row.metric_name,
        "category": row.category,
        "value": row.value,
        "timestamp": row.timestamp.isoformat() if row.timestamp else None,
        "window": row.window,
        "source_dependencies": row.source_dependencies,
        "calculation_version": row.calculation_version,
    }


def load_latest_calculated_metrics(db: Session) -> dict[str, dict]:
    """
    Return latest row per metric_name from calculated_metrics.

    When duplicate refreshes exist for the same metric, the row with the
    most recent timestamp wins. Ties on timestamp resolve to the highest id.
    """
    latest_ts = (
        db.query(
            CalculatedMetric.metric_name,
            func.max(CalculatedMetric.timestamp).label("max_timestamp"),
        )
        .group_by(CalculatedMetric.metric_name)
        .subquery()
    )

    rows = (
        db.query(CalculatedMetric)
        .join(
            latest_ts,
            (CalculatedMetric.metric_name == latest_ts.c.metric_name)
            & (CalculatedMetric.timestamp == latest_ts.c.max_timestamp),
        )
        .order_by(CalculatedMetric.metric_name, CalculatedMetric.id.desc())
        .all()
    )

    result: dict[str, dict] = {}
    for row in rows:
        if row.metric_name not in result:
            result[row.metric_name] = _row_to_dict(row)

    return result
