"""Calculated metric generation using currently ingested source data."""

import json
import math
import statistics
from datetime import datetime, time, timedelta, timezone

from models import CalculatedMetric, CryptoQuote, MacroData, MarketData


CALCULATION_VERSION = "v1"

MARKET_ASSETS = {
    "sp500": "sp500",
    "nasdaq": "nasdaq",
    "vix": "vix",
    "gold": "gold",
    "usd_index": "usd_index",
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "tlt": "tlt",
}


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _as_datetime(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)

    return datetime.combine(value, time.min)


def _is_valid_number(value):
    return value is not None and math.isfinite(float(value))


def _metric(metric_name, category, timestamp, value, window, source_dependencies):
    if not _is_valid_number(value):
        print(f"Skipped metric {metric_name}: invalid value")
        return None

    return {
        "metric_name": metric_name,
        "category": category,
        "timestamp": _as_datetime(timestamp),
        "value": float(value),
        "window": window,
        "source_dependencies": json.dumps(source_dependencies),
        "calculation_version": CALCULATION_VERSION,
    }


def _add_metric(metrics, metric):
    if metric is not None:
        metrics.append(metric)


def _latest_row_with_values(rows, fields):
    for row in reversed(rows):
        if all(_is_valid_number(getattr(row, field)) for field in fields):
            return row

    return None


def _market_series(rows, field):
    values = []

    for row in rows:
        value = getattr(row, field)
        if _is_valid_number(value):
            values.append((row.date, float(value)))

    return values


def _safe_ratio_return(current, previous):
    if not _is_valid_number(current) or not _is_valid_number(previous) or float(previous) == 0:
        return None

    return (float(current) / float(previous)) - 1


def _latest_crypto_price(crypto_rows, symbol):
    matching_rows = [row for row in crypto_rows if row.symbol == symbol and _is_valid_number(row.price)]
    if not matching_rows:
        return None

    return max(matching_rows, key=lambda row: row.timestamp)


def _calculate_macro_metrics(metrics, macro_rows):
    latest_yields = _latest_row_with_values(macro_rows, ["ten_year_yield", "two_year_yield"])
    if latest_yields is None:
        print("Skipped metric yield_curve_slope_10y_2y: missing yield data")
    else:
        _add_metric(
            metrics,
            _metric(
                "yield_curve_slope_10y_2y",
                "macro",
                latest_yields.date,
                latest_yields.ten_year_yield - latest_yields.two_year_yield,
                "latest",
                ["macro_data"],
            ),
        )

    latest_cpi = _latest_row_with_values(macro_rows, ["fed_funds_rate", "cpi"])
    if latest_cpi is None:
        print("Skipped metric real_rate_proxy: missing Fed Funds Rate or CPI")
        return

    target_date = latest_cpi.date - timedelta(days=365)
    prior_cpi_rows = [
        row
        for row in macro_rows
        if row.date <= target_date and _is_valid_number(row.cpi) and float(row.cpi) != 0
    ]

    if not prior_cpi_rows:
        print("Skipped metric real_rate_proxy: not enough CPI history")
        return

    prior_cpi = max(prior_cpi_rows, key=lambda row: row.date)
    cpi_yoy_percent = ((latest_cpi.cpi / prior_cpi.cpi) - 1) * 100

    _add_metric(
        metrics,
        _metric(
            "real_rate_proxy",
            "macro",
            latest_cpi.date,
            latest_cpi.fed_funds_rate - cpi_yoy_percent,
            "latest",
            ["macro_data"],
        ),
    )


def _calculate_dxy_return(metrics, market_rows):
    dxy_rows = _market_series(market_rows, "usd_index")
    if len(dxy_rows) < 2:
        print("Skipped metric dxy_1d_return: fewer than two USD_Index observations")
        return

    latest_date, latest_value = dxy_rows[-1]
    _, previous_value = dxy_rows[-2]

    _add_metric(
        metrics,
        _metric(
            "dxy_1d_return",
            "fx",
            latest_date,
            _safe_ratio_return(latest_value, previous_value),
            "1d",
            ["market_data"],
        ),
    )


def _calculate_btc_eth_ratio(metrics, market_rows, crypto_rows):
    btc_quote = _latest_crypto_price(crypto_rows, "BTC")
    eth_quote = _latest_crypto_price(crypto_rows, "ETH")

    if btc_quote is not None and eth_quote is not None and float(eth_quote.price) != 0:
        _add_metric(
            metrics,
            _metric(
                "btc_eth_ratio",
                "crypto",
                max(btc_quote.timestamp, eth_quote.timestamp),
                float(btc_quote.price) / float(eth_quote.price),
                "latest",
                ["crypto_quotes"],
            ),
        )
        return

    market_crypto = _latest_row_with_values(market_rows, ["bitcoin", "ethereum"])
    if market_crypto is None or float(market_crypto.ethereum) == 0:
        print("Skipped metric btc_eth_ratio: missing BTC or ETH price")
        return

    _add_metric(
        metrics,
        _metric(
            "btc_eth_ratio",
            "crypto",
            market_crypto.date,
            market_crypto.bitcoin / market_crypto.ethereum,
            "latest",
            ["market_data"],
        ),
    )


def _calculate_market_asset_metrics(metrics, market_rows):
    for metric_name, field in MARKET_ASSETS.items():
        series = _market_series(market_rows, field)
        if not series:
            print(f"Skipped market metrics for {metric_name}: no valid observations")
            continue

        latest_date, latest_value = series[-1]
        values = [value for _, value in series]

        if len(values) >= 6:
            _add_metric(
                metrics,
                _metric(
                    f"{metric_name}_5d_return",
                    "market",
                    latest_date,
                    _safe_ratio_return(values[-1], values[-6]),
                    "5d",
                    ["market_data"],
                ),
            )
        else:
            print(f"Skipped metric {metric_name}_5d_return: fewer than 6 observations")

        if len(values) >= 21 and all(value > 0 for value in values[-21:]):
            log_returns = [
                math.log(values[index] / values[index - 1])
                for index in range(len(values) - 20, len(values))
            ]
            _add_metric(
                metrics,
                _metric(
                    f"{metric_name}_20d_volatility",
                    "market",
                    latest_date,
                    statistics.stdev(log_returns) * math.sqrt(252),
                    "20d",
                    ["market_data"],
                ),
            )
        else:
            print(f"Skipped metric {metric_name}_20d_volatility: insufficient positive history")

        peak = max(values)
        if peak != 0:
            _add_metric(
                metrics,
                _metric(
                    f"{metric_name}_drawdown_from_peak",
                    "market",
                    latest_date,
                    (latest_value - peak) / peak,
                    "1y",
                    ["market_data"],
                ),
            )
        else:
            print(f"Skipped metric {metric_name}_drawdown_from_peak: peak is zero")

        if len(values) >= 252:
            zscore_values = values[-252:]
            zscore_window = "252d"
        elif len(values) >= 60:
            zscore_values = values
            zscore_window = "available_history"
        else:
            zscore_values = []
            zscore_window = None

        if zscore_values:
            std_dev = statistics.stdev(zscore_values)
            if std_dev != 0:
                _add_metric(
                    metrics,
                    _metric(
                        f"{metric_name}_price_zscore_252d",
                        "market",
                        latest_date,
                        (latest_value - statistics.mean(zscore_values)) / std_dev,
                        zscore_window,
                        ["market_data"],
                    ),
                )
            else:
                print(f"Skipped metric {metric_name}_price_zscore_252d: standard deviation is zero")
        else:
            print(f"Skipped metric {metric_name}_price_zscore_252d: fewer than 60 observations")

        for window in (50, 200):
            metric_suffix = f"ma_distance_{window}d"
            if len(values) < window:
                print(f"Skipped metric {metric_name}_{metric_suffix}: fewer than {window} observations")
                continue

            moving_average = statistics.mean(values[-window:])
            if moving_average == 0:
                print(f"Skipped metric {metric_name}_{metric_suffix}: moving average is zero")
                continue

            _add_metric(
                metrics,
                _metric(
                    f"{metric_name}_{metric_suffix}",
                    "market",
                    latest_date,
                    (latest_value - moving_average) / moving_average,
                    f"{window}d",
                    ["market_data"],
                ),
            )


def calculate_metrics(db) -> tuple[list[dict], int]:
    market_rows = db.query(MarketData).order_by(MarketData.date).all()
    macro_rows = db.query(MacroData).order_by(MacroData.date).all()
    crypto_rows = (
        db.query(CryptoQuote)
        .filter(CryptoQuote.symbol.in_(["BTC", "ETH"]))
        .order_by(CryptoQuote.timestamp)
        .all()
    )

    rows_inspected = len(market_rows) + len(macro_rows) + len(crypto_rows)
    metrics = []

    _calculate_macro_metrics(metrics, macro_rows)
    _calculate_dxy_return(metrics, market_rows)
    _calculate_btc_eth_ratio(metrics, market_rows, crypto_rows)
    _calculate_market_asset_metrics(metrics, market_rows)

    return metrics, rows_inspected


def save_calculated_metrics(db, metrics: list[dict]) -> int:
    timestamp = _utc_now_naive()

    for metric in metrics:
        row = CalculatedMetric(
            metric_name=metric["metric_name"],
            category=metric["category"],
            timestamp=metric["timestamp"],
            value=metric["value"],
            window=metric["window"],
            source_dependencies=metric["source_dependencies"],
            calculation_version=metric["calculation_version"],
            created_at=timestamp,
            updated_at=timestamp,
        )
        db.add(row)

    db.commit()
    rows_written = len(metrics)
    print(f"Saved {rows_written} calculated metric rows to database.")
    return rows_written


def refresh_calculated_metrics(db) -> tuple[int, int]:
    metrics, rows_inspected = calculate_metrics(db)
    rows_written = save_calculated_metrics(db, metrics)

    return rows_written, rows_inspected
