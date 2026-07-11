"""Stress test POST /api/scenario/run API contract."""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/api/scenario/run"

REQUIRED = [
    "summary",
    "regime",
    "confidence",
    "asset_deltas",
    "sector_impacts",
    "explanation",
]
ASSET_KEYS = ["sp500", "nasdaq", "ten_year_yield_bps", "dxy", "gold", "oil"]
SECTOR_KEYS = [
    "technology",
    "energy",
    "financials",
    "utilities",
    "healthcare",
    "consumer_discretionary",
]

VALID = {
    "fed_funds_change_bps": -50,
    "cpi_surprise_pct": -0.2,
    "oil_change_pct": 5,
    "gdp_surprise_pct": -0.5,
    "unemployment_change_pct": 0.2,
    "pmi_change": -3,
    "dxy_change_pct": 1,
    "vix_change_pct": 10,
}

BOUNDS = {
    "fed_funds_change_bps": (-100, 100),
    "cpi_surprise_pct": (-2, 2),
    "oil_change_pct": (-30, 30),
    "gdp_surprise_pct": (-5, 5),
    "unemployment_change_pct": (-2, 2),
    "pmi_change": (-10, 10),
    "dxy_change_pct": (-10, 10),
    "vix_change_pct": (-50, 100),
}

passed = 0
failed = 0
details = []


def post(payload, timeout=10):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else None, None
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            err = json.loads(raw)
        except Exception:
            err = raw
        return e.code, err, None
    except Exception as e:
        return None, None, str(e)


def check(name, ok, info=""):
    global passed, failed
    if ok:
        passed += 1
        details.append(f"PASS  {name}" + (f" ({info})" if info else ""))
    else:
        failed += 1
        details.append(f"FAIL  {name} — {info}")


def validate_response(body):
    if not isinstance(body, dict):
        return False, "not an object"
    missing = [k for k in REQUIRED if k not in body]
    if missing:
        return False, f"missing {missing}"
    if body.get("confidence") not in ("Low", "Medium", "High"):
        return False, f"bad confidence {body.get('confidence')}"
    ad = body.get("asset_deltas") or {}
    si = body.get("sector_impacts") or {}
    miss_a = [k for k in ASSET_KEYS if k not in ad]
    miss_s = [k for k in SECTOR_KEYS if k not in si]
    if miss_a or miss_s:
        return False, f"missing assets={miss_a} sectors={miss_s}"
    for k, v in {**ad, **si}.items():
        if not isinstance(v, (int, float)):
            return False, f"{k} not numeric"
    if not isinstance(body["summary"], str) or not body["summary"]:
        return False, "empty summary"
    if not isinstance(body["explanation"], str) or not body["explanation"]:
        return False, "empty explanation"
    if not isinstance(body["regime"], str) or not body["regime"]:
        return False, "empty regime"
    return True, "ok"


def main():
    print("=== Scenario Playground API contract stress test ===\n")

    status, body, err = post(VALID)
    ok, info = (False, err) if err else validate_response(body)
    check("valid request returns 200 + full shape", status == 200 and ok, f"status={status} {info}")

    for field, (lo, hi) in BOUNDS.items():
        for label, val in (("min", lo), ("max", hi)):
            payload = dict(VALID)
            payload[field] = val
            status, body, err = post(payload)
            ok, info = (False, err) if err else validate_response(body)
            check(f"boundary {field}={val} ({label})", status == 200 and ok, f"status={status} {info}")

    for field, (lo, hi) in BOUNDS.items():
        for side, val in (("below", lo - 1), ("above", hi + 1)):
            payload = dict(VALID)
            payload[field] = val
            status, body, err = post(payload)
            check(f"reject {field}={val} ({side})", status == 422, f"status={status} body={body}")

    for field in VALID:
        payload = dict(VALID)
        del payload[field]
        status, body, err = post(payload)
        check(f"reject missing {field}", status == 422, f"status={status}")

    for field in VALID:
        payload = dict(VALID)
        payload[field] = "not-a-number"
        status, body, err = post(payload)
        check(f"reject non-numeric {field}", status == 422, f"status={status}")

    status, body, err = post({})
    check("reject empty body", status == 422, f"status={status}")

    payload = dict(VALID)
    payload["extra_field"] = 123
    status, body, err = post(payload)
    ok, info = (False, err) if err else validate_response(body)
    check("extra fields still succeed (ignored)", status == 200 and ok, f"status={status} {info}")

    payload = dict(VALID)
    payload["vix_change_pct"] = None
    status, body, err = post(payload)
    check("reject null vix_change_pct", status == 422, f"status={status}")

    n = 40
    start = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        futs = [ex.submit(post, VALID) for _ in range(n)]
        for f in as_completed(futs):
            results.append(f.result())
    elapsed = time.perf_counter() - start
    ok_count = 0
    for status, body, err in results:
        if status == 200 and validate_response(body)[0]:
            ok_count += 1
    check(
        f"concurrent burst {n} requests",
        ok_count == n,
        f"ok={ok_count}/{n} in {elapsed:.2f}s ({n / max(elapsed, 1e-9):.1f} req/s)",
    )

    start = time.perf_counter()
    seq_ok = 0
    for i in range(25):
        payload = dict(VALID)
        payload["fed_funds_change_bps"] = -100 + (i * 8) % 201
        status, body, err = post(payload)
        if status == 200 and validate_response(body)[0]:
            seq_ok += 1
    elapsed = time.perf_counter() - start
    check(
        "rapid sequential 25 requests",
        seq_ok == 25,
        f"ok={seq_ok}/25 in {elapsed:.2f}s",
    )

    print("\n".join(details))
    print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
