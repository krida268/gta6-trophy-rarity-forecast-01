"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model (per trophy, all on the logit scale):
  - x_j = ln(1 + hours from 2026-11-19T00:00:00Z to read j)
  - y_j = logit(rate_j), rates clamped to [0.01, 99.99], null reads skipped
  - Fit on reads in the 24 hours before the case checkpoint; if fewer than
    3 reads fall in that window, use all reads; with fewer than 3 reads in
    total, fall back to persistence (median = last rate).
  - Robust Theil-Sen trend: slope s = median pairwise slope, intercept
    a = median(y_j - s*x_j), fitted on at most the 200 newest reads.
    Slope clamped to [-0.75, 1.5], damped s_eff = s * 0.85.
  - Median logit at target m = a + s_eff * ln(1 + hours to target); the
    median rate is inv_logit(m), clamped to [0.1x, 10x] the last rate.
  - Spread sigma = sqrt((0.5*sqrt(horizon_days))^2 + resid^2 + slope_unc^2),
    resid = 1.4826 * median |y_j - (a + s*x_j)| clamped to [0.15, 1.0],
    slope_unc = resid * max(0, x_target - x_last) capped at 0.5,
    sigma clamped to [0.35, 2.2].
  - Quantiles q_tau = inv_logit(m + z_tau * sigma), z = [-1.645, -0.674,
    0, 0.674, 1.645], rounded to 4 decimals half-up, clamped to [0, 100],
    enforced non-decreasing.

Standard library only, deterministic, no randomness, no network.
Reads every <id>.in in $MAKEFUN_INPUT_DIR (default /input) and writes
$MAKEFUN_OUTPUT_DIR/<id>.out (default /output) as compact JSON + newline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch-day origin for the log-time axis.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Model constants.
RATE_LO = 0.01
RATE_HI = 99.99
SLOPE_LO = -0.75
SLOPE_HI = 1.5
SLOPE_DAMP = 0.85
RESID_LO = 0.15
RESID_HI = 1.0
SLOPE_UNC_MAX = 0.5
SIGMA_LO = 0.35
SIGMA_HI = 2.2
MEDIAN_FLOOR_MULT = 0.1
MEDIAN_CEIL_MULT = 10.0
FIT_WINDOW_HOURS = 24.0
FIT_MAX_READS = 200
DEFAULT_RATE = 1.0


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def hours_since_epoch(moment):
    """Hours from the launch-day origin to a datetime (never negative)."""
    if moment is None:
        return 0.0
    return max(0.0, (moment - EPOCH).total_seconds() / 3600.0)


def log_time(moment):
    """x = ln(1 + hours since the origin) for a datetime."""
    return math.log1p(hours_since_epoch(moment))


def clamp_rate(rate):
    return min(RATE_HI, max(RATE_LO, rate))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(value):
    """100 / (1 + exp(-value)), overflow-safe."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value)) if value < 700 else 100.0
    exp_v = math.exp(value) if value > -700 else 0.0
    return 100.0 * exp_v / (1.0 + exp_v)


def median(values):
    """Median of a non-empty list of numbers (sorted copy, deterministic)."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def round4(x):
    """x rounded to 4 decimals, half up; whole numbers come back as int."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5)
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def theil_sen(points):
    """Robust line fit: slope = median pairwise slope, intercept = median residual.

    points: list of (x, y) sorted by x. Returns (slope, intercept).
    """
    n = len(points)
    slopes = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xj - xi
            if dx > 0:
                slopes.append((yj - yi) / dx)
    if not slopes:
        return 0.0, points[0][1] if points else 0.0
    slope = median(slopes)
    intercept = median([y - slope * x for x, y in points])
    return slope, intercept


def tier_median_rate(trophies, rates, tier):
    """Median non-null rate of same-tier trophies in one read, or None."""
    values = [
        rates[i]
        for i, trophy in enumerate(trophies)
        if trophy.get("tier") == tier and rates[i] is not None
    ]
    return median(values) if values else None


def last_rate_for(trophies, history, index):
    """Latest non-null rate for one trophy; tier median at last read; else 1.0."""
    for read in reversed(history):
        rates = read.get("rates") or []
        if index < len(rates) and rates[index] is not None:
            return float(rates[index])
    if history:
        last = history[-1]
        rates = last.get("rates") or []
        tier = trophies[index].get("tier")
        tier_rate = tier_median_rate(trophies, rates, tier)
        if tier_rate is not None:
            return float(tier_rate)
    return DEFAULT_RATE


def forecast_trophy(points, x_target, horizon_minutes, fallback_rate):
    """Forecast five quantiles for one trophy.

    points: fitted (x, y) reads, oldest first (may be empty).
    fallback_rate: rate to use when no trend can be fitted.
    Returns [q05, q25, q50, q75, q95] as numbers, non-decreasing.
    """
    horizon_days = max(0.0, horizon_minutes) / 1440.0
    rw_sigma = 0.5 * math.sqrt(horizon_days)

    if len(points) < 3:
        # Persistence: median is the last known rate, spread is the
        # random-walk term only.
        med = clamp_rate(fallback_rate)
        sigma = min(SIGMA_HI, max(SIGMA_LO, rw_sigma))
        m = logit(med)
    else:
        slope, intercept = theil_sen(points)
        slope = min(SLOPE_HI, max(SLOPE_LO, slope))
        slope_eff = slope * SLOPE_DAMP
        m = intercept + slope_eff * x_target
        med = inv_logit(m)
        # Keep the median within [0.1x, 10x] the last known rate.
        med = min(MEDIAN_CEIL_MULT * fallback_rate,
                  max(MEDIAN_FLOOR_MULT * fallback_rate, med))
        m = logit(med)
        resid = 1.4826 * median(
            [abs(y - (intercept + slope * x)) for x, y in points]
        )
        resid = min(RESID_HI, max(RESID_LO, resid))
        x_last = points[-1][0]
        slope_unc = min(SLOPE_UNC_MAX, resid * max(0.0, x_target - x_last))
        sigma = math.sqrt(rw_sigma * rw_sigma + resid * resid
                          + slope_unc * slope_unc)
        sigma = min(SIGMA_HI, max(SIGMA_LO, sigma))

    quantiles = []
    prev = 0.0
    for z in Z:
        q = round4(inv_logit(m + z * sigma))
        q = min(100.0, max(0.0, q))
        q = max(q, prev)  # enforce non-decreasing after rounding
        quantiles.append(q)
        prev = q
    return quantiles


def forecast_case(case):
    """Forecast one case dict into {"trophies": [{"id", "q"}, ...]}."""
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    checkpoint = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    # The case's history holds reads before the checkpoint; drop any read
    # stamped after it, so a late or mis-cut read cannot leak the future.
    if checkpoint is not None:
        kept = []
        for read in history:
            moment = parse_time(read.get("at"))
            if moment is None or moment <= checkpoint:
                kept.append(read)
        history = kept
    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None and checkpoint is not None and target is not None:
        horizon_minutes = (target - checkpoint).total_seconds() / 60.0
    if horizon_minutes is None:
        horizon_minutes = 0.0
    horizon_minutes = float(horizon_minutes)

    x_target = log_time(target) if target is not None else 0.0

    # Reads in the 24 hours before the checkpoint; if fewer than 3, all reads.
    window_start = None
    if checkpoint is not None:
        window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    window_reads = []
    if window_start is not None:
        for read in history:
            moment = parse_time(read.get("at"))
            if moment is not None and moment.timestamp() >= window_start:
                window_reads.append(read)
    fit_reads = window_reads if len(window_reads) >= 3 else list(history)
    # Cap the fit at the 200 newest reads (Theil-Sen is O(n^2)).
    fit_reads = fit_reads[-FIT_MAX_READS:]

    rows = []
    for index, trophy in enumerate(trophies):
        fallback = last_rate_for(trophies, history, index)
        points = []
        for read in fit_reads:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            moment = parse_time(read.get("at"))
            if moment is None:
                continue
            points.append((log_time(moment), logit(float(rate))))
        points.sort(key=lambda p: p[0])
        rows.append({
            "id": trophy["id"],
            "q": forecast_trophy(points, x_target, horizon_minutes, fallback),
        })
    return {"trophies": rows}


def safe_forecast_case(case):
    """Forecast a case, falling back to a flat 1% forecast on any error."""
    try:
        return forecast_case(case)
    except Exception:
        rows = []
        for trophy in case.get("trophies") or []:
            rows.append({"id": trophy["id"], "q": [1, 1, 1, 1, 1]})
        return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = safe_forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
