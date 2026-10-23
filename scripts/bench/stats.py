"""Summary statistics and the paired A/B decision rule."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass

# Two-sided 95 % Student-t critical values, df = 1..30; larger df use the normal 1.96.
_T95 = [
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
]


def t95(df: int) -> float:
    if df < 1:
        raise ValueError("need at least 2 samples")
    return _T95[df - 1] if df <= len(_T95) else 1.96


def percentile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy 'linear' method), q in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty sample")
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class Summary:
    n: int
    median: float
    p95: float
    mean: float
    std: float
    min: float
    max: float

    def to_dict(self) -> dict:
        return asdict(self)


def summarize(xs: list[float]) -> Summary:
    return Summary(
        n=len(xs),
        median=statistics.median(xs),
        p95=percentile(xs, 95),
        mean=statistics.fmean(xs),
        std=statistics.stdev(xs) if len(xs) > 1 else 0.0,
        min=min(xs),
        max=max(xs),
    )


@dataclass
class PairedResult:
    n: int
    mean_diff: float  # mean(b - a); negative = B faster
    ci95: tuple[float, float]
    median_improvement: float  # median(a) - median(b)
    pooled_std: float  # max of the two per-arm repetition std devs
    significant: bool  # CI excludes 0 and |median improvement| > pooled std

    @property
    def beneficial(self) -> bool:
        return self.significant and self.median_improvement > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci95"] = list(self.ci95)
        d["beneficial"] = self.beneficial
        return d


def paired_ab(a: list[float], b: list[float]) -> PairedResult:
    """Paired comparison of alternating A/B repetitions (a[i] and b[i] measured back to back)."""
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("need equal-length samples with n >= 2")
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    md = statistics.fmean(d)
    half = t95(n - 1) * statistics.stdev(d) / math.sqrt(n)
    ci = (md - half, md + half)
    improvement = statistics.median(a) - statistics.median(b)
    pooled = max(statistics.stdev(a), statistics.stdev(b))
    sig = (ci[0] > 0 or ci[1] < 0) and abs(improvement) > pooled
    return PairedResult(n, md, ci, improvement, pooled, sig)
