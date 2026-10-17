"""Sequential probability ratio test (SPRT) for confirmed regressions."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SequentialConfig:
    """Test H0: regression rate=p0 against H1: regression rate=p1."""

    p0: float = 0.02
    p1: float = 0.20
    alpha: float = 0.05
    beta: float = 0.10
    min_samples: int = 8
    max_samples: int = 32

    def __post_init__(self) -> None:
        for name in ("p0", "p1", "alpha", "beta"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or (isinstance(value, float) and not math.isfinite(value))):
                raise ValueError(f"{name} must be a finite number")
        if not 0 < self.p0 < self.p1 < 1:
            raise ValueError("require 0 < p0 < p1 < 1")
        if not 0 < self.alpha < 1 or not 0 < self.beta < 1:
            raise ValueError("alpha and beta must be in (0, 1)")
        if self.alpha + self.beta >= 1:
            raise ValueError("require alpha + beta < 1")
        if type(self.min_samples) is not int or type(self.max_samples) is not int:
            raise ValueError("sample limits must be integers")
        if not 1 <= self.min_samples <= self.max_samples:
            raise ValueError("require 1 <= min_samples <= max_samples")


@dataclass
class SequentialTest:
    config: SequentialConfig = field(default_factory=SequentialConfig)
    samples: int = 0
    regressions: int = 0
    llr: float = 0.0

    @property
    def upper_bound(self) -> float:
        return math.log1p(-self.config.beta) - math.log(self.config.alpha)

    @property
    def lower_bound(self) -> float:
        return math.log(self.config.beta) - math.log1p(-self.config.alpha)

    def add(self, regressed: bool) -> str:
        if type(regressed) is not bool:
            raise ValueError("regressed must be a boolean")
        if self.decision() != "running":
            raise ValueError("cannot add observations after the sequential test has stopped")
        self.samples += 1
        if regressed:
            self.regressions += 1
            self.llr += math.log(self.config.p1) - math.log(self.config.p0)
        else:
            self.llr += math.log1p(-self.config.p1) - math.log1p(-self.config.p0)
        return self.decision()

    def decision(self) -> str:
        if self.samples < self.config.min_samples:
            return "running"
        if self.llr >= self.upper_bound:
            return "regressed"
        if self.llr <= self.lower_bound:
            return "stable"
        if self.samples >= self.config.max_samples:
            return "inconclusive"
        return "running"

    def snapshot(self) -> dict[str, float | int | str]:
        return {
            "decision": self.decision(),
            "samples": self.samples,
            "confirmed_regressions": self.regressions,
            "llr": round(self.llr, 6),
            "lower_bound": round(self.lower_bound, 6),
            "upper_bound": round(self.upper_bound, 6),
            "p0": self.config.p0,
            "p1": self.config.p1,
            "alpha": self.config.alpha,
            "beta": self.config.beta,
            "min_samples": self.config.min_samples,
            "max_samples": self.config.max_samples,
        }


def _binomial_tail(n: int, p: float, start: int, stop: int) -> float:
    """Sum an inclusive binomial tail in log space, without optional dependencies."""
    log_factorial = math.lgamma(n + 1)
    terms = (
        math.exp(
            log_factorial - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p)
        )
        for k in range(start, stop + 1)
    )
    return min(1.0, max(0.0, math.fsum(terms)))


def fixed_sample_verdict(
    samples: int, regressions: int, config: SequentialConfig | None = None
) -> dict[str, float | int | str]:
    """Evaluate all available pairs, using two exact one-sided binomial tests.

    The p0 upper tail tests excessive failures; the p1 lower tail tests evidence
    against material regression. Both tests rejecting places the result in the
    indifference region, so it remains inconclusive. max_samples applies only to
    sequential runs and never discards observations in this fixed-size test.
    """
    config = config or SequentialConfig()
    if type(samples) is not int or type(regressions) is not int or not 0 <= regressions <= samples:
        raise ValueError("require integer counts with 0 <= regressions <= samples")
    regression_p = _binomial_tail(samples, config.p0, regressions, samples)
    stable_p = _binomial_tail(samples, config.p1, 0, regressions)
    excessive = regression_p <= config.alpha
    below_material = stable_p <= config.beta
    decision = "inconclusive"
    if samples >= config.min_samples:
        if excessive and not below_material:
            decision = "regressed"
        elif below_material and not excessive:
            decision = "stable"
    llr = regressions * (math.log(config.p1) - math.log(config.p0))
    llr += (samples - regressions) * (math.log1p(-config.p1) - math.log1p(-config.p0))
    return {
        "method": "fixed_binomial",
        "decision": decision,
        "samples": samples,
        "confirmed_regressions": regressions,
        "llr": round(llr, 6),
        "regression_p_value": regression_p,
        "stable_p_value": stable_p,
        "p0": config.p0,
        "p1": config.p1,
        "alpha": config.alpha,
        "beta": config.beta,
        "min_samples": config.min_samples,
    }
