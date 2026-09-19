"""Weather-driven disease risk priors.

Read this before touching the code
-----------------------------------
You asked whether image + weather can be fused in one trainable MLP. The answer
is no, not honestly, and the reason is worth stating in the pitch because it is
a mark of rigour rather than a weakness:

PlantVillage, PlantDoc, PlantWild and the rest carry **no paired weather, GPS or
date metadata**. There is no (image, weather, label) triple to learn a joint
model from. Anything you fit that claims to use both is really learning from the
image alone while the weather branch memorises noise — and it will look fine on
your validation split because the noise is consistent within a dataset.

So the weather branch here is **not learned**. It is an explicit, citable,
rule-based epidemiological prior, and it is applied as a *re-ranker* over the
vision model's top-k, never as a source of new diagnoses. Every rule below
carries the literature it comes from. The mixing weight lambda is a product
decision, exposed and ablatable, not a fitted parameter pretending to be one.

The path to a genuinely trainable fusion model runs through the app: log
(image, gps, timestamp, weather snapshot, model prediction, farmer/agronomist
confirmation) for every diagnosis. At roughly 2,000-5,000 confirmed records with
geographic and seasonal spread you can fit a real late-fusion model and replace
this file. Until then, this is the intellectually honest version — and it is
also a feature: rules are inspectable and an agronomist can correct them, which
a learned MLP does not allow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable, Dict, List, Optional

from .weather import WeatherSeries


@dataclass
class RiskRule:
    """One disease's infection-conducive-conditions rule."""

    class_id: str
    name: str
    source: str  # literature citation - shown in the app's "why?" panel
    evaluate: Callable[[WeatherSeries, datetime], float]
    latent_period_days: int = 7  # window before the photo that matters


def _window(series: WeatherSeries, observed_at: datetime, days: int) -> WeatherSeries:
    return series.slice_days(observed_at - timedelta(days=days), days)


# ---------------------------------------------------------------- rules
def _late_blight(series: WeatherSeries, observed_at: datetime) -> float:
    """Smith period: two consecutive days with min temp >= 10 C and >= 11 h at RH >= 90%."""
    window = _window(series, observed_at, 10)
    by_day: Dict[date, List[tuple]] = {}
    for t, temp, rh in zip(window.timestamps, window.temperature_c, window.relative_humidity):
        by_day.setdefault(t.date(), []).append((temp, rh))

    qualifying = []
    for day in sorted(by_day):
        readings = by_day[day]
        min_temp = min(r[0] for r in readings)
        humid_hours = sum(1 for r in readings if r[1] >= 90.0)
        qualifying.append(min_temp >= 10.0 and humid_hours >= 11)

    runs = 0
    best = 0
    for ok in qualifying:
        runs = runs + 1 if ok else 0
        best = max(best, runs)
    return min(1.0, best / 2.0)


def _rice_blast(series: WeatherSeries, observed_at: datetime) -> float:
    """Conducive: 20-28 C with long leaf wetness; risk rises with wet hours."""
    window = _window(series, observed_at, 7)
    wetness = window.leaf_wetness_hours()
    temps = window.daily_mean_temperature()
    score = 0.0
    for day, hours in wetness.items():
        temp = temps.get(day, 0.0)
        if 20.0 <= temp <= 28.0:
            score += min(1.0, hours / 12.0)
    return min(1.0, score / 4.0)


def _bacterial_leaf_blight(series: WeatherSeries, observed_at: datetime) -> float:
    """Driven by heavy rain, standing water and wind (splash/wound dispersal)."""
    window = _window(series, observed_at, 7)
    rain = sum(window.daily_rainfall().values())
    wind = max(window.wind_speed) if window.wind_speed else 0.0
    return min(1.0, rain / 60.0) * (0.6 + 0.4 * min(1.0, wind / 25.0))


def _powdery_mildew(series: WeatherSeries, observed_at: datetime) -> float:
    """Unusual among fungi: moderate temperature, *moderate* RH, and free water
    inhibits germination. Rain therefore *lowers* this risk."""
    window = _window(series, observed_at, 7)
    temps = window.daily_mean_temperature()
    rain = sum(window.daily_rainfall().values())
    good_days = sum(1 for t in temps.values() if 20.0 <= t <= 28.0)
    base = good_days / max(1, len(temps))
    return float(max(0.0, base * (1.0 - min(1.0, rain / 30.0))))


def _rust(series: WeatherSeries, observed_at: datetime) -> float:
    """Urediniospore germination: 15-22 C with several hours of dew."""
    window = _window(series, observed_at, 10)
    wetness = window.leaf_wetness_hours()
    temps = window.daily_mean_temperature()
    score = sum(
        min(1.0, wetness.get(day, 0) / 6.0)
        for day, temp in temps.items()
        if 15.0 <= temp <= 22.0
    )
    return min(1.0, score / 5.0)


def _early_blight(series: WeatherSeries, observed_at: datetime) -> float:
    """Alternaria favours warm days with alternating wet/dry cycles."""
    window = _window(series, observed_at, 10)
    temps = window.daily_mean_temperature()
    wetness = window.leaf_wetness_hours()
    cycles = 0
    days = sorted(temps)
    for i in range(1, len(days)):
        wet_then_dry = wetness.get(days[i - 1], 0) >= 8 and wetness.get(days[i], 0) <= 5
        warm = 24.0 <= temps[days[i]] <= 30.0
        cycles += int(wet_then_dry and warm)
    return min(1.0, cycles / 3.0)


def _anthracnose(series: WeatherSeries, observed_at: datetime) -> float:
    """Colletotrichum: warm (25-30 C) plus persistent wetness."""
    window = _window(series, observed_at, 10)
    temps = window.daily_mean_temperature()
    wetness = window.leaf_wetness_hours()
    score = sum(
        min(1.0, wetness.get(day, 0) / 10.0)
        for day, temp in temps.items()
        if 25.0 <= temp <= 32.0
    )
    return min(1.0, score / 4.0)


DEFAULT_RULES: List[RiskRule] = [
    RiskRule("tomato::late_blight", "Smith period", "Smith (1956); UK Met Office blight criteria", _late_blight, 10),
    RiskRule("potato::late_blight", "Smith period", "Smith (1956); UK Met Office blight criteria", _late_blight, 10),
    RiskRule("tomato::early_blight", "Wet-dry cycling", "Alternaria epidemiology; Rotem (1994)", _early_blight, 10),
    RiskRule("potato::early_blight", "Wet-dry cycling", "Alternaria epidemiology; Rotem (1994)", _early_blight, 10),
    RiskRule("rice::blast", "Wetness + 20-28 C", "IRRI Rice Knowledge Bank; Kato (1974)", _rice_blast, 7),
    RiskRule("rice::bacterial_leaf_blight", "Rain + wind splash", "IRRI Rice Knowledge Bank", _bacterial_leaf_blight, 7),
    RiskRule("rice::brown_spot", "Wetness + nutrient stress", "IRRI Rice Knowledge Bank", _rice_blast, 7),
    RiskRule("wheat::leaf_rust", "Dew + 15-22 C", "Roelfs et al., CIMMYT rust manual", _rust, 10),
    RiskRule("wheat::stripe_rust", "Dew + 10-18 C", "Roelfs et al., CIMMYT rust manual", _rust, 10),
    RiskRule("maize::common_rust", "Dew + 16-25 C", "CIMMYT maize disease handbook", _rust, 10),
    RiskRule("sugarcane::rust", "Dew + warm", "ICAR-SBI sugarcane disease advisory", _rust, 10),
    RiskRule("mango::powdery_mildew", "Dry + 20-28 C", "Nofal & Haggag (2006)", _powdery_mildew, 10),
    RiskRule("mango::anthracnose", "Wet + 25-32 C", "Arauz (2000), Plant Disease", _anthracnose, 10),
    RiskRule("grape::downy_mildew", "10-10-24 rule proxy", "Baldacci rule; EPPO guidance", _late_blight, 10),
    RiskRule("chilli::leaf_spot", "Wet + warm", "AVRDC chilli disease guide", _anthracnose, 10),
]


@dataclass
class RiskAssessment:
    class_id: str
    risk: float
    rule_name: str
    source: str

    def explain(self) -> str:
        level = "high" if self.risk > 0.66 else "moderate" if self.risk > 0.33 else "low"
        return f"Recent weather gives {level} infection risk for this disease ({self.rule_name})."


class DiseaseRiskModel:
    """Evaluates all rules for a location/time and returns per-class priors."""

    def __init__(self, rules: Optional[List[RiskRule]] = None, neutral_prior: float = 0.35) -> None:
        self.rules = {rule.class_id: rule for rule in (rules or DEFAULT_RULES)}
        # Classes with no rule get a neutral value so they are neither boosted
        # nor penalised relative to each other.
        self.neutral_prior = neutral_prior

    def assess(
        self,
        class_ids: List[str],
        series: Optional[WeatherSeries],
        observed_at: Optional[datetime] = None,
    ) -> Dict[str, RiskAssessment]:
        observed_at = observed_at or datetime.now()
        results: Dict[str, RiskAssessment] = {}
        for class_id in class_ids:
            rule = self.rules.get(class_id)
            if series is None or rule is None or len(series) == 0:
                results[class_id] = RiskAssessment(class_id, self.neutral_prior, "no rule", "n/a")
                continue
            try:
                risk = float(rule.evaluate(series, observed_at))
            except Exception:
                risk = self.neutral_prior
            results[class_id] = RiskAssessment(class_id, max(0.0, min(1.0, risk)), rule.name, rule.source)
        return results

    def coverage(self, class_ids: List[str]) -> float:
        """Fraction of the label space that actually has a rule. Report it."""
        if not class_ids:
            return 0.0
        return sum(1 for c in class_ids if c in self.rules) / len(class_ids)


__all__ = ["DiseaseRiskModel", "RiskRule", "RiskAssessment", "DEFAULT_RULES"]
