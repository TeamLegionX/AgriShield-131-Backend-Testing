"""Predictive residue kinetics and the safe-harvest countdown.

This module closes the gap your backend currently has: ``assess_crop_safety()``
takes a residue value as an *input*, and nothing yet estimates that residue from
a spray date. This is that estimator.

The physics, and what is defensible
------------------------------------
Field dissipation of a foliar pesticide is conventionally modelled as
first order in the *surface* compartment:

    dC/dt = -k_eff(t) * C           =>      C(t) = C0 * exp(-integral k_eff dt)

with ``k = ln(2) / DT50``, where DT50 is the experimentally measured field
half-life for that active ingredient on that crop. This is the standard
regulatory treatment (EFSA/FAO dissipation kinetics) and is a genuinely sound
first approximation. What is *not* defensible is treating k as a constant
independent of weather, because DT50 values are reported under specific trial
conditions. So k_eff is modulated by three physically motivated factors:

  1. **Temperature** - Arrhenius behaviour, approximated by a Q10 coefficient:
         k_T = k_ref * Q10 ** ((T - T_ref) / 10),  Q10 ~ 2.0 for hydrolysis and
     microbial degradation over the 15-40 C range.

  2. **Photolysis** - a term proportional to daily solar irradiance, active only
     for photolabile actives. Off unless the compound record says otherwise.

  3. **Rain wash-off** - and this one is *not* a rate term. Wash-off is an
     event: a storm removes a fraction of the surface deposit in an hour, not
     an exponentially decaying amount over a week. Modelled as

         C <- C * (1 - W),  W = W_max * (1 - exp(-R / R50))

     where R is event rainfall in mm. W_max depends on whether the deposit is
     still within its rainfastness window and on formulation. Systemic actives
     that have moved into leaf tissue are largely protected, so their W_max is
     small. Folding rain into an exponential rate term would be physically
     wrong and would systematically under-predict residue after a dry spell.

  4. **Growth dilution** - for rapidly expanding tissue, concentration falls as
     biomass rises even with zero degradation: C <- C * (M_t0 / M_t).

The safety rule that overrides the model
-----------------------------------------
    recommended_harvest = max(model_date_at_MRL, label_PHI_date)

The pre-harvest interval printed on the CIB&RC-registered label is the legal
instrument. A model may only ever recommend waiting **longer**, never shorter.
Encoding that asymmetry in code, not in a footnote, is what makes this feature
responsible rather than reckless — and it is a strong thing to say out loud to
a judging panel.

Every output is an interval, not a point. DT50 values in the literature span a
factor of two or more; a single number would be false precision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from ..fusion.weather import WeatherSeries

R_GAS = 8.314  # J/(mol K)


class DepositType(str, Enum):
    SURFACE = "surface"      # contact fungicide/insecticide, washes off readily
    SYSTEMIC = "systemic"    # absorbed into tissue, largely rainfast once dry
    TRANSLAMINAR = "translaminar"


@dataclass
class PesticideRecord:
    """Physicochemical + regulatory record for one active ingredient on one crop.

    ``source`` is mandatory. A residue prediction with no citable half-life is
    not a prediction, it is a guess with a decimal point.
    """

    active_ingredient: str
    crop: str
    dt50_days: float
    dt50_low: float                  # literature lower bound
    dt50_high: float                 # literature upper bound
    mrl_mg_per_kg: float
    label_phi_days: int
    deposit: DepositType = DepositType.SURFACE
    q10: float = 2.0
    reference_temp_c: float = 25.0
    rainfastness_hours: float = 6.0
    washoff_max: float = 0.55        # surface default; systemic much lower
    washoff_r50_mm: float = 12.0
    photolabile: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError(
                f"PesticideRecord for {self.active_ingredient}/{self.crop} needs a source citation"
            )
        if self.deposit is DepositType.SYSTEMIC and self.washoff_max > 0.2:
            self.washoff_max = 0.12  # systemic actives are largely rainfast once dry


@dataclass
class SprayEvent:
    applied_at: datetime
    active_ingredient: str
    crop: str
    initial_residue_mg_per_kg: float
    notes: str = ""


@dataclass
class ResiduePoint:
    at: datetime
    concentration: float
    concentration_low: float    # fast-degradation bound (DT50 low)
    concentration_high: float   # slow-degradation bound (DT50 high)
    rainfall_mm: float = 0.0
    temperature_c: float = 0.0


@dataclass
class SafeHarvestResult:
    model_clear_date: Optional[date]
    label_phi_date: date
    recommended_date: date
    days_remaining: int
    limiting_factor: str            # "label_phi" | "residue_model" | "never_clears"
    mrl_mg_per_kg: float
    predicted_at_recommended: float
    uncertainty_band: Tuple[float, float]
    series: List[ResiduePoint] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommended_harvest_date": self.recommended_date.isoformat(),
            "days_remaining": self.days_remaining,
            "limiting_factor": self.limiting_factor,
            "model_clear_date": self.model_clear_date.isoformat() if self.model_clear_date else None,
            "label_phi_date": self.label_phi_date.isoformat(),
            "mrl_mg_per_kg": self.mrl_mg_per_kg,
            "predicted_residue_at_harvest": round(self.predicted_at_recommended, 5),
            "uncertainty_band": [round(v, 5) for v in self.uncertainty_band],
            "caveats": self.caveats,
        }


class ResidueKineticsEngine:
    """Weather-modulated first-order dissipation with discrete wash-off events."""

    def __init__(self, records: Dict[Tuple[str, str], PesticideRecord]) -> None:
        #: keyed by (active_ingredient.lower(), crop.lower())
        self.records = {(a.lower(), c.lower()): r for (a, c), r in records.items()}

    def get_record(self, active_ingredient: str, crop: str) -> Optional[PesticideRecord]:
        return self.records.get((active_ingredient.lower(), crop.lower()))

    # ------------------------------------------------------------ simulate
    def simulate(
        self,
        spray: SprayEvent,
        weather: Optional[WeatherSeries] = None,
        horizon_days: int = 45,
        step_hours: int = 1,
        growth_rate_per_day: float = 0.0,
    ) -> List[ResiduePoint]:
        """Integrate the residue curve hour by hour from the spray event."""
        record = self.get_record(spray.active_ingredient, spray.crop)
        if record is None:
            raise KeyError(
                f"no pesticide record for {spray.active_ingredient} on {spray.crop}; "
                "refuse to extrapolate rather than invent a half-life"
            )

        weather_index = _index_weather(weather)
        points: List[ResiduePoint] = []
        concentrations = {
            "mid": spray.initial_residue_mg_per_kg,
            "low": spray.initial_residue_mg_per_kg,
            "high": spray.initial_residue_mg_per_kg,
        }
        dt50_variants = {"mid": record.dt50_days, "low": record.dt50_low, "high": record.dt50_high}

        cursor = spray.applied_at
        end = spray.applied_at + timedelta(days=horizon_days)
        dt_days = step_hours / 24.0

        while cursor <= end:
            temp, rain, radiation = weather_index(cursor)
            hours_since_spray = (cursor - spray.applied_at).total_seconds() / 3600.0

            for key, dt50 in dt50_variants.items():
                k = math.log(2.0) / max(dt50, 1e-3)
                k *= record.q10 ** ((temp - record.reference_temp_c) / 10.0)
                if record.photolabile:
                    # Radiation in W/m^2; ~250 is a typical daytime mean in the
                    # tropics, used here as the normalising reference.
                    k *= 1.0 + 0.4 * (radiation / 250.0)
                concentrations[key] *= math.exp(-k * dt_days)

                if rain > 0.2:
                    washoff = _washoff_fraction(record, rain, hours_since_spray)
                    concentrations[key] *= 1.0 - washoff

                if growth_rate_per_day > 0:
                    concentrations[key] /= 1.0 + growth_rate_per_day * dt_days

            points.append(
                ResiduePoint(
                    at=cursor,
                    concentration=concentrations["mid"],
                    concentration_low=concentrations["low"],   # faster decay -> lower residue
                    concentration_high=concentrations["high"],
                    rainfall_mm=rain,
                    temperature_c=temp,
                )
            )
            cursor += timedelta(hours=step_hours)

        return points

    # -------------------------------------------------------- safe harvest
    def safe_harvest(
        self,
        spray: SprayEvent,
        weather: Optional[WeatherSeries] = None,
        horizon_days: int = 45,
        safety_factor: float = 1.0,
        growth_rate_per_day: float = 0.0,
    ) -> SafeHarvestResult:
        """Earliest date at which predicted residue is at or below the MRL,
        never earlier than the label PHI.

        ``safety_factor`` < 1 tightens the target below the MRL (e.g. 0.7 for
        export consignments where the destination limit is stricter or sampling
        variability must be absorbed).
        """
        record = self.get_record(spray.active_ingredient, spray.crop)
        if record is None:
            raise KeyError(f"no pesticide record for {spray.active_ingredient} on {spray.crop}")

        series = self.simulate(spray, weather, horizon_days, growth_rate_per_day=growth_rate_per_day)
        target = record.mrl_mg_per_kg * safety_factor

        # Use the pessimistic (slow-degradation) bound to decide clearance.
        model_clear: Optional[date] = None
        for point in series:
            if point.concentration_high <= target:
                model_clear = point.at.date()
                break

        label_phi_date = (spray.applied_at + timedelta(days=record.label_phi_days)).date()

        caveats = [
            "Estimate from a literature field half-life, not a measurement. "
            "Only laboratory residue analysis is definitive.",
            f"Half-life used: {record.dt50_days:.1f} d (range {record.dt50_low:.1f}-{record.dt50_high:.1f} d). "
            f"Source: {record.source}",
        ]

        if model_clear is None:
            recommended = max(label_phi_date, (spray.applied_at + timedelta(days=horizon_days)).date())
            limiting = "never_clears"
            caveats.append(
                f"Predicted residue does not fall below the MRL within {horizon_days} days. "
                "Do not harvest for this market; consult an agronomist."
            )
        elif label_phi_date >= model_clear:
            recommended = label_phi_date
            limiting = "label_phi"
            caveats.append(
                "The legal pre-harvest interval on the product label is the binding constraint. "
                "The model never shortens it."
            )
        else:
            recommended = model_clear
            limiting = "residue_model"
            caveats.append(
                "Weather since spraying slowed dissipation, so the model recommends waiting "
                "beyond the label interval."
            )

        at_harvest = _interpolate_at(series, recommended)
        return SafeHarvestResult(
            model_clear_date=model_clear,
            label_phi_date=label_phi_date,
            recommended_date=recommended,
            days_remaining=max(0, (recommended - datetime.now().date()).days),
            limiting_factor=limiting,
            mrl_mg_per_kg=record.mrl_mg_per_kg,
            predicted_at_recommended=at_harvest[0],
            uncertainty_band=(at_harvest[1], at_harvest[2]),
            series=series,
            caveats=caveats,
        )


# ------------------------------------------------------------------ helpers
def _washoff_fraction(record: PesticideRecord, rain_mm: float, hours_since_spray: float) -> float:
    """Fraction of the current deposit removed by a rain event of ``rain_mm``."""
    w_max = record.washoff_max
    if hours_since_spray < record.rainfastness_hours:
        # Deposit has not dried/bound yet - substantially more vulnerable.
        w_max = min(0.95, w_max * 1.6)
    return w_max * (1.0 - math.exp(-rain_mm / max(record.washoff_r50_mm, 1e-3)))


def _index_weather(weather: Optional[WeatherSeries]):
    """Return a callable ``t -> (temp_c, rain_mm_this_hour, radiation)``.

    Falls back to a benign climatological default when offline, and the caller
    is expected to surface that fallback in the UI: the estimate is weaker
    without real weather and the farmer should know.
    """
    if weather is None or len(weather) == 0:
        return lambda _t: (28.0, 0.0, 200.0)

    lookup = {
        stamp.replace(minute=0, second=0, microsecond=0): (temp, rain, rad)
        for stamp, temp, rain, rad in zip(
            weather.timestamps,
            weather.temperature_c,
            weather.precipitation_mm,
            weather.shortwave_radiation,
        )
    }

    def query(t: datetime) -> Tuple[float, float, float]:
        return lookup.get(t.replace(minute=0, second=0, microsecond=0), (28.0, 0.0, 200.0))

    return query


def _interpolate_at(series: Sequence[ResiduePoint], target: date) -> Tuple[float, float, float]:
    for point in series:
        if point.at.date() >= target:
            return point.concentration, point.concentration_low, point.concentration_high
    last = series[-1]
    return last.concentration, last.concentration_low, last.concentration_high


# ------------------------------------------------------------------ registry
def load_records_from_csv(path: str) -> Dict[Tuple[str, str], PesticideRecord]:
    """Load the pesticide registry from CSV.

    Expected columns:
      active_ingredient, crop, dt50_days, dt50_low, dt50_high, mrl_mg_per_kg,
      label_phi_days, deposit, rainfastness_hours, washoff_max, photolabile, source

    Extend ``data/mrl_data.csv`` with the four kinetics columns rather than
    creating a second file, so the MRL row and its half-life stay together and
    cannot drift apart.
    """
    import csv

    records: Dict[Tuple[str, str], PesticideRecord] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            record = PesticideRecord(
                active_ingredient=row["active_ingredient"],
                crop=row["crop"],
                dt50_days=float(row["dt50_days"]),
                dt50_low=float(row.get("dt50_low") or row["dt50_days"]),
                dt50_high=float(row.get("dt50_high") or row["dt50_days"]),
                mrl_mg_per_kg=float(row["mrl_mg_per_kg"]),
                label_phi_days=int(row["label_phi_days"]),
                deposit=DepositType(row.get("deposit", "surface")),
                rainfastness_hours=float(row.get("rainfastness_hours") or 6.0),
                washoff_max=float(row.get("washoff_max") or 0.55),
                photolabile=str(row.get("photolabile", "")).lower() in {"1", "true", "yes"},
                source=row["source"],
            )
            records[(record.active_ingredient, record.crop)] = record
    return records


__all__ = [
    "ResidueKineticsEngine",
    "PesticideRecord",
    "SprayEvent",
    "SafeHarvestResult",
    "ResiduePoint",
    "DepositType",
    "load_records_from_csv",
]
