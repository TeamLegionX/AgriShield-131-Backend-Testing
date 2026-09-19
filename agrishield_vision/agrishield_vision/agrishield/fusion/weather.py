"""Open-Meteo client.

Two endpoints, two purposes:
  * ``forecast``  - next 7-16 days, drives the safe-harvest countdown and the
                    forward-looking infection-risk warning.
  * ``archive``   - past N days, drives the *retrospective* disease-risk prior
                    that re-ranks the vision output (you need the conditions
                    during the latent period, not today's weather).

Open-Meteo is used because it needs no API key and permits non-commercial use
freely — which matters for a hackathon demo that has to work on a judge's
network at 9am. Verify the licence terms before any commercial deployment.

Everything here degrades gracefully: offline is the normal state in a field, so
a failed fetch must return ``None`` and the pipeline must fall back to
vision-only. It must never block a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "shortwave_radiation",
    "wind_speed_10m",
    "soil_temperature_0cm",
]


@dataclass
class WeatherSeries:
    """Hourly weather aligned to a single point. All lists are the same length."""

    timestamps: List[datetime]
    temperature_c: List[float]
    relative_humidity: List[float]
    precipitation_mm: List[float]
    shortwave_radiation: List[float]
    wind_speed: List[float]
    latitude: float
    longitude: float
    source: str = "open-meteo"

    def __len__(self) -> int:
        return len(self.timestamps)

    def slice_days(self, start: datetime, days: int) -> "WeatherSeries":
        end = start + timedelta(days=days)
        keep = [i for i, t in enumerate(self.timestamps) if start <= t < end]
        return WeatherSeries(
            timestamps=[self.timestamps[i] for i in keep],
            temperature_c=[self.temperature_c[i] for i in keep],
            relative_humidity=[self.relative_humidity[i] for i in keep],
            precipitation_mm=[self.precipitation_mm[i] for i in keep],
            shortwave_radiation=[self.shortwave_radiation[i] for i in keep],
            wind_speed=[self.wind_speed[i] for i in keep],
            latitude=self.latitude,
            longitude=self.longitude,
            source=self.source,
        )

    def daily_rainfall(self) -> Dict[date, float]:
        totals: Dict[date, float] = {}
        for t, mm in zip(self.timestamps, self.precipitation_mm):
            totals[t.date()] = totals.get(t.date(), 0.0) + (mm or 0.0)
        return totals

    def daily_mean_temperature(self) -> Dict[date, float]:
        sums: Dict[date, List[float]] = {}
        for t, c in zip(self.timestamps, self.temperature_c):
            sums.setdefault(t.date(), []).append(c)
        return {d: sum(v) / len(v) for d, v in sums.items()}

    def leaf_wetness_hours(self, humidity_threshold: float = 90.0) -> Dict[date, int]:
        """Proxy for leaf wetness: hours with RH above threshold, or with rain.

        Real leaf wetness needs a sensor. This proxy is standard in field
        advisories when sensors are absent and should be labelled as a proxy
        wherever it is shown to a user.
        """
        hours: Dict[date, int] = {}
        for t, rh, rain in zip(self.timestamps, self.relative_humidity, self.precipitation_mm):
            wet = (rh or 0) >= humidity_threshold or (rain or 0) > 0.2
            hours[t.date()] = hours.get(t.date(), 0) + int(wet)
        return hours


def _parse(payload: dict, latitude: float, longitude: float) -> WeatherSeries:
    hourly = payload.get("hourly", {})
    stamps = [datetime.fromisoformat(t) for t in hourly.get("time", [])]

    def column(name: str) -> List[float]:
        values = hourly.get(name) or [0.0] * len(stamps)
        return [float(v) if v is not None else 0.0 for v in values]

    return WeatherSeries(
        timestamps=stamps,
        temperature_c=column("temperature_2m"),
        relative_humidity=column("relative_humidity_2m"),
        precipitation_mm=column("precipitation"),
        shortwave_radiation=column("shortwave_radiation"),
        wind_speed=column("wind_speed_10m"),
        latitude=latitude,
        longitude=longitude,
    )


def fetch_weather(
    latitude: float,
    longitude: float,
    past_days: int = 14,
    forecast_days: int = 10,
    timeout: float = 8.0,
) -> Optional[WeatherSeries]:
    """Fetch past + forecast hourly weather. Returns ``None`` on any failure."""
    try:
        import requests
    except ImportError:  # pragma: no cover
        return None

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(HOURLY_VARIABLES),
        "past_days": min(past_days, 92),
        "forecast_days": min(forecast_days, 16),
        "timezone": "auto",
    }
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=timeout)
        response.raise_for_status()
        return _parse(response.json(), latitude, longitude)
    except Exception as exc:  # network, JSON, schema - all non-fatal
        print(f"[weather] fetch failed ({exc}); falling back to vision-only")
        return None


def cache_key(latitude: float, longitude: float, day: date) -> str:
    """Grid-snapped cache key. ~11km cells: weather does not vary meaningfully
    below that for this purpose, and snapping turns thousands of farmer
    coordinates into a few dozen cacheable requests."""
    return f"{round(latitude, 1)}_{round(longitude, 1)}_{day.isoformat()}"


__all__ = ["WeatherSeries", "fetch_weather", "cache_key", "HOURLY_VARIABLES"]
