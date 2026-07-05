"""
src/pvgis.py
============
Independent PV-output estimate from **PVGIS** — the European Commission Joint
Research Centre's *Photovoltaic Geographical Information System*.

PVGIS takes a latitude/longitude and a PV-system specification and returns the
expected annual & monthly energy yield, computed from its OWN validated
satellite radiation databases (PVGIS-SARAH2 / SARAH3 / ERA5) — completely
independently of this project's Open-Meteo / NASA POWER inputs.

We use it as a *third-party cross-check* of the project's own physics-based
yield.  Because it is coordinate-driven (exactly like NASA POWER and Open-Meteo)
it fits the project's live design, and it gives a citable, independent number to
validate the yield estimate in a dissertation / paper.

Free, no API key.

Transport hardening (fixes "[WinError 10054] forcibly closed by remote host")
-----------------------------------------------------------------------------
The previous version called PVGIS with a bare `urllib` request, a bot-looking
User-Agent, a single timeout, no retries and a single (now partly deprecated)
unversioned-ish endpoint.  That is the most reset-prone way to reach JRC.  This
version keeps stdlib-only (no new dependency) but adds:
  * A browser-like User-Agent (JRC's edge resets bare script clients).
  * Endpoint fallback: v5_2 -> v5_3 -> legacy, first that answers wins.
  * A retry loop with exponential backoff (transient resets / 5xx / 429).
  * Exception classification, so a TCP reset is reported AS a reset (a firewall
    / antivirus / proxy on the user's network) and NOT as "no internet". That
    distinction is why this bug went misdiagnosed.

The public `pvgis_yield(...)` signature and return dict are unchanged, so
app.py needs no change to keep working (it still does
`except Exception -> {"error": str(exc)}`).  The error string is now accurate.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import config

# PVcalc endpoints, tried in order. First that answers wins.
_PVCALC_BASES = (
    "https://re.jrc.ec.europa.eu/api/v5_2/PVcalc",   # documented stable
    "https://re.jrc.ec.europa.eu/api/v5_3/PVcalc",   # newest (SARAH3)
    "https://re.jrc.ec.europa.eu/api/PVcalc",        # legacy fallback
)
PVGIS_PVCALC_URL = _PVCALC_BASES[0]  # kept for backward compatibility

_TIMEOUT = 60
_MAX_ATTEMPTS_PER_URL = 3
_BACKOFF_BASE = 1.5  # sleep seconds = _BACKOFF_BASE ** attempt

# A real browser UA. JRC's WAF resets bare "python"/"urllib" style clients.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36 pv-wind-impact/1.0"
    ),
    "Accept": "application/json",
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class PVGISError(RuntimeError):
    """Raised on network/HTTP/parse failure with a human-readable, classified
    message.  app.py surfaces str(this) directly in the UI."""


def _classify(exc: BaseException | None) -> str:
    """Turn a transport exception into an accurate, actionable message."""
    if exc is None:
        return "PVGIS unreachable: no endpoint responded."

    reason = getattr(exc, "reason", exc)
    text = (str(reason) + " " + str(exc)).lower()

    if isinstance(exc, (socket.timeout, TimeoutError)) or "timed out" in text:
        return ("PVGIS did not respond in time. The server may be slow, or a "
                "firewall/proxy is silently dropping the connection. Retry, or "
                "test on a different network (e.g. a phone hotspot).")

    if any(m in text for m in ("10054", "forcibly closed", "reset",
                               "remotedisconnected", "connection aborted",
                               "broken pipe")):
        return ("Connection to PVGIS was reset (WinError 10054). This is almost "
                "always a firewall, antivirus HTTPS-inspection, or proxy on your "
                "network resetting traffic to re.jrc.ec.europa.eu - NOT a lack of "
                "internet. Test on a phone hotspot: if PVGIS works there, "
                "whitelist the domain or use an unrestricted network.")

    if any(m in text for m in ("getaddrinfo", "name or service not known",
                               "temporary failure in name resolution",
                               "nodename nor servname", "no address associated")):
        return ("Could not reach PVGIS at all (DNS/connection failure). The "
                "machine appears to be offline or DNS is blocked.")

    return (f"Network error contacting PVGIS: {exc}. Likely a firewall/proxy on "
            "your network - try a different network to confirm.")


def _get_json(params: dict) -> dict:
    """GET PVcalc JSON with endpoint fallback + retries. Raises PVGISError."""
    query = urllib.parse.urlencode(params)
    last_exc: BaseException | None = None

    for base in _PVCALC_BASES:
        url = f"{base}?{query}"
        for attempt in range(_MAX_ATTEMPTS_PER_URL):
            req = urllib.request.Request(url, headers=_HEADERS)
            try:
                with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                # 5xx / 429 are transient -> retry on this same URL.
                if exc.code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    time.sleep(_BACKOFF_BASE ** attempt)
                    continue
                # 4xx -> bad params / location without data. Surface immediately.
                body = ""
                try:
                    body = exc.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                raise PVGISError(
                    f"PVGIS returned HTTP {exc.code}. Usually bad input (tilt/"
                    f"aspect out of range, or a location with no data). "
                    f"Server said: {body}") from exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                # URLError wraps ConnectionResetError / gaierror / socket.timeout.
                last_exc = exc
                time.sleep(_BACKOFF_BASE ** attempt)
                continue
        # exhausted this base -> try the next endpoint
    raise PVGISError(_classify(last_exc))


def pvgis_yield(latitude: float, longitude: float,
                peakpower_kwp: float | None = None, loss: float = 14.0,
                tilt: float | None = None, aspect: float = 0.0,
                pvtech: str = "crystSi", mounting: str = "building") -> dict:
    """
    Query PVGIS PVcalc for a fixed PV system at this coordinate and return an
    independent yield estimate.

    aspect: PVGIS azimuth convention - 0 = south (correct for India / Northern
            Hemisphere), -90 = east, +90 = west.

    Returns a dict with annual energy, specific yield (kWh/kWp), in-plane
    irradiation, a 12-row monthly-energy DataFrame, the radiation database used,
    and a source label.  Raises PVGISError on network/HTTP/parse failure with a
    classified, human-readable message (callers handle it).
    """
    if peakpower_kwp is None:
        peakpower_kwp = config.PV_AREA_M2 * config.PV_EFFICIENCY      # DC nameplate
    if tilt is None:
        tilt = config.PV_TILT_DEG

    params = {
        "lat": round(float(latitude), 4),
        "lon": round(float(longitude), 4),
        "peakpower": round(float(peakpower_kwp), 3),
        "loss": float(loss),
        "angle": float(tilt),
        "aspect": float(aspect),
        "pvtechchoice": pvtech,
        "mountingplace": mounting,
        "pvcalculation": 1,
        "outputformat": "json",
        "browser": 0,
    }

    data = _get_json(params)

    try:
        outputs = data["outputs"]
        totals = outputs["totals"]["fixed"]

        annual_kwh = float(totals["E_y"])
        in_plane = float(totals.get("H(i)_y", float("nan")))

        monthly = pd.DataFrame(
            [{"month": int(m["month"]),
              "Month": MONTHS[int(m["month"]) - 1],
              "Energy_kWh": float(m["E_m"])}
             for m in outputs["monthly"]["fixed"]])

        rad_db = (data.get("inputs", {}).get("meteo_data", {})
                  .get("radiation_db", "PVGIS"))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        # 200 OK but unexpected shape, e.g. "Location without data".
        msg = ""
        if isinstance(data, dict):
            msg = str(data.get("message") or data.get("status") or "")
        raise PVGISError(
            "PVGIS responded but returned no PV yield for this site "
            + (f"({msg})." if msg else "(unexpected response shape).")
        ) from exc

    return {
        "peakpower_kwp": float(peakpower_kwp),
        "annual_energy_kwh": annual_kwh,
        "specific_yield": annual_kwh / peakpower_kwp if peakpower_kwp else float("nan"),
        "in_plane_irradiation_kwh_m2": in_plane,
        "loss_pct": float(loss),
        "tilt_deg": float(tilt),
        "aspect_deg": float(aspect),
        "monthly": monthly,
        "radiation_db": rad_db,
        "source": "PVGIS v5.2 (EU Joint Research Centre)",
    }
