"""HTTP client helper with retries, backoff and Retry-After handling."""
from __future__ import annotations

import random
import time

import httpx

from btc5m.config import ca_bundle
from btc5m.logging import get_logger

log = get_logger("http")
UA = "btc5m-research/0.1 (read-only market data research)"


def client(timeout: float = 30.0) -> httpx.Client:
    return httpx.Client(timeout=timeout, verify=ca_bundle(), headers={"User-Agent": UA}, follow_redirects=True)


def async_client(timeout: float = 30.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, verify=ca_bundle(), headers={"User-Agent": UA}, follow_redirects=True)


def get_json(c: httpx.Client, url: str, params=None, retries: int = 6, min_interval: float = 0.0):
    """GET with exponential backoff. Honors Retry-After on 429/503. Raises on final failure."""
    delay = 1.0
    for attempt in range(retries):
        try:
            r = c.get(url, params=params)
            if r.status_code == 200:
                if min_interval:
                    time.sleep(min_interval)
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.replace(".", "").isdigit() else delay
                log.warning("http_retry", url=url, status=r.status_code, wait=wait, attempt=attempt)
                time.sleep(wait + random.random() * 0.5)
                delay = min(delay * 2, 60)
                continue
            r.raise_for_status()
        except (httpx.TransportError, httpx.TimeoutException) as e:
            log.warning("http_transport_error", url=url, err=repr(e), attempt=attempt)
            time.sleep(delay + random.random())
            delay = min(delay * 2, 60)
    raise RuntimeError(f"GET failed after {retries} attempts: {url} {params}")
