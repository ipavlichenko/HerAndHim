"""Content-free error classification; never inspect provider messages or bodies."""

from __future__ import annotations

import httpx
from openai import APIConnectionError, APITimeoutError


def classify_failure(exc: Exception) -> dict:
    status = getattr(exc, "status_code", None)
    if type(status) is int and 100 <= status <= 599:
        code = {
            401: "provider_auth_failed", 403: "provider_access_denied",
            402: "provider_payment_required", 429: "provider_rate_limited",
            408: "provider_timeout", 504: "provider_timeout",
        }.get(status)
        if code is None:
            code = "provider_unavailable" if status >= 500 else "provider_request_rejected"
        return {"error_code": code, "http_status": status}

    chain = []
    current = exc
    while current is not None and len(chain) < 8 and all(current is not e for e in chain):
        chain.append(current)
        current = current.__cause__
    if any(isinstance(e, (APITimeoutError, httpx.TimeoutException, TimeoutError)) for e in chain):
        phase = "unknown"
        for error_type, name in (
            (httpx.ConnectTimeout, "connect"), (httpx.ReadTimeout, "read"),
            (httpx.WriteTimeout, "write"), (httpx.PoolTimeout, "pool"),
        ):
            if any(isinstance(e, error_type) for e in chain):
                phase = name
                break
        return {"error_code": "provider_timeout", "timeout_phase": phase}
    if any(isinstance(e, (APIConnectionError, httpx.NetworkError)) for e in chain):
        return {"error_code": "provider_connection_failed"}
    return {"error_code": "provider_request_failed"}
