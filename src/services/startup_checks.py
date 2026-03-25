"""Startup checks for API key configuration and provider connectivity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import structlog
from groq import AsyncGroq

from ..config import Settings

logger = structlog.get_logger(__name__)


@dataclass
class KeyCheckResult:
    provider: str
    env_var: str
    required: bool
    configured: bool
    valid: bool
    reason: str
    fingerprint: str


def _fingerprint(secret: str) -> str:
    if not secret:
        return "missing"
    digest = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:10]
    return f"len={len(secret)} sha256={digest}"


def _looks_like(prefix: str, value: str) -> bool:
    return bool(value) and value.startswith(prefix) and len(value) > len(prefix) + 8


async def _validate_groq_key(groq_key: str) -> tuple[bool, str]:
    if not groq_key:
        return False, "missing"

    if not _looks_like("gsk_", groq_key):
        return False, "invalid_format_expected_prefix_gsk_"

    try:
        client = AsyncGroq(api_key=groq_key)
        await client.models.list()
        return True, "ok"
    except Exception as exc:
        message = str(exc).replace("\n", " ").strip()
        if len(message) > 220:
            message = f"{message[:220]}..."
        return False, f"provider_auth_failed: {message}"


def _format_check_results(results: list[KeyCheckResult]) -> None:
    for result in results:
        level = "info"
        if result.required and (not result.configured or not result.valid):
            level = "error"
        elif not result.valid:
            level = "warning"

        getattr(logger, level)(
            "API key startup check",
            provider=result.provider,
            env_var=result.env_var,
            required=result.required,
            configured=result.configured,
            valid=result.valid,
            reason=result.reason,
            key_fingerprint=result.fingerprint,
        )


async def run_startup_key_checks(settings: Settings, fail_on_any_invalid_configured: bool = False) -> bool:
    """Validate API keys at startup and emit safe diagnostic logs."""
    checks: list[KeyCheckResult] = []

    groq_valid, groq_reason = await _validate_groq_key(settings.groq_api_key)
    checks.append(
        KeyCheckResult(
            provider="groq",
            env_var="GROQ_API_KEY",
            required=True,
            configured=bool(settings.groq_api_key),
            valid=groq_valid,
            reason=groq_reason,
            fingerprint=_fingerprint(settings.groq_api_key),
        )
    )

    gemini_key = settings.gemini_api_key
    checks.append(
        KeyCheckResult(
            provider="gemini",
            env_var="GEMINI_API_KEY",
            required=False,
            configured=bool(gemini_key),
            valid=(not gemini_key) or _looks_like("AIza", gemini_key),
            reason="ok" if (not gemini_key) or _looks_like("AIza", gemini_key) else "invalid_format_expected_prefix_AIza",
            fingerprint=_fingerprint(gemini_key),
        )
    )

    elevenlabs_key = settings.elevenlabs_api_key
    checks.append(
        KeyCheckResult(
            provider="elevenlabs",
            env_var="ELEVENLABS_API_KEY",
            required=False,
            configured=bool(elevenlabs_key),
            valid=(not elevenlabs_key) or _looks_like("sk_", elevenlabs_key),
            reason="ok" if (not elevenlabs_key) or _looks_like("sk_", elevenlabs_key) else "invalid_format_expected_prefix_sk_",
            fingerprint=_fingerprint(elevenlabs_key),
        )
    )

    lingo_key = settings.lingodotdev_api_key
    checks.append(
        KeyCheckResult(
            provider="lingodotdev",
            env_var="LINGODOTDEV_API_KEY",
            required=False,
            configured=bool(lingo_key),
            valid=(not lingo_key) or _looks_like("api_", lingo_key),
            reason="ok" if (not lingo_key) or _looks_like("api_", lingo_key) else "invalid_format_expected_prefix_api_",
            fingerprint=_fingerprint(lingo_key),
        )
    )

    uplift_key = settings.upliftai_api_key
    checks.append(
        KeyCheckResult(
            provider="upliftai",
            env_var="UPLIFTAI_API_KEY",
            required=False,
            configured=bool(uplift_key),
            valid=(not uplift_key) or _looks_like("sk_api_", uplift_key),
            reason="ok" if (not uplift_key) or _looks_like("sk_api_", uplift_key) else "invalid_format_expected_prefix_sk_api_",
            fingerprint=_fingerprint(uplift_key),
        )
    )

    _format_check_results(checks)

    if fail_on_any_invalid_configured:
        configured_failures = [check for check in checks if check.configured and not check.valid]
        return not configured_failures

    required_failures = [check for check in checks if check.required and (not check.configured or not check.valid)]
    return not required_failures
