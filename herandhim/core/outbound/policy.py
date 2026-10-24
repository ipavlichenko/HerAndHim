"""Immutable D-029 policy schema and request validation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from .types import OutboundPurpose


class PolicyError(RuntimeError):
    """A request was rejected locally before network I/O."""


_DEFAULT_LIMITS: dict[OutboundPurpose, tuple[int, int, int, int]] = {
    OutboundPurpose.DIALOGUE: (12_000, 1_600, 4, 90),
    OutboundPurpose.VISION: (6_000, 400, 1, 90),
    OutboundPurpose.IMAGE_GENERATION: (6_000, 0, 1, 180),
    OutboundPurpose.MEMORY_CONSOLIDATION: (3_000, 800, 1, 30),
    OutboundPurpose.RELATIONSHIP_ANALYSIS: (2_000, 256, 1, 30),
}


@dataclass(frozen=True)
class PurposePolicy:
    purpose: OutboundPurpose
    model: str
    allowed_providers: tuple[str, ...]
    active_provider: str
    max_input_tokens_estimated: int
    max_output_tokens: int
    max_calls: int
    connect_timeout_seconds: int
    first_byte_timeout_seconds: int
    total_timeout_seconds: int
    max_cost_usd_per_call: float
    max_cost_usd_per_day: float
    allowed_reference_assets: tuple[str, ...] = ()
    transport: str = "chat_completions"
    resolution: str = ""
    aspect_ratio: str = ""
    max_images: int = 0

    @property
    def provider_block(self) -> Mapping[str, Any]:
        slug = self.active_provider
        return MappingProxyType({
            "only": (slug,),
            "order": (slug,),
            "allow_fallbacks": False,
            "data_collection": "deny",
            "zdr": True,
            "require_parameters": True,
        })


@dataclass(frozen=True)
class OutboundPolicy:
    version: int
    trace_retention_days: int
    aggregate_retention_days: int
    purposes: Mapping[OutboundPurpose, PurposePolicy]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OutboundPolicy":
        if int(raw.get("version", 0)) != 1:
            raise PolicyError("outboundPolicy.version must equal 1")
        openrouter = raw.get("openrouter")
        required_flags = (
            "requireAccountPromptLoggingDisabled",
            "requireAccountZdr",
            "requireNonZeroKeyCreditLimit",
        )
        if not isinstance(openrouter, Mapping) or not all(openrouter.get(flag) is True for flag in required_flags):
            raise PolicyError("OpenRouter account privacy and credit-limit assertions are required")
        purpose_rows = raw.get("purposes")
        if not isinstance(purpose_rows, Mapping):
            raise PolicyError("outboundPolicy.purposes is required")
        parsed: dict[OutboundPurpose, PurposePolicy] = {}
        for purpose in OutboundPurpose:
            row = purpose_rows.get(purpose.value)
            if not isinstance(row, Mapping):
                raise PolicyError(f"missing outbound policy for {purpose.value}")
            if row.get("enabled") is False:
                continue
            model = str(row.get("model", "")).strip()
            allowed = tuple(str(v).strip() for v in row.get("allowedProviders", ()) if str(v).strip())
            active = str(row.get("activeProvider", "")).strip()
            per_call = _positive_float(row.get("maxCostUsdPerCall"), "maxCostUsdPerCall", purpose)
            per_day = _positive_float(row.get("maxCostUsdPerDay"), "maxCostUsdPerDay", purpose)
            if not model or model.startswith("REQUIRED_"):
                raise PolicyError(f"exact model is required for {purpose.value}")
            if model == "openrouter/auto" or model.endswith((":free", ":floor", ":nitro")):
                raise PolicyError(f"router/fallback model is forbidden for {purpose.value}")
            is_image = purpose is OutboundPurpose.IMAGE_GENERATION
            if not is_image and (not allowed or not active or active not in allowed):
                raise PolicyError(f"active provider must belong to non-empty allowlist for {purpose.value}")
            if is_image and (allowed or active):
                raise PolicyError("image_generation provider pin is unsupported by the dedicated Image API")
            defaults = _DEFAULT_LIMITS[purpose]
            parsed[purpose] = PurposePolicy(
                purpose=purpose,
                model=model,
                allowed_providers=allowed,
                active_provider=active,
                max_input_tokens_estimated=_positive_int(row.get("maxInputTokensEstimated", defaults[0])),
                max_output_tokens=_non_negative_int(row.get("maxOutputTokens", defaults[1])),
                max_calls=_positive_int(row.get("maxCallsPerTurn", row.get("maxCallsPerWrite", defaults[2]))),
                connect_timeout_seconds=_positive_int(row.get("connectTimeoutSeconds", 5)),
                first_byte_timeout_seconds=_positive_int(row.get("firstByteTimeoutSeconds", 20)),
                total_timeout_seconds=_positive_int(row.get("totalTimeoutSeconds", defaults[3])),
                max_cost_usd_per_call=per_call,
                max_cost_usd_per_day=per_day,
                allowed_reference_assets=tuple(
                    str(v).strip() for v in row.get("allowedReferenceAssets", ())
                    if str(v).strip()
                ),
                transport=str(row.get("transport", "images_api" if is_image else "chat_completions")),
                resolution=str(row.get("resolution", "1K" if is_image else "")),
                aspect_ratio=str(row.get("aspectRatio", "2:3" if is_image else "")),
                max_images=_positive_int(row.get("maxImages", 1)) if is_image else 0,
            )
            if is_image:
                image_policy = parsed[purpose]
                if image_policy.transport != "images_api":
                    raise PolicyError("image_generation transport must be images_api")
                if image_policy.model != "bytedance-seed/seedream-5-0-flash":
                    raise PolicyError("image_generation requires exact Seedream 5 Flash model")
                if image_policy.resolution != "1K" or image_policy.aspect_ratio != "2:3":
                    raise PolicyError("image_generation requires resolution=1K and aspectRatio=2:3")
                if image_policy.max_images != 1:
                    raise PolicyError("image_generation permits exactly one image")
                if len(image_policy.allowed_reference_assets) != 1:
                    raise PolicyError("image_generation requires exactly one allowedReferenceAsset")
        return cls(
            version=1,
            trace_retention_days=_positive_int(raw.get("traceRetentionDays", 30)),
            aggregate_retention_days=_positive_int(raw.get("aggregateRetentionDays", 180)),
            purposes=MappingProxyType(parsed),
        )

    def for_purpose(self, value: object) -> PurposePolicy:
        try:
            purpose = OutboundPurpose.parse(value)
        except ValueError as exc:
            raise PolicyError(str(exc)) from exc
        try:
            return self.purposes[purpose]
        except KeyError as exc:
            raise PolicyError(f"purpose is disabled or not configured: {purpose.value}") from exc


def _positive_int(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError("budget must be an integer") from exc
    if result <= 0:
        raise PolicyError("budget must be positive")
    return result


def _non_negative_int(value: object) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError("budget must be an integer") from exc
    if result < 0:
        raise PolicyError("budget must be non-negative")
    return result


def _positive_float(value: object, field: str, purpose: OutboundPurpose) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PolicyError(f"{field} is required for {purpose.value}") from exc
    if result <= 0:
        raise PolicyError(f"{field} must be positive for {purpose.value}")
    return result
