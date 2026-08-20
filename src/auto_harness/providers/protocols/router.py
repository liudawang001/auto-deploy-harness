"""Fail-closed provider protocol selection."""

from dataclasses import dataclass
from typing import Any, Optional


PROTOCOLS = frozenset({"json_action", "native_tools", "auto"})


class ProviderProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class ProtocolSelection:
    configured_protocol: str
    selected_protocol: str
    reason: str
    native_implemented: bool
    native_capability: bool

    def to_dict(self):
        return {
            "configured_protocol": self.configured_protocol,
            "selected_protocol": self.selected_protocol,
            "reason": self.reason,
            "native_implemented": self.native_implemented,
            "native_capability": self.native_capability,
        }


def select_provider_protocol(
    configured_protocol: str,
    provider: Any,
    capabilities: Optional[Any] = None,
) -> ProtocolSelection:
    configured = str(configured_protocol or "json_action").strip().lower()
    if configured not in PROTOCOLS:
        raise ProviderProtocolError("provider_protocol must be json_action, native_tools, or auto")
    implemented = callable(getattr(provider, "complete_with_tools", None))
    capability = bool(
        getattr(capabilities, "supports_tool_calling", False)
        if capabilities is not None
        else getattr(provider, "native_tool_calling", False)
    )
    if configured == "json_action":
        return ProtocolSelection(
            configured, "json_action", "explicit_json_action",
            implemented, capability,
        )
    if configured == "native_tools":
        if not implemented:
            raise ProviderProtocolError("native_tools requested but provider has no complete_with_tools implementation")
        if not capability:
            raise ProviderProtocolError("native_tools requested but provider/model capability is not enabled")
        return ProtocolSelection(
            configured, "native_tools", "explicit_native_tools",
            implemented, capability,
        )
    if implemented and capability:
        return ProtocolSelection(
            configured, "native_tools", "auto_selected_native_tools",
            implemented, capability,
        )
    return ProtocolSelection(
        configured, "json_action", "auto_selected_json_action",
        implemented, capability,
    )
