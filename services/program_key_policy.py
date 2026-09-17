"""Program-key policy used by the existing Provider authentication and API routes.

The existing credential service must authenticate
and reload the current key before using this module. Routes/capabilities below
are resolved by trusted adapters, never accepted from an HTTP client's claim.
No accounts, credentials, budgets, queues or provider retries are owned here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class Capability(str, Enum):
    CHAT_IMAGE = "chat_image"
    CHAT_TEXT = "chat_text"
    CODEX_CODING = "codex_coding"
    CODEX_IMAGE = "codex_image"


class Route(str, Enum):
    CHAT = "chat"
    CODEX = "codex"


ROUTE_FOR = {
    Capability.CHAT_IMAGE: Route.CHAT,
    Capability.CHAT_TEXT: Route.CHAT,
    Capability.CODEX_CODING: Route.CODEX,
    Capability.CODEX_IMAGE: Route.CODEX,
}
LIMIT_SOURCE_FOR = {
    Capability.CHAT_IMAGE: "chat_image_limit",
    Capability.CHAT_TEXT: "chat_model_limit",
    Capability.CODEX_CODING: "codex_shared_limit",
    Capability.CODEX_IMAGE: "codex_shared_limit",
}


class PolicyError(ValueError):
    """Only stable, non-secret codes are exposed to callers."""
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


@dataclass(frozen=True)
class ProgramKeyPolicy:
    """Stored on the EXISTING key, not in a second policy/credential database."""
    revision: int
    capabilities: frozenset[Capability]

    def __post_init__(self) -> None:
        if not _positive_integer(self.revision):
            raise PolicyError("KEY_POLICY_INVALID_REVISION")
        if (not isinstance(self.capabilities, frozenset) or not self.capabilities
                or any(not isinstance(item, Capability) for item in self.capabilities)):
            raise PolicyError("KEY_POLICY_INVALID_CAPABILITIES")

    @classmethod
    def from_record(cls, value: object) -> "ProgramKeyPolicy":
        if value is None:
            # A missing legacy policy is NOT interpreted as unlimited permission.
            # Deployment must reconcile existing consumers before enabling guards.
            raise PolicyError("KEY_POLICY_RECONCILIATION_REQUIRED")
        if not isinstance(value, dict) or set(value) != {"version", "revision", "capabilities"}:
            raise PolicyError("KEY_POLICY_INVALID_RECORD")
        if type(value["version"]) is not int or value["version"] != 1:
            raise PolicyError("KEY_POLICY_UNSUPPORTED_VERSION")
        items = value["capabilities"]
        if (not isinstance(items, list) or not items
                or any(type(item) is not str for item in items)):
            raise PolicyError("KEY_POLICY_INVALID_CAPABILITIES")
        if len(items) != len(set(items)):
            raise PolicyError("KEY_POLICY_DUPLICATE_CAPABILITY")
        try:
            capabilities = frozenset(Capability(item) for item in items)
        except ValueError:
            raise PolicyError("KEY_POLICY_UNKNOWN_CAPABILITY") from None
        return cls(value["revision"], capabilities)

    def to_record(self) -> dict[str, object]:
        return {"version": 1, "revision": self.revision,
                "capabilities": sorted(item.value for item in self.capabilities)}


def make_policy(capabilities: Sequence[str], *, revision: int,
                ready: frozenset[Capability]) -> ProgramKeyPolicy:
    """A management operation: don't let an unshipped feature look enabled."""
    if not isinstance(capabilities, Sequence) or isinstance(capabilities, (str, bytes)):
        raise PolicyError("KEY_POLICY_INVALID_CAPABILITIES")
    policy = ProgramKeyPolicy.from_record({
        "version": 1, "revision": revision, "capabilities": list(capabilities),
    })
    if not policy.capabilities <= ready:
        raise PolicyError("CAPABILITY_NOT_READY")
    return policy


@dataclass(frozen=True)
class RequestUse:
    """Resolved from the actual destination/adapter and ALL upstream tools."""
    route: Route
    capabilities: frozenset[Capability]

    def __post_init__(self) -> None:
        if (not isinstance(self.route, Route) or not isinstance(self.capabilities, frozenset)
                or not self.capabilities
                or any(not isinstance(item, Capability) for item in self.capabilities)):
            raise PolicyError("REQUEST_USE_INVALID")
        if any(ROUTE_FOR[item] is not self.route for item in self.capabilities):
            raise PolicyError("REQUEST_ROUTE_MISMATCH")


@dataclass(frozen=True)
class TaskBinding:
    """Projection of the existing durable task; not a new task identity."""
    key_id: str
    account_ref: str
    task_id: str
    use: RequestUse

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value.strip()
               for value in (self.key_id, self.account_ref, self.task_id)):
            raise PolicyError("TASK_BINDING_INVALID")
        if not isinstance(self.use, RequestUse):
            raise PolicyError("TASK_BINDING_INVALID")


def authorize_submission(policy: ProgramKeyPolicy, *, key_enabled: bool,
                         use: RequestUse, ready: frozenset[Capability],
                         key_id: str = "", original: TaskBinding | None = None) -> RequestUse:
    """Authorize a new write. This is NOT permission to retry an unknown write."""
    if key_enabled is not True:
        raise PolicyError("KEY_DISABLED")
    if not isinstance(policy, ProgramKeyPolicy) or not isinstance(use, RequestUse):
        raise PolicyError("KEY_POLICY_INVALID_RECORD")
    if original is not None:
        if original.key_id != key_id:
            raise PolicyError("TASK_NOT_FOUND")
        if original.use != use:
            raise PolicyError("TASK_ROUTE_CHANGE_FORBIDDEN")
    if not use.capabilities <= policy.capabilities:
        raise PolicyError("KEY_CAPABILITY_DENIED")
    if not use.capabilities <= ready:
        raise PolicyError("CAPABILITY_NOT_READY")
    return use


def authorize_read(*, key_id: str, key_enabled: bool, original: TaskBinding) -> TaskBinding:
    """Own original receipt/download/read-only reconciliation, never a resubmit.

A narrower policy does not destroy previously accepted receipts. Revocation
still denies subsequent client authentication. Server-owned observation of an
accepted task must continue independently of this client read gate.
"""
    if key_enabled is not True:
        raise PolicyError("KEY_DISABLED")
    if original.key_id != key_id:
        raise PolicyError("TASK_NOT_FOUND")
    return original


def image_use(model: str, *, model_routes: Mapping[str, Route]) -> RequestUse:
    """Use the same explicit model routing table as the actual image adapter."""
    route = model_routes.get(model)
    if not isinstance(route, Route):
        raise PolicyError("MODEL_ROUTE_UNKNOWN")
    capability = Capability.CHAT_IMAGE if route is Route.CHAT else Capability.CODEX_IMAGE
    return RequestUse(route, frozenset({capability}))


def codex_use(payload: Mapping[str, object], *,
              native_tool_capabilities: Mapping[str, frozenset[Capability]]) -> RequestUse:
    """Inspect tools without modifying the original Responses payload.

Tool dispatch is versioned by the native adapter. Unknown tool types fail
closed until that adapter explicitly supports and classifies them. Function
and custom tools execute at the caller; their names/prose are not route proof.
This does not sandbox programs that can contact providers outside our service.
"""
    if not isinstance(payload, Mapping):
        raise PolicyError("REQUEST_TOOL_INVALID")
    required = {Capability.CODEX_CODING}

    def inspect(tool: object, depth: int = 0) -> None:
        # The real HTTP boundary still owns the complete request-size limit.
        if depth > 16:
            raise PolicyError("REQUEST_TOOL_NESTING_EXCEEDED")
        if not isinstance(tool, dict) or not isinstance(tool.get("type"), str):
            raise PolicyError("REQUEST_TOOL_INVALID")
        kind = tool["type"]
        if kind in {"namespace", "allowed_tools"}:
            members = tool.get("tools")
            if not isinstance(members, list):
                raise PolicyError("REQUEST_TOOL_INVALID")
            for member in members:
                inspect(member, depth + 1)
        elif kind not in {"function", "custom"}:
            declared = native_tool_capabilities.get(kind)
            if (not isinstance(declared, frozenset) or not declared
                    or any(not isinstance(cap, Capability)
                           or ROUTE_FOR[cap] is not Route.CODEX for cap in declared)):
                raise PolicyError("NATIVE_TOOL_NOT_CLASSIFIED")
            required.update(declared)

    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise PolicyError("REQUEST_TOOL_INVALID")
    for tool in tools:
        inspect(tool)
    choice = payload.get("tool_choice")
    if choice is not None:
        if isinstance(choice, dict):
            # Forced tool selection is an independent route input, not just a name.
            inspect(choice)
        elif choice not in ("none", "auto", "required"):
            raise PolicyError("REQUEST_TOOL_INVALID")
    return RequestUse(Route.CODEX, frozenset(required))


def shared_limit_identity(account_ref: str, capability: Capability,
                          upstream_limit_id: str, window_id: str) -> tuple[str, str, str, str]:
    """Grouping identity only. No synthetic balance or per-image charge exists."""
    if not isinstance(capability, Capability) or any(
            not isinstance(v, str) or not v.strip()
            for v in (account_ref, upstream_limit_id, window_id)):
        raise PolicyError("LIMIT_IDENTITY_UNKNOWN")
    return (account_ref, LIMIT_SOURCE_FOR[capability], upstream_limit_id, window_id)
