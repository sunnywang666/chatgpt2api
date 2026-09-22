"""Pure admission-selection candidate for the existing Provider.

NOT a second queue, HTTP service, account store or production scheduler.
An adapter must build a trusted, consistent snapshot from the original receipts,
account state and clocks. After choosing, it must atomically reserve against the
returned revision in that same authority *before* any external send. The planner
has no network, file, clock, retry or credential access. It never makes a limit up.

Resources are separate constraints, not quantities to add together: a Chat
image submission may require an account turn, an image lifecycle reservation and
an execution worker. Text can use the same turn constraint without an image slot.
The owning implementation releases each resource at its own proven boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
from math import isfinite
from typing import Literal

State = Literal['queued', 'active', 'unknown', 'succeeded', 'failed', 'cancelled']
_STATES = frozenset({'queued', 'active', 'unknown', 'succeeded', 'failed', 'cancelled'})
_TERMINAL = frozenset({'succeeded', 'failed', 'cancelled'})


class InvalidSnapshot(ValueError):
    """Do not dispatch on malformed or contradictory authority data."""


def _name(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 300 or any(ord(c) < 32 for c in value):
        raise InvalidSnapshot('invalid identifier')


def _number(value: float, *, nonnegative: bool = True) -> None:
    try:
        finite = not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise InvalidSnapshot('invalid finite number')
    if nonnegative and value < 0:
        raise InvalidSnapshot('negative number')


def _integer(value: int, *, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum or value > (2**63 - 1):
        raise InvalidSnapshot('invalid integer')


@dataclass(frozen=True, order=True)
class RequestRef:
    owner: str
    kind: str
    request_id: str

    def __post_init__(self) -> None:
        for value in (self.owner, self.kind, self.request_id):
            _name(value)


@dataclass(frozen=True)
class Need:
    resource: str
    units: int = 1

    def __post_init__(self) -> None:
        _name(self.resource)
        _integer(self.units, minimum=1)


@dataclass(frozen=True)
class Resource:
    key: str
    capacity: int
    occupied: int | None
    next_at: float | None
    interval: float = 0.0

    def __post_init__(self) -> None:
        _name(self.key)
        _integer(self.capacity)
        if self.occupied is not None:
            _integer(self.occupied)
        if self.next_at is not None:
            _number(self.next_at)
        _number(self.interval)
        # occupied > capacity is allowed: capacity may shrink under in-flight work.
        # It admits nothing new. It does not kill or free that original work.


@dataclass(frozen=True)
class Offer:
    account: str                 # stable upstream identity, never access token
    route: str
    model: str
    operation: str
    needs: tuple[Need, ...]
    enabled: bool = True
    preference: int = 0          # server-owned, only after source/request fairness

    def __post_init__(self) -> None:
        for value in (self.account, self.route, self.model, self.operation):
            _name(value)
        if type(self.enabled) is not bool or not self.needs:
            raise InvalidSnapshot('invalid account offer')
        _unique_needs(self.needs)
        _integer(self.preference)

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.account, self.route, self.model, self.operation


@dataclass(frozen=True)
class WaitingRequest:
    ref: RequestRef
    source: str                  # from authenticated source, not client priority
    sequence: int                # authoritative acceptance order, not client clock
    route: str
    model: str
    operation: str
    payload_saved: bool
    state: State = 'queued'
    ready_at: float = 0.0
    bound_account: str | None = None
    order_group: str | None = None
    needs: tuple[Need, ...] = () # global worker/memory demands, distinct from Offer

    def __post_init__(self) -> None:
        for value in (self.source, self.route, self.model, self.operation):
            _name(value)
        _integer(self.sequence)
        _number(self.ready_at)
        if type(self.payload_saved) is not bool or self.state not in _STATES:
            raise InvalidSnapshot('invalid request state')
        for value in (self.bound_account, self.order_group):
            if value is not None:
                _name(value)
        _unique_needs(self.needs)


@dataclass(frozen=True)
class Snapshot:
    revision: str                # must cover accounts, clocks, receipts and cursor
    observed_at: float
    expires_at: float
    resources: tuple[Resource, ...]
    offers: tuple[Offer, ...]
    requests: tuple[WaitingRequest, ...]
    last_source: str | None = None
    last_account: str | None = None

    def __post_init__(self) -> None:
        _name(self.revision)
        _number(self.observed_at)
        _number(self.expires_at)
        if self.expires_at < self.observed_at:
            raise InvalidSnapshot('invalid observation interval')
        for value in (self.last_source, self.last_account):
            if value is not None:
                _name(value)


@dataclass(frozen=True)
class Dispatch:
    ref: RequestRef
    account: str
    source: str
    expected_revision: str
    needs: tuple[Need, ...]
    pacing_preview: tuple[tuple[str, float], ...]  # diagnostic only; actual clock stays with sending-edge pacer
    selected_at: float


@dataclass(frozen=True)
class Deferred:
    ref: RequestRef
    reasons: tuple[str, ...]
    next_at: float | None


@dataclass(frozen=True)
class Selection:
    dispatch: Dispatch | None
    deferred: tuple[Deferred, ...]
    next_wake_at: float | None
    snapshot_problem: str | None = None


def _unique_needs(needs: tuple[Need, ...]) -> None:
    if not isinstance(needs, tuple) or any(not isinstance(n, Need) for n in needs):
        raise InvalidSnapshot('needs must be a tuple of Need')
    if len({n.resource for n in needs}) != len(needs):
        raise InvalidSnapshot('duplicate resource demand')


def _deduplicate(items, key):
    result = {}
    for item in items:
        identity = key(item)
        previous = result.get(identity)
        if previous is not None and previous != item:
            raise InvalidSnapshot('contradictory duplicate snapshot entry')
        result[identity] = item
    return result


def _rotate(values: list[str], after: str | None) -> list[str]:
    values = sorted(set(values))
    if not values or after is None:
        return values
    offset = bisect_right(values, after)
    return values[offset:] + values[:offset]


def choose_next(snapshot: Snapshot, now: float) -> Selection:
    """Return at most one dispatch; the caller MUST reserve it atomically.

    None + next_wake_at means wait for the time and/or a state-change signal.
    None + no time means a state change/reconciliation is needed. Never spin or
    create a replacement request. Sent/unknown work is never a dispatch candidate.
    Fairness cursor advances only on successful reservation, not on this read.
    pacing_preview must NOT be pre-written into the existing pacer; doing so would
    make it wait on its own reservation. Keep the original sending-edge clock,
    using actual start time, as the final authority for pacing.
    """
    _number(now)
    if now < snapshot.observed_at or now >= snapshot.expires_at:
        return Selection(None, (), None, 'snapshot_stale_or_future')
    resources = _deduplicate(snapshot.resources, lambda r: r.key)
    offers = tuple(_deduplicate(snapshot.offers, lambda o: o.key).values())
    requests = tuple(_deduplicate(snapshot.requests, lambda r: r.ref).values())
    # Malformed capability wiring is an error, not a reason to silently bypass it.
    for offer in offers:
        for need in offer.needs:
            if need.resource not in resources:
                raise InvalidSnapshot('offer refers to absent resource')
    for request in requests:
        for need in request.needs:
            if need.resource not in resources:
                raise InvalidSnapshot('request refers to absent resource')

    pending = [r for r in requests if r.state == 'queued']
    order_head: dict[tuple[str, str], WaitingRequest] = {}
    for request in requests:
        if request.order_group is not None and request.state not in _TERMINAL:
            group = (request.ref.owner, request.order_group)
            old = order_head.get(group)
            if old is None or (request.sequence, request.ref) < (old.sequence, old.ref):
                order_head[group] = request

    deferred: list[Deferred] = []
    wakeups: list[float] = []
    sources = _rotate([r.source for r in pending], snapshot.last_source)
    for source in sources:
        source_requests = sorted((r for r in pending if r.source == source),
                                 key=lambda r: (r.sequence, r.ref))
        for request in source_requests:
            reasons: set[str] = set()
            if not request.payload_saved:
                reasons.add('input_not_durable')
            if request.order_group is not None:
                group = (request.ref.owner, request.order_group)
                if order_head[group].ref != request.ref:
                    reasons.add('earlier_group_request_unfinished')
            if request.ready_at > now:
                reasons.add('request_wait_until')
            if reasons:
                wake = request.ready_at if reasons == {'request_wait_until'} else None
                deferred.append(Deferred(request.ref, tuple(sorted(reasons)), wake))
                if wake is not None:
                    wakeups.append(wake)
                continue

            matches = [o for o in offers if o.enabled
                       and (o.route, o.model, o.operation) ==
                           (request.route, request.model, request.operation)
                       and (request.bound_account is None or request.bound_account == o.account)]
            account_order = _rotate([o.account for o in matches], snapshot.last_account)
            position = {name: i for i, name in enumerate(account_order)}
            matches.sort(key=lambda o: (o.preference, position[o.account]))
            possible_wakeups: list[float] = []
            if not matches:
                reasons.add('bound_account_unavailable' if request.bound_account else 'no_matching_account')
            for offer in matches:
                needs = request.needs + offer.needs
                _unique_needs(needs)
                blockers: set[str] = set()
                future_times: list[float] = []
                reservations: list[tuple[str, float]] = []
                for need in needs:
                    resource = resources[need.resource]
                    if resource.occupied is None:
                        blockers.add('occupancy_unknown')
                    elif resource.occupied + need.units > resource.capacity:
                        blockers.add('resource_full')
                    if resource.next_at is None:
                        blockers.add('availability_time_unknown')
                    elif resource.next_at > now:
                        blockers.add('account_or_shared_pacing')
                        future_times.append(resource.next_at)
                    if resource.interval:
                        reservations.append((resource.key, now + resource.interval))
                if not blockers:
                    return Selection(Dispatch(request.ref, offer.account, source,
                                              snapshot.revision, needs,
                                              tuple(reservations), now),
                                     tuple(deferred), None)
                reasons.update(blockers)
                # When occupancy is full/unknown, a release/readback is needed.
                # A guessed timer cannot promise that an in-flight job is done.
                if blockers <= {'account_or_shared_pacing'} and future_times:
                    possible_wakeups.append(max(future_times))
            wake = min(possible_wakeups) if possible_wakeups else None
            deferred.append(Deferred(request.ref, tuple(sorted(reasons)), wake))
            if wake is not None:
                wakeups.append(wake)
    return Selection(None, tuple(deferred), min(wakeups) if wakeups else None)
