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
    # The adapter may retain why an otherwise matching account was excluded.
    # This stays in the in-process snapshot; the adapter is responsible for
    # publishing only its redacted projection on the original receipt.
    ineligible_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (self.account, self.route, self.model, self.operation):
            _name(value)
        if type(self.enabled) is not bool or not self.needs:
            raise InvalidSnapshot('invalid account offer')
        _unique_needs(self.needs)
        _integer(self.preference)
        if (not isinstance(self.ineligible_reasons, tuple)
                or any(not isinstance(reason, str) or not reason for reason in self.ineligible_reasons)
                or len(set(self.ineligible_reasons)) != len(self.ineligible_reasons)):
            raise InvalidSnapshot('invalid account offer diagnostics')

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
    # Scheduled work may rotate within one authenticated source.  Legacy
    # receipts leave this unset and retain the owner/sequence ordering below.
    fair_group: str | None = None
    needs: tuple[Need, ...] = () # global worker/memory demands, distinct from Offer
    fair_subgroup: str | None = None  # existing text/image lanes within a trusted root
    # Trusted product-parent classification from PoolAdmission.  Planner never
    # infers this from a workflow/fairness string supplied by another feature.
    product_lane: str | None = None

    def __post_init__(self) -> None:
        for value in (self.source, self.route, self.model, self.operation):
            _name(value)
        _integer(self.sequence)
        _number(self.ready_at)
        if type(self.payload_saved) is not bool or self.state not in _STATES:
            raise InvalidSnapshot('invalid request state')
        for value in (self.bound_account, self.order_group, self.fair_group, self.fair_subgroup):
            if value is not None:
                _name(value)
        if self.product_lane not in {None, 'new', 'continuation'}:
            raise InvalidSnapshot('invalid product lane')
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
    last_owner_by_source: tuple[tuple[str, str], ...] = ()
    last_group_by_source: tuple[tuple[str, str], ...] = ()
    # One already accepted request may use a temporary second Chat turn. The
    # extra place is never offered to another request or another account.
    scoped_extra_turn: tuple[RequestRef, str] | None = None
    last_subgroup_by_group: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _name(self.revision)
        _number(self.observed_at)
        _number(self.expires_at)
        if self.expires_at < self.observed_at:
            raise InvalidSnapshot('invalid observation interval')
        for value in (self.last_source, self.last_account):
            if value is not None:
                _name(value)
        for source, owner in self.last_owner_by_source:
            _name(source)
            _name(owner)
        for source, group in (*self.last_group_by_source, *self.last_subgroup_by_group):
            _name(source)
            _name(group)
        if self.scoped_extra_turn is not None:
            ref, resource = self.scoped_extra_turn
            if not isinstance(ref, RequestRef):
                raise InvalidSnapshot('invalid scoped extra turn request')
            _name(resource)


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
    offer_diagnostics: tuple['OfferDiagnostic', ...] = ()


@dataclass(frozen=True)
class ResourceBlock:
    """One observed resource constraint for a candidate offer.

    These are private planner facts. The durable adapter hashes the account and
    resource identifiers before they leave this snapshot.
    """
    resource: str
    capacity: int
    occupied: int | None
    required_units: int
    next_at: float | None
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _name(self.resource)
        _integer(self.capacity)
        if self.occupied is not None:
            _integer(self.occupied)
        _integer(self.required_units, minimum=1)
        if self.next_at is not None:
            _number(self.next_at)
        if (not isinstance(self.blockers, tuple)
                or not self.blockers
                or any(not isinstance(blocker, str) or not blocker for blocker in self.blockers)
                or len(set(self.blockers)) != len(self.blockers)):
            raise InvalidSnapshot('invalid resource diagnostics')


@dataclass(frozen=True)
class OfferDiagnostic:
    account: str
    ineligible_reasons: tuple[str, ...] = ()
    resource_blocks: tuple[ResourceBlock, ...] = ()

    def __post_init__(self) -> None:
        _name(self.account)
        if (not isinstance(self.ineligible_reasons, tuple)
                or any(not isinstance(reason, str) or not reason for reason in self.ineligible_reasons)
                or len(set(self.ineligible_reasons)) != len(self.ineligible_reasons)
                or not isinstance(self.resource_blocks, tuple)
                or any(not isinstance(block, ResourceBlock) for block in self.resource_blocks)):
            raise InvalidSnapshot('invalid offer diagnostics')


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


def fairness_lane(fair_group: str | None) -> str:
    """Return the persisted subgroup cursor without colliding with legacy."""
    return 'legacy' if fair_group is None else 'workflow:' + fair_group


def fairness_subgroup_key(source: str, group: str | None) -> str:
    # Length-prefix the authenticated source; caller root IDs cannot collide.
    return str(len(source)) + ':' + source + ':' + fairness_lane(group)


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
    def select_request(request: WaitingRequest) -> Dispatch | None:
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
            return None

        candidates = [o for o in offers
                      if (o.route, o.model, o.operation) ==
                      (request.route, request.model, request.operation)
                      and (request.bound_account is None or request.bound_account == o.account)]
        matches = [o for o in candidates if o.enabled]
        account_order = _rotate([o.account for o in matches], snapshot.last_account)
        position = {name: i for i, name in enumerate(account_order)}
        matches.sort(key=lambda o: (o.preference, position[o.account]))
        possible_wakeups: list[float] = []
        offer_diagnostics: list[OfferDiagnostic] = []
        if not candidates and request.bound_account:
            # This only establishes the absence of a matching offer in this
            # snapshot, not why an account or model is unavailable.
            offer_diagnostics.append(OfferDiagnostic(
                request.bound_account, ('matching_offer_missing',)))
        for offer in candidates:
            if not offer.enabled and offer.ineligible_reasons:
                offer_diagnostics.append(OfferDiagnostic(offer.account, offer.ineligible_reasons))
        if not matches:
            reasons.add('bound_account_unavailable' if request.bound_account else 'no_matching_account')
        # Keep this exact sorted match order for both dispatch and diagnostics.
        # Disabled offers are eligibility-only evidence: evaluating their needs
        # would make a diagnostic path change the former selection/raise
        # behavior for malformed or duplicate demand combinations.
        for offer in matches:
            needs = request.needs + offer.needs
            _unique_needs(needs)
            blockers: set[str] = set()
            future_times: list[float] = []
            reservations: list[tuple[str, float]] = []
            resource_blocks: list[ResourceBlock] = []
            for need in needs:
                resource = resources[need.resource]
                resource_blockers: set[str] = set()
                if resource.occupied is None:
                    blockers.add('occupancy_unknown')
                    resource_blockers.add('occupancy_unknown')
                elif resource.occupied + need.units > resource.capacity:
                    blockers.add('resource_full')
                    resource_blockers.add('resource_full')
                elif (snapshot.scoped_extra_turn is not None
                      and need.resource == snapshot.scoped_extra_turn[1]
                      and resource.occupied >= 1
                      and request.ref != snapshot.scoped_extra_turn[0]):
                    blockers.add('resource_full')
                    resource_blockers.add('resource_full')
                if resource.next_at is None:
                    blockers.add('availability_time_unknown')
                    resource_blockers.add('availability_time_unknown')
                elif resource.next_at > now:
                    blockers.add('account_or_shared_pacing')
                    resource_blockers.add('account_or_shared_pacing')
                    future_times.append(resource.next_at)
                if resource.interval:
                    reservations.append((resource.key, now + resource.interval))
                if resource_blockers:
                    resource_blocks.append(ResourceBlock(
                        resource.key, resource.capacity, resource.occupied,
                        need.units, resource.next_at, tuple(sorted(resource_blockers))))
            if not blockers:
                return Dispatch(request.ref, offer.account, request.source,
                                snapshot.revision, needs, tuple(reservations), now)
            reasons.update(blockers)
            offer_diagnostics.append(OfferDiagnostic(offer.account, resource_blocks=tuple(resource_blocks)))
            # When occupancy is full/unknown, a release/readback is needed.
            # A guessed timer cannot promise that an in-flight job is done.
            if blockers <= {'account_or_shared_pacing'} and future_times:
                possible_wakeups.append(max(future_times))
        wake = min(possible_wakeups) if possible_wakeups else None
        deferred.append(Deferred(request.ref, tuple(sorted(reasons)), wake, tuple(offer_diagnostics)))
        if wake is not None:
            wakeups.append(wake)
        return None

    def scan(candidate_requests):
        """Select within a phase while retaining the normal source rotation."""
        sources = _rotate([r.source for r in candidate_requests], snapshot.last_source)
        for source in sources:
            source_pending = [r for r in candidate_requests if r.source == source]
            owners = _rotate([r.ref.owner for r in source_pending],
                             dict(snapshot.last_owner_by_source).get(source))
            owner_position = {owner: index for index, owner in enumerate(owners)}
            grouped: dict[str, list[WaitingRequest]] = {}
            for request in source_pending:
                grouped.setdefault(fairness_lane(request.fair_group), []).append(request)
            for lane in _rotate(list(grouped), dict(snapshot.last_group_by_source).get(source)):
                subgroups: dict[str, list[WaitingRequest]] = {}
                for request in grouped[lane]:
                    subgroups.setdefault(fairness_lane(request.fair_subgroup), []).append(request)
                cursor = fairness_subgroup_key(source, grouped[lane][0].fair_group)
                for subgroup in _rotate(list(subgroups), dict(snapshot.last_subgroup_by_group).get(cursor)):
                    requests_in_group = sorted(subgroups[subgroup],
                                               key=lambda r: (owner_position[r.ref.owner], r.sequence, r.ref))
                    for request in requests_in_group:
                        dispatch = select_request(request)
                        if dispatch is not None:
                            return dispatch
        return None

    # Existing continuations and true non-product traffic share the established
    # source/group rotation.  Fresh products are a separate last phase, so a
    # cursor pointing at a product-new workflow cannot jump ahead of ready
    # continuations. `product_lane` is trusted adapter data, never a name
    # inferred from another workflow's fair_group.
    primary = [request for request in pending if request.product_lane != 'new']
    fresh_products = [request for request in pending if request.product_lane == 'new']
    dispatch = scan(primary)
    if dispatch is None:
        dispatch = scan(fresh_products)
    if dispatch is not None:
        return Selection(dispatch, tuple(deferred), None)
    return Selection(None, tuple(deferred), min(wakeups) if wakeups else None)
