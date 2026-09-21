import sys
import unittest
from dataclasses import replace
from pathlib import Path
from services.admission_planner import *


def request(n=0, source='happy', account=None, group=None, state='queued', operation='image', route='chat', model='fixture-model'):
    return WaitingRequest(RequestRef(source, operation, f'request-{n}'), source, n, route, model, operation,
                          True, state=state, bound_account=account, order_group=group,
                          needs=(Need('workers'),))


def fixture(accounts=1, per_account=4, requests=None, now=100., interval=10.):
    # Numbers are engineering fixtures, NOT production recommendations.
    resources = [Resource('workers', 100, 0, 0)]
    offers = []
    for i in range(accounts):
        identity = f'account-{i}'
        resources += [Resource(f'{identity}:turn', 1, 0, 0, interval),
                      Resource(f'{identity}:image', per_account, 0, 0)]
        offers += [Offer(identity, 'chat', 'fixture-model', 'image',
                         (Need(f'{identity}:turn'), Need(f'{identity}:image'))),
                   Offer(identity, 'chat', 'fixture-model', 'text', (Need(f'{identity}:turn'),))]
    return Snapshot('revision-0', now, now + 10000, tuple(resources), tuple(offers), tuple(requests or [request()]))


def resource_update(snapshot, key, **changes):
    return replace(snapshot, resources=tuple(replace(r, **changes) if r.key == key else r for r in snapshot.resources),
                   revision=snapshot.revision + '+change')


def apply_in_fixture(snapshot, dispatch, *, finish_turn=False):
    """Test-only store projection; NOT the production atomic adapter.

    The actual Provider adapter must reserve original receipts and occupancy in
    one transaction. Clock updates here simulate the sending edge, not a
    pre-write into the existing account pacer. The tests below make no claim of
    that production integration.
    """
    if snapshot.revision != dispatch.expected_revision:
        raise ValueError('stale reservation')
    requirements = {n.resource: n.units for n in dispatch.needs}
    deadlines = dict(dispatch.pacing_preview)
    resources = []
    for resource in snapshot.resources:
        updated = resource
        if resource.key in requirements:
            updated = replace(updated, occupied=resource.occupied + requirements[resource.key],
                              next_at=deadlines.get(resource.key, resource.next_at))
        if finish_turn and (resource.key.endswith(':turn') or resource.key == 'workers'):
            updated = replace(updated, occupied=resource.occupied)
        resources.append(updated)
    rows = tuple(replace(r, state='active', bound_account=dispatch.account) if r.ref == dispatch.ref else r for r in snapshot.requests)
    return replace(snapshot, resources=tuple(resources), requests=rows, revision=snapshot.revision + '+reserved',
                   last_source=dispatch.source, last_account=dispatch.account)


class PlannerTests(unittest.TestCase):
    def test_ready_item_is_selected(self):
        self.assertEqual(choose_next(fixture(),100).dispatch.account, 'account-0')

    def test_no_fixed_global_four_cap(self):
        jobs = [request(i) for i in range(30)]
        snap = fixture(4, requests=jobs, interval=1)
        now = 100
        selected = []
        for _ in range(16):
            d = choose_next(snap, now).dispatch
            self.assertIsNotNone(d)
            selected.append(d)
            snap = apply_in_fixture(snap, d, finish_turn=True)
            now += 1
        self.assertIsNone(choose_next(snap, now).dispatch)
        new = fixture(5, requests=jobs)
        additional_resources = tuple(r for r in new.resources if r.key.startswith('account-4:'))
        additional_offers = tuple(o for o in new.offers if o.account == 'account-4')
        snap = replace(snap, resources=snap.resources+additional_resources, offers=snap.offers+additional_offers,
                       revision='fifth-account-ready')
        for _ in range(4):
            d = choose_next(snap, now).dispatch
            self.assertEqual(d.account, 'account-4')
            selected.append(d)
            snap = apply_in_fixture(snap, d, finish_turn=True)
            now += 10
        self.assertEqual(len({d.ref for d in selected}),20)
        self.assertIsNone(choose_next(snap,now).dispatch)

    def test_duplicate_account_offer_does_not_add_capacity(self):
        snap = fixture()
        snap = replace(snap,offers=snap.offers*2,resources=snap.resources*2)
        snap = resource_update(snap,'account-0:image',occupied=4)
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_conflicting_duplicate_is_rejected(self):
        snap = fixture()
        with self.assertRaises(InvalidSnapshot):
            choose_next(replace(snap,resources=snap.resources+(Resource('workers',5,0,0),)),100)

    def test_disabled_offer_adds_no_usable_capacity(self):
        snap = fixture()
        snap = replace(snap,offers=tuple(replace(o,enabled=False) for o in snap.offers))
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_restore_account_wakes_same_request(self):
        snap = fixture()
        disabled = replace(snap,offers=tuple(replace(o,enabled=False) for o in snap.offers))
        self.assertIsNone(choose_next(disabled,100).dispatch)
        self.assertEqual(choose_next(snap,100).dispatch.ref, snap.requests[0].ref)

    def test_unknown_usage_cannot_be_invented_as_zero(self):
        snap = resource_update(fixture(),'account-0:image',occupied=None)
        out = choose_next(snap,100)
        self.assertIsNone(out.dispatch)
        self.assertIn('occupancy_unknown',out.deferred[0].reasons)

    def test_shrink_under_active_occupancy_does_not_dispatch(self):
        snap = resource_update(fixture(),'account-0:image',capacity=1,occupied=3)
        self.assertIsNone(choose_next(snap,100).dispatch)
        self.assertEqual(next(r.occupied for r in snap.resources if r.key.endswith(':image')),3)

    def test_known_full_account_does_not_block_another_account(self):
        snap = resource_update(fixture(2),'account-0:image',occupied=4)
        self.assertEqual(choose_next(snap,100).dispatch.account,'account-1')

    def test_binding_cannot_move_to_free_account(self):
        snap = resource_update(fixture(2,requests=[request(account='account-0')]),'account-0:image',occupied=4)
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_future_start_gates_even_with_free_image_slots(self):
        snap = resource_update(fixture(),'account-0:turn',next_at=130)
        result = choose_next(snap,100)
        self.assertIsNone(result.dispatch)
        self.assertEqual(result.next_wake_at,130)
        self.assertIsNotNone(choose_next(snap,130).dispatch)

    def test_cooldown_does_not_take_worker(self):
        snap = resource_update(fixture(),'account-0:turn',next_at=300)
        result = choose_next(snap,100)
        self.assertIsNone(result.dispatch)
        self.assertEqual(snap.resources[0].occupied,0)

    def test_wait_is_max_of_constraints_min_of_accounts(self):
        snap = fixture(2)
        snap = resource_update(snap,'workers',next_at=120)
        snap = resource_update(snap,'account-0:turn',next_at=140)
        snap = resource_update(snap,'account-1:turn',next_at=130)
        self.assertEqual(choose_next(snap,100).next_wake_at,130)

    def test_unknown_ready_time_does_not_loop(self):
        snap = resource_update(fixture(),'account-0:turn',next_at=None)
        out = choose_next(snap,100)
        self.assertIsNone(out.dispatch)
        self.assertIsNone(out.next_wake_at)

    def test_fair_sources_round_robin(self):
        snap = fixture(3, requests=[request(i,s) for i in range(4) for s in ['happy','wb','ozon']],interval=0)
        sources=[]
        for _ in range(9):
            out=choose_next(snap,100)
            sources.append(out.dispatch.source)
            snap=apply_in_fixture(snap,out.dispatch,finish_turn=True)
        self.assertEqual(sources,['happy','ozon','wb']*3)

    def test_missing_old_cursor_does_not_reset_to_first(self):
        snap=replace(fixture(2,requests=[request(1,'happy'),request(2,'wb')]),last_source='ozon')
        self.assertEqual(choose_next(snap,100).dispatch.source,'wb')

    def test_busy_source_does_not_block_others(self):
        snap=fixture(2,requests=[request(0,'happy',account='account-0'),request(1,'wb',account='account-1')])
        snap=resource_update(snap,'account-0:turn',next_at=200)
        self.assertEqual(choose_next(snap,100).dispatch.source,'wb')

    def test_turn_resource_shared_between_text_and_images(self):
        snap=fixture(requests=[request(0,operation='text'),request(1,operation='image')],interval=0)
        dispatch=choose_next(snap,100).dispatch
        snap=apply_in_fixture(snap,dispatch)
        self.assertIsNone(choose_next(snap,100).dispatch)
        self.assertEqual(next(r.occupied for r in snap.resources if r.key.endswith(':image')),0)

    def test_image_slots_are_not_text_slots(self):
        snap=fixture(requests=[request(operation='text')])
        snap=resource_update(snap,'account-0:image',occupied=4)
        self.assertIsNotNone(choose_next(snap,100).dispatch)

    def test_saved_input_is_required_before_admission(self):
        snap=fixture(requests=[replace(request(),payload_saved=False)])
        self.assertIn('input_not_durable',choose_next(snap,100).deferred[0].reasons)

    def test_sent_unknown_and_terminal_are_never_replayed(self):
        snap=fixture(requests=[request(i,state=s) for i,s in enumerate(['active','unknown','failed','succeeded','cancelled'])])
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_unknown_previous_turn_preserves_group_order(self):
        snap=fixture(requests=[request(0,group='session-1',state='unknown'),request(1,group='session-1')])
        self.assertIn('earlier_group_request_unfinished',choose_next(snap,100).deferred[0].reasons)

    def test_other_group_can_progress(self):
        snap=fixture(requests=[request(0,group='one',state='active'),request(1,group='one'),request(2,group='two')])
        self.assertEqual(choose_next(snap,100).dispatch.ref.request_id,'request-2')

    def test_terminal_previous_turn_does_not_permanently_hold_session(self):
        snap=fixture(requests=[request(0,group='one',state='succeeded'),request(1,group='one')])
        self.assertEqual(choose_next(snap,100).dispatch.ref.request_id,'request-1')

    def test_other_owner_same_group_name_is_independent(self):
        snap=fixture(requests=[request(0,'a',group='one',state='unknown'),request(1,'b',group='one')])
        self.assertEqual(choose_next(snap,100).dispatch.source,'b')

    def test_chat_only_offer_does_not_substitute_codex(self):
        snap=fixture(requests=[request(route='codex')])
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_no_model_alias_guess(self):
        snap=fixture(requests=[request(model='new-unverified-model')])
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_global_server_limit_is_not_ignored(self):
        snap=resource_update(fixture(20),'workers',occupied=100)
        self.assertIsNone(choose_next(snap,100).dispatch)

    def test_future_and_expired_snapshot_are_not_dispatchable(self):
        for now in [99,10100,10101]:
            self.assertEqual(choose_next(fixture(),now).snapshot_problem,'snapshot_stale_or_future')

    def test_reservation_carries_expected_revision_and_clock(self):
        d=choose_next(fixture(),100).dispatch
        self.assertEqual(d.expected_revision,'revision-0')
        self.assertIn(('account-0:turn',110),d.pacing_preview)

    def test_stale_test_store_commit_rejected(self):
        snap=fixture(requests=[request(0),request(1)])
        first=choose_next(snap,100).dispatch
        second=choose_next(snap,100).dispatch
        after=apply_in_fixture(snap,first)
        with self.assertRaises(ValueError): apply_in_fixture(after,second)

    def test_no_input_mutation(self):
        snap=fixture()
        before=repr(snap)
        choose_next(snap,100)
        self.assertEqual(repr(snap),before)

    def test_duplicate_request_is_one_candidate(self):
        snap=fixture(requests=[request(),request()])
        d=choose_next(snap,100).dispatch
        self.assertEqual(d.ref,request().ref)

    def test_conflicting_request_identity_rejected(self):
        snap=fixture(requests=[request(),replace(request(),model='different')])
        with self.assertRaises(InvalidSnapshot): choose_next(snap,100)

    def test_invalid_numbers_rejected(self):
        for bad in [True,-1,float('inf'),float('nan')]:
            with self.assertRaises(InvalidSnapshot): Resource('slot',bad,0,0)

    def test_oversized_values_fail_closed(self):
        with self.assertRaises(InvalidSnapshot): Resource('slot',10**400,0,0)
        with self.assertRaises(InvalidSnapshot): Resource('slot',1,0,10**400)

    def test_absent_or_duplicate_resource_demand_rejected(self):
        snap=fixture()
        bad=replace(snap.offers[0],needs=(Need('missing'),))
        with self.assertRaises(InvalidSnapshot): choose_next(replace(snap,offers=(bad,)),100)
        with self.assertRaises(InvalidSnapshot): Offer('a','chat','m','image',(Need('x'),Need('x')))

    def test_new_account_does_not_release_unknown_occupancy(self):
        snap=fixture(2,requests=[request(0,state='unknown',account='account-0'),request(1)])
        snap=resource_update(snap,'account-0:image',occupied=4)
        out=choose_next(snap,100)
        self.assertEqual(out.dispatch.account,'account-1')
        self.assertEqual(snap.requests[0].state,'unknown')


if __name__=='__main__': unittest.main(verbosity=2)
