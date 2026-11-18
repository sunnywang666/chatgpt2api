"""Existing images resume the original review through normal claim/send/close."""
from copy import deepcopy
from datetime import datetime, timezone
import json

import pytest

from services.product_work import advanced_continuation_chain
from services.image_task_service import ImageTaskService
from services.image_task_service import _request_hash
from services.image_quota_successor import prepared_payload
from services.request_context import AdmissionLost
from services.pool_admission import ExecutionContext
from test.test_no_final_successor import product_no_final, product_lifecycle
from test.test_product_compact_successor import (advanced_to_own_successor,
    add_completed_image_tail, add_completed_tail)
from test.test_explicit_unknown_successor import row


def image_row(h, rid):
    with h.service.store.connect() as db:
        return h.service.store.read_receipt(db, 'image', 'owner', rid)


def image_chain(h, monkeypatch, *, quota):
    original, successor = advanced_to_own_successor(h, monkeypatch, workbench=True)
    original, _ = add_completed_image_tail(h, successor, monkeypatch)
    with h.service.store.transaction() as db:
        image = h.service.store.read_receipt(db, 'image', 'owner', 'image-tail')
        image['_sequence'] = h.service.store.next_sequence(db)
        parent = h.service.store.runtime(db, image['_product_work_key'])
        saved = h.service.store.load_input(image['_input_ref'])
        if quota:
            root = deepcopy(image)
            root.update(status='error', upstream_outcome='failed', error_code='NO_IMAGE_GENERATED',
                _authoritative_failure_reason='image_generation_quota_exhausted', data=[], result_file_ids=[],
                _completion={'state':'result_ready', 'quota_failover':True, 'max_extra_requests':1,
                    'replacement_id':'quota-child', 'selected_id':'quota-child'})
            payload = prepared_payload(saved['payload'], root, 'quota-child')
            child = deepcopy(image)
            child.update(id='quota-child', _sequence=h.service.store.next_sequence(db),
                provider_binding_id='second-binding', provider_account_identity='second-account',
                client_conversation_id='second-client', conversation_id='second-chat',
                request_message_id='second-image-user', parent_message_id='second-image-final',
                _completion_of='image-tail', _image_quota_failover=payload['_image_quota_failover'],
                request_hash=_request_hash(saved['mode'], payload),
                _input_ref=h.service.store.save_input({**saved, 'payload':payload}))
            parent['members'].append({'kind':'image','owner':'owner','request_id':'quota-child',
                'step_ref':image['_product_work']['step_ref']})
            h.service.store.write_receipt(db,'image','owner','image-tail',root)
            h.service.store.write_receipt(db,'image','owner','quota-child',child)
        else:
            # Historical failed attempt explicitly never reached upstream. Its
            # later success has a new step ref, as on the original 0310 task.
            old = deepcopy(image)
            old_payload = deepcopy(saved['payload'])
            old_payload['_product_work']['step_ref'] = 'old-unsent-image'
            old.update(id='old-unsent-image',status='error',upstream_outcome='not_submitted',
                error_code='RESULT_UNRECOVERABLE',_submission_started=False,upstream_submission_started=False,
                _product_work=old_payload['_product_work'], _sequence=image['_sequence']-0.5,
                parent_message_id=old_payload['parent_message_id'], request_message_id='old-unsent-user',
                data=[],result_file_ids=[], request_hash=_request_hash(saved['mode'],old_payload),
                _input_ref=h.service.store.save_input({**saved,'payload':old_payload}),
                _completion={'state':'needs_attention','next_at':None,'bound_retry':True,
                    'same_request_retry':True,'allow_unconfirmed_retry':False,'max_extra_requests':0,'reason':'COMPLETION_ORIGINAL_NOT_RETRYABLE'})
            # Use durable integer sequence positions.
            old['_sequence']=image['_sequence']; image['_sequence']=h.service.store.next_sequence(db)
            parent['members'].append({'kind':'image','owner':'owner','request_id':'old-unsent-image',
                'step_ref':'old-unsent-image'})
            h.service.store.write_receipt(db,'image','owner','old-unsent-image',old)
            h.service.store.write_receipt(db,'image','owner','image-tail',image)
            child=image
        h.service.store.set_runtime(db,parent['key'],parent)
    if quota:
        accounts=json.loads((h.root/'accounts.json').read_text())
        accounts.append({**accounts[0],'account_id':'second','provider_account_identity':'second-account',
                         'conversation_binding_ids':['second-binding']})
        (h.root/'accounts.json').write_text(json.dumps(accounts))
    return original,child


def queued_image_correction_after_reviews(h, monkeypatch, *, retired=True):
    """Model 0310: retired image -> saved image -> reviews -> final image."""
    if retired:
        original, image = image_chain(h, monkeypatch, quota=False)
    else:
        original, image = image_chain(h, monkeypatch, quota=False)
    review = add_completed_tail(h, 'review', previous_image=image)
    review = add_completed_tail(h, 'review-2', previous_id=review['request_id'])
    if not retired:
        # Keep the completed image and reviews but remove the registered
        # retired member: a later correction may follow an ordinary saved
        # image just as safely as the narrow 0310 bridge.
        with h.service.store.transaction() as db:
            parent = h.service.store.runtime(db, image['_product_work_key'])
            parent['members'] = [member for member in parent['members']
                                 if member['request_id'] != 'old-unsent-image']
            h.service.store.set_runtime(db, parent['key'], parent)
    h.admission.settings = lambda: {'chat_account_concurrency': 6, 'image_account_concurrency': 4}
    h.admission.accounts.update_account('fixture', {'limits_progress': [
        {'feature_name': 'image_gen', 'remaining': 100}],
        'capacity_observed_at': datetime.now(timezone.utc).isoformat(),
    })
    images = ImageTaskService(h.root / 'correction-images.json', admission=h.admission,
                              store=h.service.store)
    saved = h.service.store.load_input(image['_input_ref'])
    binding = {key: review[key] for key in (
        'provider_binding_id', 'provider_account_identity', 'client_conversation_id',
        'conversation_id', 'parent_message_id')}
    images.submit_generation(
        {'id': 'owner', 'role': 'admin', '_trusted_source': original['_source']},
        client_task_id='final-image-correction', prompt=saved['payload']['prompt'],
        model='gpt-image-2', size=saved['payload'].get('size'), quality=saved['payload'].get('quality', 'auto'),
        **binding, product_work={**review['_product_work'], 'step_ref': 'final-image-correction'},
    )
    # The production regression had already reached `before_send`; the
    # fixture's scheduler retains the unresolved original and is deliberately
    # not the behavior this send-edge test changes.  Install the durable
    # running state claim_next would have written, then exercise final
    # admission below.
    claim = 'final-image-correction-claim'
    with h.service.store.transaction() as db:
        correction = h.service.store.read_receipt(db, 'image', 'owner', 'final-image-correction')
        correction.update(status='running', _claim_id=claim, _claim_until=h.clock() + 30,
                          _executing=True, _turn_reserved=True, upstream_unfinished=True)
        h.service.store.write_receipt(db, 'image', 'owner', 'final-image-correction', correction)
    context = ExecutionContext(h.admission, 'image', 'owner', 'final-image-correction', claim,
                               claimed_at=h.clock())
    return original, image, context


def test_saved_image_and_reviews_allow_one_final_image_correction(product_no_final, monkeypatch):
    h = product_no_final
    original, _image, context = queued_image_correction_after_reviews(h, monkeypatch)

    context.before_send_with_marker()

    correction = image_row(h, 'final-image-correction')
    assert correction['_submission_started'] is True
    assert correction['parent_message_id'] == row(h.service, 'review-2')['parent_message_id']
    assert row(h.service) == original


def test_completed_image_and_reviews_allow_final_image_correction(product_no_final, monkeypatch):
    h = product_no_final
    _original, _image, context = queued_image_correction_after_reviews(h, monkeypatch, retired=False)

    context.before_send_with_marker()

    assert image_row(h, 'final-image-correction')['_submission_started'] is True


@pytest.mark.parametrize('drift', ['stored_identity', 'stored_mode'])
def test_normal_completed_image_requires_frozen_identity_and_mode(product_no_final, monkeypatch, drift):
    h = product_no_final
    _original, image, context = queued_image_correction_after_reviews(h, monkeypatch, retired=False)
    with h.service.store.transaction() as db:
        saved = h.service.store.read_receipt(db, 'image', 'owner', image['id'])
        stored = h.service.store.load_input(saved['_input_ref'])
        if drift == 'stored_identity':
            stored['identity']['id'] = 'other-owner'
        else:
            stored['mode'] = 'edit'
            saved['request_hash'] = _request_hash('edit', stored['payload'])
        saved['_input_ref'] = h.service.store.save_input(stored)
        h.service.store.write_receipt(db, 'image', 'owner', image['id'], saved)

    with pytest.raises(AdmissionLost, match='conversation is occupied'):
        context.before_send_with_marker()


def test_final_image_correction_cannot_send_twice(product_no_final, monkeypatch):
    h = product_no_final
    _original, _image, context = queued_image_correction_after_reviews(h, monkeypatch)
    context.before_send_with_marker()

    with pytest.raises(AdmissionLost, match='already submitted'):
        context.before_send_with_marker()


@pytest.mark.parametrize('drift', ['changed_input', 'active_old', 'mismatched_cursor', 'image_hold', 'unrelated_unknown'])
def test_final_image_correction_keeps_unproved_image_history_blocked(product_no_final, monkeypatch, drift):
    h = product_no_final
    _original, image, context = queued_image_correction_after_reviews(h, monkeypatch)
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, 'image', 'owner', 'old-unsent-image')
        saved = h.service.store.read_receipt(db, 'image', 'owner', image['id'])
        if drift == 'changed_input':
            body = h.service.store.load_input(old['_input_ref'])
            body['payload']['prompt'] = 'changed frozen image input'
            old['_input_ref'] = h.service.store.save_input(body)
            old['request_hash'] = _request_hash(body['mode'], body['payload'])
        elif drift == 'active_old':
            old['_executing'] = True
        elif drift == 'mismatched_cursor':
            saved['parent_message_id'] = 'changed-image-result-cursor'
        elif drift == 'image_hold':
            saved['_send_hold'] = True
        else:
            correction = h.service.store.read_receipt(db, 'image', 'owner', 'final-image-correction')
            foreign = deepcopy(row(h.service))
            foreign.update(id='foreign-unknown', request_id='foreign-unknown', status='unknown',
                           _product_work_key='foreign-product',
                           _product_work={'product_ref': 'foreign', 'step_ref': 'foreign'},
                           _claim_id=None, _claim_until=None, _executing=False, _turn_reserved=False,
                           _recovery_suppressed=False, _sequence=correction['_sequence'] - 1)
            assert (foreign['conversation_id'] == correction['conversation_id']
                    and foreign['provider_account_identity'] == correction['provider_account_identity']
                    and foreign['_sequence'] < correction['_sequence'])
            h.service.store.write_receipt(db, 'text', 'owner', 'foreign-unknown', foreign)
            parent = h.service.store.runtime(db, old['_product_work_key'])
            parent['members'].append({'kind': 'text', 'owner': 'owner',
                                      'request_id': 'foreign-unknown', 'step_ref': 'foreign'})
            h.service.store.set_runtime(db, parent['key'], parent)
        h.service.store.write_receipt(db, 'image', 'owner', 'old-unsent-image', old)
        h.service.store.write_receipt(db, 'image', 'owner', image['id'], saved)

    with pytest.raises(AdmissionLost, match='conversation is occupied'):
        context.before_send_with_marker()
    assert image_row(h, 'final-image-correction')['_submission_started'] is False


@pytest.mark.parametrize('quota',[False,True])
def test_saved_image_continues_review_and_next_step_without_regenerating(product_no_final,monkeypatch,quota):
    h=product_no_final
    original,image=image_chain(h,monkeypatch,quota=quota)
    before=image_row(h,image['id'])
    review=add_completed_tail(h,'review',previous_image=image)
    assert review['status']=='succeeded' and review['_submission_started'] is True
    # No additional old-conversation GET is possible for a new-account review;
    # its saved terminal result must nevertheless let the next step proceed.
    next_step=add_completed_tail(h,'next-step',previous_id='review')
    assert next_step['status']=='succeeded'
    product,lifecycle,calls,identity=product_lifecycle(h)
    identity['_trusted_source']=original['_source']
    def archive_image(owner,rid,archived):
        calls.append((owner,rid,archived))
        return {'request_id':rid,'conversation_id':image_row(h,rid)['conversation_id'],'archived':archived}
    lifecycle.images.archive_product_work=archive_image
    assert product.close(identity,'fixture-product',results_saved=True)['state']=='closing'
    assert lifecycle.process_one()
    if quota:
        assert lifecycle.process_one()
    assert product.get(identity,'fixture-product')['state']=='completed'
    assert ('owner','next-step',True) in calls
    if quota:
        assert ('owner','image-tail',True) in calls
    assert image_row(h,image['id'])==before
    assert row(h.service)==original
    assert len([x for x in h.sent if x.get('client_request_id')=='review'])==1


@pytest.mark.parametrize('drift',['submitted','upstream_submitted','unknown','active','paused','hash','foreign','result','unconfirmed','bound_retry'])
def test_unsent_image_exception_keeps_ambiguous_and_changed_attempts_blocked(product_no_final,monkeypatch,drift):
    h=product_no_final
    original,image=image_chain(h,monkeypatch,quota=False)
    def drift_before_claim():
      with h.service.store.transaction() as db:
        old=h.service.store.read_receipt(db,'image','owner','old-unsent-image')
        if drift=='submitted': old['_submission_started']=True
        elif drift=='upstream_submitted': old['upstream_submission_started']=True
        elif drift=='unconfirmed': old['_completion']['allow_unconfirmed_retry']=True
        elif drift=='bound_retry': old['_completion']['bound_retry']=False
        elif drift=='unknown': old['upstream_outcome']='unknown'
        elif drift=='active': old['_executing']=True
        elif drift=='paused': old['_recovery_paused']=True
        elif drift=='hash': old['request_hash']='changed'
        elif drift=='foreign': old['_product_work_key']='other-product'
        elif drift=='result': old['result_file_ids']=['unexpected-result']
        h.service.store.write_receipt(db,'image','owner','old-unsent-image',old)
    review=add_completed_tail(h,'review',previous_image=image,expect_claim=False,before_claim=drift_before_claim)
    assert review['_submission_started'] is False


@pytest.mark.parametrize('drift',['selection','hash','active','paused','account','result','fork','foreign'])
@pytest.mark.parametrize('edge',['claim','send'])
def test_quota_bridge_revalidates_saved_result_at_claim_and_send(product_no_final,monkeypatch,drift,edge):
    h=product_no_final
    original,image=image_chain(h,monkeypatch,quota=True)
    def change():
        with h.service.store.transaction() as db:
            root=h.service.store.read_receipt(db,'image','owner','image-tail')
            child=h.service.store.read_receipt(db,'image','owner','quota-child')
            if drift=='selection': root['_completion']['selected_id']='other'
            elif drift=='hash': child['request_hash']='changed'
            elif drift=='active': child['_executing']=True
            elif drift=='paused': root['_recovery_paused']=True
            elif drift=='account': child['provider_account_identity']=root['provider_account_identity']
            elif drift=='result': child['result_file_ids']=[]
            elif drift=='fork':
                parent=h.service.store.runtime(db,root['_product_work_key'])
                parent['members'].append({'kind':'image','owner':'owner','request_id':'fork','step_ref':'image-tail'})
                h.service.store.set_runtime(db,parent['key'],parent)
                h.service.store.write_receipt(db,'image','owner','fork',{**child,'id':'fork'})
            elif drift=='foreign':
                h.service.store.write_receipt(db,'image','owner','foreign',{
                    **child,'id':'foreign','_completion_of':None,'_product_work_key':'other-product'})
            h.service.store.write_receipt(db,'image','owner','image-tail',root)
            h.service.store.write_receipt(db,'image','owner','quota-child',child)
    kwargs={'before_claim':change,'expect_claim':False} if edge=='claim' else {'before_send':change,'expect_sent':False}
    review=add_completed_tail(h,'review',previous_image=image,**kwargs)
    assert review['_submission_started'] is False
    assert not any(x.get('client_request_id')=='review' for x in h.sent)
    assert row(h.service)==original


@pytest.mark.parametrize('ambiguous',[False,True])
def test_multiple_retired_unsent_attempts_are_history_but_unknown_is_not(product_no_final,monkeypatch,ambiguous):
    h=product_no_final
    original,image=image_chain(h,monkeypatch,quota=False)
    with h.service.store.transaction() as db:
        old=h.service.store.read_receipt(db,'image','owner','old-unsent-image')
        clone=deepcopy(old);saved=h.service.store.load_input(old['_input_ref'])
        saved['payload']['_product_work']['step_ref']='another-unsent'
        clone.update(id='another-unsent',_product_work=saved['payload']['_product_work'],
                     _input_ref=h.service.store.save_input(saved),request_hash=_request_hash(saved['mode'],saved['payload']))
        if ambiguous:
            clone.update(upstream_outcome='unknown',upstream_submission_started=True)
        parent=h.service.store.runtime(db,old['_product_work_key'])
        parent['members'].append({'kind':'image','owner':'owner','request_id':'another-unsent','step_ref':'another-unsent'})
        h.service.store.set_runtime(db,parent['key'],parent)
        h.service.store.write_receipt(db,'image','owner','another-unsent',clone)
    review=add_completed_tail(h,'review',previous_image=image,expect_claim=not ambiguous)
    assert review['_submission_started'] is (not ambiguous)


def correction_with_unsent_suffix(h, monkeypatch):
    original, _, context = queued_image_correction_after_reviews(h, monkeypatch)
    with h.service.store.transaction() as db:
        current = h.service.store.read_receipt(db, 'image', 'owner', context.request_id)
        parent = h.service.store.runtime(db, current['_product_work_key'])
        old_rows = []
        for i in range(2):
            old = deepcopy(current)
            stored = h.service.store.load_input(current['_input_ref'])
            name = f'unsent-correction-{i}'
            stored['payload']['_product_work']['step_ref'] = name
            old.update(id=name, status='error', error_code='RESULT_UNRECOVERABLE',
                upstream_outcome='not_submitted', upstream_unfinished=False,
                _submission_started=False, upstream_submission_started=False,
                _executing=False, _turn_reserved=True, _claim_id=f'expired-{i}',
                _claim_until=h.clock()-1, _completion=None,
                _sequence=current['_sequence'] if i == 0 else h.service.store.next_sequence(db),
                _product_work=stored['payload']['_product_work'], request_message_id=f'unsent-user-{i}',
                _input_ref=h.service.store.save_input(stored),
                request_hash=_request_hash(stored['mode'], stored['payload']))
            parent['members'].append({'kind':'image','owner':'owner','request_id':name,'step_ref':name})
            h.service.store.write_receipt(db, 'image', 'owner', name, old)
            old_rows.append(old)
        current['_sequence'] = h.service.store.next_sequence(db)
        current['upstream_submission_started'] = False
        current['_completion'] = {'state':'checking_original','bound_retry':True,
            'same_request_retry':True,'max_extra_requests':0,'allow_unconfirmed_retry':False}
        h.service.store.write_receipt(db, 'image', 'owner', current['id'], current)
        h.service.store.set_runtime(db, parent['key'], parent)
    return original, old_rows, context


def test_bound_correction_passes_identical_unsent_suffix_without_rewriting_history(product_no_final, monkeypatch):
    h = product_no_final
    original, old_rows, context = correction_with_unsent_suffix(h, monkeypatch)
    context.before_send_with_marker()
    assert image_row(h, context.request_id)['_submission_started'] is True
    assert row(h.service) == original
    assert [image_row(h, old['id']) for old in old_rows] == old_rows
    with pytest.raises(AdmissionLost):
        context.before_send_with_marker()
    stages = image_row(h, context.request_id).get('_execution_timeline', [])
    assert sum(event.get('stage') == 'send_call_started' for event in stages) == 1


@pytest.mark.parametrize('drift', [
    'unknown', 'sent', 'unfinished', 'live_claim', 'invalid_claim', 'missing_claim',
    'executing', 'recovery_claim', 'held', 'paused', 'result', 'pending_output',
    'completion', 'hash', 'input', 'cursor', 'foreign', 'duplicate_step',
    'not_reserved', 'invalid_claim_id', 'recovery_lease', 'recovery_lease_invalid',
])
def test_unsent_suffix_requires_exact_retired_input_and_preserves_guards(product_no_final, monkeypatch, drift):
    h = product_no_final
    original, old_rows, context = correction_with_unsent_suffix(h, monkeypatch)
    with h.service.store.transaction() as db:
        old = h.service.store.read_receipt(db, 'image', 'owner', old_rows[0]['id'])
        if drift == 'unknown': old['upstream_outcome'] = 'unknown'
        elif drift == 'sent': old['upstream_submission_started'] = True
        elif drift == 'unfinished': old['upstream_unfinished'] = True
        elif drift == 'live_claim': old['_claim_until'] = h.clock()+60
        elif drift == 'invalid_claim': old['_claim_until'] = 'expired'
        elif drift == 'missing_claim': old['_claim_id'] = None
        elif drift == 'not_reserved': old['_turn_reserved'] = False
        elif drift == 'invalid_claim_id': old['_claim_id'] = 42
        elif drift == 'recovery_lease': old['recovery_lease_until'] = h.clock()+60
        elif drift == 'recovery_lease_invalid': old['_recovery_claim_until'] = 'unknown'
        elif drift == 'executing': old['_executing'] = True
        elif drift == 'recovery_claim': old['recovery_claim_id'] = 'active-recovery'
        elif drift == 'held': old['_send_hold'] = True
        elif drift == 'paused': old['_recovery_paused'] = True
        elif drift == 'result': old['result_file_ids'] = ['saved-result']
        elif drift == 'pending_output': old['_pending_image_output'] = {'output_ref':'saved-output'}
        elif drift == 'completion': old['_completion'] = {'state':'checking_original'}
        elif drift == 'hash': old['request_hash'] = 'changed'
        elif drift == 'input':
            stored = h.service.store.load_input(old['_input_ref'])
            stored['payload']['prompt'] += ' changed'
            old.update(_input_ref=h.service.store.save_input(stored),
                       request_hash=_request_hash(stored['mode'], stored['payload']))
        elif drift == 'cursor': old['parent_message_id'] = 'different-parent'
        elif drift == 'foreign': old['_product_work_key'] = 'other-product'
        elif drift == 'duplicate_step':
            parent = h.service.store.runtime(db, old['_product_work_key'])
            member = next(m for m in parent['members'] if m['request_id'] == old['id'])
            parent['members'].append(deepcopy(member))
            h.service.store.set_runtime(db, parent['key'], parent)
        h.service.store.write_receipt(db, 'image', 'owner', old['id'], old)
    with pytest.raises(AdmissionLost):
        context.before_send_with_marker()
    assert image_row(h, context.request_id)['_submission_started'] is False
    assert row(h.service) == original


@pytest.mark.parametrize('drift', ['missing', 'state', 'bound', 'same_request', 'unconfirmed',
    'extra', 'selected', 'replacement', 'completion_of', 'result', 'pending'])
def test_unsent_suffix_only_allows_current_bound_same_request_retry(product_no_final, monkeypatch, drift):
    h = product_no_final
    original, _, context = correction_with_unsent_suffix(h, monkeypatch)
    with h.service.store.transaction() as db:
        current = h.service.store.read_receipt(db, 'image', 'owner', context.request_id)
        marker = current['_completion']
        if drift == 'missing': current.pop('_completion')
        elif drift == 'state': marker['state'] = 'result_ready'
        elif drift == 'bound': marker['bound_retry'] = False
        elif drift == 'same_request': marker['same_request_retry'] = False
        elif drift == 'unconfirmed': marker['allow_unconfirmed_retry'] = True
        elif drift == 'extra': marker['max_extra_requests'] = 1
        elif drift == 'selected': marker['selected_id'] = 'another-task'
        elif drift == 'replacement': marker['replacement_id'] = 'another-task'
        elif drift == 'completion_of': current['_completion_of'] = 'another-task'
        elif drift == 'result': current['result_file_ids'] = ['saved-result']
        elif drift == 'pending': current['_pending_image_output'] = {'output_ref':'pending'}
        h.service.store.write_receipt(db, 'image', 'owner', current['id'], current)
    with pytest.raises(AdmissionLost):
        context.before_send_with_marker()
    assert image_row(h, context.request_id)['_submission_started'] is False
    assert row(h.service) == original
