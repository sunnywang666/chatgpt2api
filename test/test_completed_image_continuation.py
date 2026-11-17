"""Existing images resume the original review through normal claim/send/close."""
from copy import deepcopy
import json

import pytest

from services.product_work import advanced_continuation_chain
from services.image_task_service import _request_hash
from services.image_quota_successor import prepared_payload
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
