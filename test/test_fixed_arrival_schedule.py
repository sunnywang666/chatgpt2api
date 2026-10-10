from pathlib import Path
import importlib.util


spec = importlib.util.spec_from_file_location(
    "fixed_arrival_schedule", Path(__file__).parents[1] / "scripts/acceptance/fixed_arrival_schedule.py")
schedule = importlib.util.module_from_spec(spec)
spec.loader.exec_module(schedule)


def test_slow_submit_keeps_fixed_arrival_and_marks_full_capacity_unsent():
    scheduler = schedule.FixedArrivalScheduler(count=4, interval=1, workers=1)
    scheduler.start(100)
    first = scheduler.due(100)[0]
    assert first.action == "dispatch" and first.scheduled_at == 100
    scheduler.dispatched(0, 100.01)
    assert scheduler.due(101)[0].action == "unsent_capacity"
    assert scheduler.due(102)[0].action == "unsent_capacity"
    scheduler.finished(0, 102.5)
    last = scheduler.due(103)[0]
    assert last.action == "dispatch"
    assert [row["action"] for row in scheduler.records()] == ["dispatch", "unsent_capacity", "unsent_capacity", "dispatch"]


def test_driver_stall_never_causes_a_catch_up_burst():
    scheduler = schedule.FixedArrivalScheduler(count=5, interval=1, workers=4)
    scheduler.start(10)
    decisions = scheduler.due(13.8)
    assert [decision.action for decision in decisions] == ["unsent_lag", "unsent_lag", "unsent_lag", "dispatch"]
    assert decisions[-1].index == 3
    assert scheduler.active_workers == 1
    assert scheduler.due(13.8) == []


def test_stop_accounts_for_every_unsent_slot_without_a_full_pass():
    scheduler = schedule.FixedArrivalScheduler(count=4, interval=1, workers=2)
    scheduler.start(0)
    scheduler.stop("raw_rejection", 0.2)
    summary = scheduler.summary()
    assert summary["counts"] == {"dispatch": 0, "unsent_capacity": 0, "unsent_lag": 0,
                                 "not_sent_after_stop": 4, "reserved_for_submission": 0,
                                 "actual_started": 0, "planned": 4, "accounted": 4}
    assert summary["dispatch_lag_seconds"] is None
    assert scheduler.complete


def test_segmented_summary_uses_only_real_timestamp_pairs():
    summary = schedule.segmented_timing([
        {"scheduled_arrival_monotonic": 0, "actual_dispatch_monotonic": 1,
         "admitted_monotonic": 3, "provider_success_observed_monotonic": 4,
         "saved_monotonic": 5, "archive_confirmed_monotonic": 8},
        {"scheduled_arrival_monotonic": 10, "actual_dispatch_monotonic": 13,
         "admitted_monotonic": 14},
        {"scheduled_arrival_monotonic": "unparsed", "actual_dispatch_monotonic": 20},
    ])
    assert summary["arrival"] == {"count": 2, "p50": 1.0, "p95": 3.0, "max": 3.0}
    assert summary["admission"] == {"count": 2, "p50": 1.0, "p95": 2.0, "max": 2.0}
    assert summary["dispatch_to_upstream_send"] is None
    assert summary["upstream_send_to_finished"] is None
    assert summary["upstream_send_to_success_observed"] is None
    assert summary["dispatch_to_result_save"] == {"count": 1, "p50": 4.0, "p95": 4.0, "max": 4.0}
    assert summary["dispatch_to_archive"] == {"count": 1, "p50": 7.0, "p95": 7.0, "max": 7.0}


def test_failed_executor_reservation_is_not_an_actual_start():
    scheduler = schedule.FixedArrivalScheduler(count=2, interval=1, workers=1)
    scheduler.start(0)
    assert scheduler.due(0)[0].action == "dispatch"
    scheduler.submission_failed(0, "executor_rejected", .1)
    scheduler.stop("executor_rejected", .1)
    assert scheduler.summary()["counts"] == {"dispatch": 0, "unsent_capacity": 0, "unsent_lag": 0,
                                              "not_sent_after_stop": 2, "reserved_for_submission": 1,
                                              "actual_started": 0, "planned": 2, "accounted": 2}


def test_rejects_invalid_monotonic_transitions_and_nonfinite_interval():
    import math
    for interval in (0, -1, math.inf, math.nan):
        try:
            schedule.FixedArrivalScheduler(count=1, interval=interval, workers=1)
        except ValueError:
            pass
        else:
            raise AssertionError(interval)
    scheduler = schedule.FixedArrivalScheduler(count=1, interval=1, workers=1)
    scheduler.start(10)
    scheduler.due(10)
    for invalid in (9, 9.9):
        try:
            scheduler.dispatched(0, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(invalid)
    scheduler.dispatched(0, 10)
    try:
        scheduler.finished(0, 9.9)
    except ValueError:
        pass
    else:
        raise AssertionError("finish predating dispatch was accepted")


def test_controlled_load_keeps_collecting_accepted_work_until_a_later_rejection_stops_arrivals():
    scheduler = schedule.FixedArrivalScheduler(count=4, interval=1, workers=1)
    scheduler.start(0)
    first = scheduler.due(0)[0]
    scheduler.dispatched(first.index, 0)
    # This represents the ordinary collection loop saving an accepted first
    # request while the independent arrival clock continues to advance.
    collected = [{"scheduled_arrival_monotonic": 0, "actual_dispatch_monotonic": 0,
                  "admitted_monotonic": .1, "provider_success_observed_monotonic": .5,
                  "saved_monotonic": .6, "archive_confirmed_monotonic": .8}]
    assert scheduler.due(1)[0].action == "unsent_capacity"
    scheduler.finished(first.index, 1.1)
    second = scheduler.due(2)[0]
    assert second.action == "dispatch"
    scheduler.dispatched(second.index, 2)
    # A raw rejection is known after this started original POST; future slots
    # are stopped, while the already accepted first result remains collected.
    scheduler.stop("raw_rejection", 2.2)
    scheduler.finished(second.index, 2.3)
    assert scheduler.complete
    assert [row["action"] for row in scheduler.records()] == ["dispatch", "unsent_capacity", "dispatch", "not_sent_after_stop"]
    assert schedule.segmented_timing(collected)["dispatch_to_archive"]["count"] == 1
