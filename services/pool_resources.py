"""Management-only observations from the existing pool and original task ledger."""
from datetime import datetime, timezone

from services.config import config
from services.owned_accounts import chat_projection, observed_capacity


def resource_snapshot(accounts, image_tasks, codex) -> dict:
    rows = [row for row in accounts.list_accounts()
            if chat_projection(row)["authorization_status"] == "saved"]
    settings = config.resource_settings()
    durable = image_tasks.resource_occupancy()
    held = durable["by_account"]
    known = []
    observed = 0
    eligible = total_slots = free_min = free_max = process_inflight = 0
    uncertain = bool(durable["unattributed"])
    identities = set()
    for account in rows:
        capacity = observed_capacity(account)
        if capacity["remaining"] is not None:
            known.append(capacity["remaining"])
        try:
            at = datetime.fromisoformat(str(capacity["observed_at"]).replace("Z", "+00:00"))
            age = (datetime.now(timezone.utc) - at.astimezone(timezone.utc)).total_seconds()
            if capacity["state"] == "observed" and 0 <= age <= 300:
                observed += 1
        except (ValueError, TypeError):
            pass
        identity = str(account.get("provider_account_identity") or "")
        if identity:
            identities.add(identity)
        live_count = max(0, int(account.get("image_inflight") or 0))
        durable_count = held.get(identity, 0) if identity else 0
        process_inflight += live_count
        if (account.get("managed_disabled") or not accounts._is_image_account_available(account)):
            continue
        eligible += 1
        slots = min(settings["image_account_concurrency"], max(0, int(account.get("quota") or 0)))
        total_slots += slots
        # The legacy synchronous counter does not carry task IDs. It can
        # overlap a durable receipt. Report bounds, never sum as exact usage.
        low = max(0, slots - live_count - durable_count)
        high = max(0, slots - max(live_count, durable_count))
        free_min += low
        free_max += high
        uncertain = uncertain or low != high
    unattributed = durable["unattributed"] + sum(count for identity, count in held.items() if identity not in identities)
    if unattributed:
        uncertain = True
        free_min = 0
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "image": {"remaining": sum(known) if rows and observed == len(rows) else None,
                  "known_remaining": sum(known) if known else None,
                  "observed_accounts": observed, "total_accounts": len(rows),
                  "eligible_accounts": eligible, "slots_total": total_slots,
                  "slots_free": None if uncertain else free_min,
                  "slots_free_min": free_min, "slots_free_max": free_max,
                  "process_inflight": process_inflight,
                  "durable_inflight": sum(held.values()) + durable["unattributed"],
                  "occupancy_uncertain": uncertain},
        "codex": codex.resource_snapshot(),
        "queue": {"mode": "immediate_or_existing_wait", "queued": None},
    }
