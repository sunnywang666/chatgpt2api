"""A backup must actually restore executable inputs, not only receipt IDs."""
import io
import json
import os
import sqlite3
import tarfile

from services import backup_service
from services.task_store import TaskStore


def test_original_task_backup_restores_private_input_and_committed_wire_prefix(tmp_path, monkeypatch):
    source = tmp_path / "source"
    store = TaskStore(source / "text_tasks.sqlite3")
    body = {"prompt": "original", "images": [("image.png", b"original-image", "image/png")]}
    ref = store.save_input(body)
    output = store.create_output()
    with store.output_file(output, append=True) as handle:
        handle.write(b"committed-uncommitted")
    receipt = {"id": "original", "owner_id": "owner", "status": "queued",
               "_input_ref": ref, "_wire_output": output, "_wire_size": 9}
    with store.transaction() as db:
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", ("owner", "original", "fixture", json.dumps(receipt)))
    monkeypatch.setattr(backup_service, "DATA_DIR", source)
    monkeypatch.setattr(backup_service, "IMAGE_INDEX_FILE", source / "missing-index.json")
    archive = backup_service.BackupService()._build_backup_archive({"include": {"image_tasks": True}}, trigger="test")
    restored = tmp_path / "restored"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for item in bundle.getmembers():
            if not item.name.startswith("data/"):
                continue
            assert item.mode == 0o600
            path = restored / item.name.removeprefix("data/")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(bundle.extractfile(item).read())
            os.chmod(path, item.mode)
    recovered = TaskStore(restored / "text_tasks.sqlite3")
    assert recovered.load_input(ref) == body
    with recovered.output_file(output) as handle:
        assert handle.read() == b"committed"
    with sqlite3.connect(recovered.path) as db:
        assert recovered.read_receipt(db, "text", "owner", "original")["_input_ref"] == ref
