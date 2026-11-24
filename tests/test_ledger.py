import json
import time

import pytest

from btc5m.live.ledger import Ledger, LedgerError


def _pred(start, k=30):
    return {"round_start": start, "checkpoint": k, "model_version": "test", "p_up": 0.6, "p_down": 0.4}


def test_chain_verifies_and_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    start = int(time.time()) // 300 * 300
    r1 = led.add_prediction(_pred(start, 0))
    led.add_prediction(_pred(start, 15))
    ok, n, _ = led.verify()
    assert ok and n == 2
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["p_up"] = 0.99  # retroactive edit
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    (tmp_path / "l.jsonl").write_text("\n".join(lines) + "\n")
    ok, _, msg = Ledger(tmp_path / "l.jsonl").verify()
    assert not ok and "modified" in msg
    assert r1["prev_hash"] == "0" * 64


def test_deleting_a_line_breaks_chain(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    start = int(time.time()) // 300 * 300
    for k in (0, 15, 30):
        led.add_prediction(_pred(start, k))
    lines = (tmp_path / "l.jsonl").read_text().splitlines()
    (tmp_path / "l.jsonl").write_text("\n".join([lines[0], lines[2]]) + "\n")
    ok, _, msg = Ledger(tmp_path / "l.jsonl").verify()
    assert not ok


def test_duplicate_prediction_ignored(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    start = int(time.time()) // 300 * 300
    assert led.add_prediction(_pred(start)) is not None
    assert led.add_prediction(_pred(start)) is None


def test_prediction_after_round_end_refused(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    old = int(time.time()) // 300 * 300 - 600
    with pytest.raises(LedgerError):
        led.add_prediction(_pred(old))


def test_settlement_rules(tmp_path):
    led = Ledger(tmp_path / "l.jsonl")
    start = int(time.time()) // 300 * 300
    p = led.add_prediction(_pred(start))
    with pytest.raises(LedgerError):  # round not over
        led.add_settlement(p["hash"], {"outcome_up": 1})
    with pytest.raises(LedgerError):
        led.add_settlement("nope", {"outcome_up": 1})
    # simulate an old prediction written legitimately in the past by patching the clock
    led2 = Ledger(tmp_path / "l2.jsonl")
    real = time.time
    try:
        time.time = lambda: start - 600 + 10  # inside an earlier round
        q = led2.add_prediction(_pred(start - 600))
    finally:
        time.time = real
    s = led2.add_settlement(q["hash"], {"outcome_up": 1})
    assert s is not None and s["ref_hash"] == q["hash"]
    assert led2.add_settlement(q["hash"], {"outcome_up": 0}) is None  # only once
    assert Ledger(tmp_path / "l2.jsonl").verify()[0]


def test_single_writer_lock(tmp_path):
    import subprocess
    import sys

    from btc5m.live.ledger import acquire_writer_lock

    path = tmp_path / "ledger" / "predictions.jsonl"
    held = acquire_writer_lock(path)
    code = (
        "import sys; from btc5m.live.ledger import acquire_writer_lock, LedgerError\n"
        f"try:\n    acquire_writer_lock(r'{path}')\nexcept LedgerError:\n    sys.exit(7)\n"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 7
    held.close()
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
