import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

from btc5m.storage import store


def test_compact_merges_old_files_and_keeps_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "data_dir", lambda: tmp_path)
    d = tmp_path / "raw" / "live" / "s1" / "date=2026-11-24"
    d.mkdir(parents=True)
    for i in range(5):
        f = d / f"part{i}.parquet"
        pq.write_table(pa.table({"rx": [float(i)], "v": [i]}), f)
        old = time.time() - 7200
        os.utime(f, (old, old))
    fresh = d / "fresh.parquet"
    pq.write_table(pa.table({"rx": [9.0], "v": [9]}), fresh)
    res = store.compact(min_age_s=3600)
    files = sorted(p.name for p in d.glob("*.parquet"))
    assert res["s1/date=2026-11-24"]["rows"] == 5
    assert "fresh.parquet" in files and len(files) == 2
    total = sum(pq.read_table(d / f).num_rows for f in files)
    assert total == 6
