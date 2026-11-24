"""Verify the settlement mechanism against live recordings.

For every resolved round covered by the recorded RTDS ``crypto_prices_twap_sixty`` stream,
compare the official Gamma ``eventMetadata.priceToBeat`` / ``finalPrice`` with the TWAP-60
stream value at the round start / end second. Also reports the spot-stream value at the
end second for contrast.

Usage: python scripts/verify_settlement.py
"""
from __future__ import annotations

import duckdb
import pandas as pd

from btc5m.history.gamma_rounds import fetch_rounds
from btc5m.http import client
from btc5m.storage.store import live_glob


def main() -> None:
    con = duckdb.connect()
    cl = con.execute(f"select src_ts, kind, value from read_parquet('{live_glob('chainlink')}', union_by_name=true)").df()
    tw = cl[cl.kind == "chainlink_twap60"].drop_duplicates("src_ts").set_index("src_ts")["value"]
    sp = cl[cl.kind == "chainlink_spot"].drop_duplicates("src_ts").set_index("src_ts")["value"]
    lo, hi = int(tw.index.min()), int(tw.index.max())
    rows = [r for r in fetch_rounds(list(range((lo // 300 + 1) * 300, hi - 300, 300)), client()) if r["resolved"]]
    out = []
    for r in rows:
        s, e = float(r["start"]), float(r["end"])
        out.append({
            "start": r["start"],
            "ptb_minus_twap_at_start": tw.get(s) - r["price_to_beat"] if s in tw.index and r["price_to_beat"] else None,
            "final_minus_twap_at_end": tw.get(e) - r["final_price"] if e in tw.index and r["final_price"] else None,
            "final_minus_spot_at_end": sp.get(e) - r["final_price"] if e in sp.index and r["final_price"] else None,
        })
    df = pd.DataFrame(out)
    print(df.to_string())
    print(df.drop(columns="start").abs().describe().loc[["count", "mean", "max"]].to_string())


if __name__ == "__main__":
    main()
