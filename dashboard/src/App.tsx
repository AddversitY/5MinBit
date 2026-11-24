import { useEffect, useMemo, useState } from "react";
import Chart, { SeriesSpec } from "./Chart";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

const usd = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? "—" : "$" + v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number | null | undefined, d = 1) => (v == null || !isFinite(v) ? "—" : (v * 100).toFixed(d) + "%");
const num = (v: number | null | undefined, d = 3) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));
const cents = (v: number | null | undefined) => (v == null || !isFinite(v) ? "—" : (v * 100).toFixed(1) + "¢");

function useJson<T>(url: string, everyMs: number, init: T): T {
  const [v, setV] = useState<T>(init);
  useEffect(() => {
    let alive = true;
    const go = () =>
      fetch(url)
        .then((r) => r.json())
        .then((j) => alive && setV(j))
        .catch(() => {});
    go();
    const id = setInterval(go, everyMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [url, everyMs]);
  return v;
}

function useLiveState(): [Any, boolean] {
  const [s, setS] = useState<Any>({});
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let ws: WebSocket | null = null;
    let stop = false;
    let poll: number | undefined;
    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => setConnected(true);
      ws.onmessage = (e) => setS(JSON.parse(e.data));
      ws.onclose = () => {
        setConnected(false);
        if (!stop) setTimeout(connect, 2000);
      };
    };
    connect();
    poll = window.setInterval(() => {
      if (!ws || ws.readyState !== 1) fetch("/api/state").then((r) => r.json()).then(setS).catch(() => {});
    }, 2000);
    return () => {
      stop = true;
      ws?.close();
      clearInterval(poll);
    };
  }, []);
  return [s, connected];
}

function Countdown({ s }: { s: Any }) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now() / 1000), 250);
    return () => clearInterval(id);
  }, []);
  const end = s.round_end ?? Math.ceil(now / 300) * 300;
  const rem = Math.max(0, end - now);
  const mm = Math.floor(rem / 60);
  const ss = Math.floor(rem % 60);
  const start = end - 300;
  return (
    <div className="card">
      <h2>
        Round countdown <span className="tag verified">clock</span>
      </h2>
      <div className="hero num">
        {mm}:{ss.toString().padStart(2, "0")}
      </div>
      <div className="bar" aria-label="round progress">
        <div style={{ width: `${((300 - rem) / 300) * 100}%` }} />
      </div>
      <div className="row small">
        <span className="lbl">Elapsed</span>
        <span className="v">{Math.floor(300 - rem)} s</span>
      </div>
      <div className="row small">
        <span className="lbl">Round</span>
        <span className="v">btc-updown-5m-{start}</span>
      </div>
      <div className="row small">
        <span className="lbl">Settlement</span>
        <span className="v">{s.settlement_kind === "twap60" ? "Chainlink 60 s TWAP" : s.settlement_kind ?? "—"}</span>
      </div>
    </div>
  );
}

function Banner({ s, wsOk }: { s: Any; wsOk: boolean }) {
  const reasons: string[] = s.reasons ?? ["waiting for engine"];
  const insufficient = s.status !== "OK";
  return (
    <div className={`banner ${insufficient ? "warn" : "ok"}`} role="status">
      <div className="icon">{insufficient ? "!" : "✓"}</div>
      <div>
        <div className="title">
          {insufficient ? "INSUFFICIENT EVIDENCE" : "Within validated conditions"} · data {s.data_status ?? "—"} · stream {wsOk ? "live" : "reconnecting"}
        </div>
        {insufficient && (
          <ul>
            {reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

const PRICE_SERIES: SeriesSpec[] = [
  { label: "Chainlink spot", color: "--s1" },
  { label: "Chainlink TWAP-60", color: "--s2" },
  { label: "Kraken last", color: "--s3", width: 1.5 },
  { label: "Price to beat", color: "--ink-2", dash: [6, 4], width: 1.5 },
];
const PROB_SERIES: SeriesSpec[] = [
  { label: "Model P(Up)", color: "--s1" },
  { label: "Market mid P(Up)", color: "--s2" },
];

export default function App() {
  const [s, wsOk] = useLiveState();
  const series: Any[] = useJson("/api/series?seconds=900", 2000, []);
  const perf: Any = useJson("/api/performance", 30000, {});
  const ledger: Any[] = useJson("/api/ledger?limit=40", 15000, []);
  const health: Any = useJson("/api/health", 10000, {});
  const [theme, setTheme] = useState<string | null>(null);

  useEffect(() => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
  }, [theme]);

  const priceData = useMemo(() => {
    const x = series.map((p) => p.ts);
    return [x, series.map((p) => p.chainlink), series.map((p) => p.twap60), series.map((p) => p.kraken), series.map((p) => p.ptb)];
  }, [series]);
  const probData = useMemo(() => [series.map((p) => p.ts), series.map((p) => p.p_up), series.map((p) => p.pm_mid)], [series]);

  const m = s.market ?? {};
  const ev = s.expected_value_per_share_after_costs ?? {};
  const f = s.features ?? {};
  const wf = perf.walk_forward;
  const paper = perf.paper ?? {};
  const maxAbs = Math.max(1e-9, ...(s.explain ?? []).map((e: Any) => Math.abs(e.contribution_logodds ?? 0)));

  return (
    <div className="wrap">
      <header className="top">
        <div>
          <h1>Polymarket BTC 5-minute Up/Down · prediction research engine</h1>
          <div className="sub">Read-only research tool. No wallet, no orders. All trading figures are hypothetical paper results.</div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span className="badge paper">PAPER MODE</span>
          <span className="badge">
            model {s.model_version ?? "—"} {s.model_validated ? "· validated" : "· not validated"}
          </span>
          <button className="toggle" onClick={() => setTheme(theme === "dark" ? "light" : theme === "light" ? null : "dark")}>
            theme: {theme ?? "auto"}
          </button>
        </div>
      </header>

      <Banner s={s} wsOk={wsOk} />

      <div className="grid g4">
        <Countdown s={s} />
        <div className="card">
          <h2>
            Price to beat <span className="tag verified">verified</span>
          </h2>
          <div className="hero sm num">{usd(s.price_to_beat)}</div>
          <div className="lbl">Chainlink BTC/USD TWAP-60 at round start · {s.price_to_beat_status ?? "—"}</div>
          <div className="row small" style={{ marginTop: 8 }}>
            <span className="lbl">Chainlink spot − price to beat</span>
            <span className="v">{s.chainlink_spot != null && s.price_to_beat != null ? usd(s.chainlink_spot - s.price_to_beat) : "—"}</span>
          </div>
        </div>
        <div className="card">
          <h2>
            Chainlink reference <span className="tag verified">verified</span>
          </h2>
          <div className="hero sm num">{usd(s.chainlink_spot)}</div>
          <div className="row small">
            <span className="lbl">TWAP-60 (settlement stream)</span>
            <span className="v">{usd(s.chainlink_twap60)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Feed age</span>
            <span className="v">{num(s.chainlink_age_s, 1)} s</span>
          </div>
        </div>
        <div className="card">
          <h2>
            Bitcoin spot <span className="tag verified">verified</span>
          </h2>
          <div className="hero sm num">{usd(s.coinbase_last)}</div>
          <div className="row small">
            <span className="lbl">Coinbase</span>
            <span className="v">{usd(s.coinbase_last)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Kraken</span>
            <span className="v">{usd(s.kraken_last)}</span>
          </div>
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>
            Probability estimate <span className="tag model">model estimate</span>
          </h2>
          <div className="probs">
            <div>
              <div className="lbl">P(Up)</div>
              <div className="hero num">{pct(s.p_up)}</div>
            </div>
            <div>
              <div className="lbl">P(Down)</div>
              <div className="hero num">{pct(s.p_down)}</div>
            </div>
          </div>
          <div className="row small">
            <span className="lbl">Model disagreement band</span>
            <span className="v">{s.p_band ? `${pct(s.p_band[0])} – ${pct(s.p_band[1])}` : "—"}</span>
          </div>
          <div className="row small">
            <span className="lbl">Projected settlement (TWAP)</span>
            <span className="v">{usd(s.projected_settlement)}</span>
          </div>
          <div className="row small">
            <span className="lbl">90% interval</span>
            <span className="v">{s.projected_interval90 ? `${usd(s.projected_interval90[0], 0)} – ${usd(s.projected_interval90[1], 0)}` : "—"}</span>
          </div>
          <div className="row small">
            <span className="lbl">Vol-adjusted distance z</span>
            <span className="v">{num(s.z, 2)}</span>
          </div>
        </div>
        <div className="card">
          <h2>
            Polymarket contract prices <span className="tag verified">verified market data</span>
          </h2>
          <table>
            <thead>
              <tr>
                <th>Outcome</th>
                <th>Bid</th>
                <th>Ask</th>
                <th>Ask size</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Up</td>
                <td>{cents(m.up_bid)}</td>
                <td>{cents(m.up_ask)}</td>
                <td>{num(m.up_ask_size, 0)}</td>
              </tr>
              <tr>
                <td>Down</td>
                <td>{cents(m.down_bid)}</td>
                <td>{cents(m.down_ask)}</td>
                <td>{num(m.down_ask_size, 0)}</td>
              </tr>
            </tbody>
          </table>
          <div className="row small" style={{ marginTop: 6 }}>
            <span className="lbl">Market-implied P(Up) (mid)</span>
            <span className="v">{pct(m.implied_p_up_mid)}</span>
          </div>
          <div className="row small">
            <span className="lbl">
              EV/share after fee + 1 tick <span className="tag hyp">hypothetical</span>
            </span>
            <span className="v">
              Up {cents(ev.up)} · Down {cents(ev.down)}
            </span>
          </div>
          <div className="row small">
            <span className="lbl">Taker fee</span>
            <span className="v">
              {s.fee_schedule ? `${s.fee_schedule.rate} × p(1−p)` : "—"}
            </span>
          </div>
        </div>
        <div className="card">
          <h2>
            Volatility &amp; order flow <span className="tag verified">computed from verified feeds</span>
          </h2>
          <div className="row small">
            <span className="lbl">σ (USD per √s)</span>
            <span className="v">{num(s.sigma_usd_per_sqrt_s, 2)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Annualised vol</span>
            <span className="v">{num(s.vol_annualized_pct, 1)}%</span>
          </div>
          <div className="row small">
            <span className="lbl">Vol percentile (7 d)</span>
            <span className="v">{pct(s.vol_pct_7d, 0)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Buy/sell imbalance 60 s (Kraken)</span>
            <span className="v">{num(f.ofi60, 2)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Cumulative delta this round</span>
            <span className="v">{num(f.cvd_round, 2)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Coinbase book imbalance (top 5)</span>
            <span className="v">{num(f.book_imb5_cb, 2)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Exchange − Chainlink gap (σ units)</span>
            <span className="v">{num(f.cl_lag_gap, 2)}</span>
          </div>
          <div className="row small">
            <span className="lbl">Futures basis (bps) · liq. 60 s</span>
            <span className="v">
              {num(f.fut_basis_bps, 1)} · {num(f.liq_net60, 3)}
            </span>
          </div>
        </div>
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="card span2">
          <h2>
            BTC vs price to beat (last 15 min) <span className="tag verified">verified data</span>
          </h2>
          <Chart data={priceData} series={PRICE_SERIES} fmt={(v) => usd(v, 0)} height={280} ariaLabel="Bitcoin price versus the official price to beat" />
        </div>
        <div className="card span2">
          <h2>
            P(Up): model vs market <span className="tag model">model estimate</span>
          </h2>
          <Chart data={probData} series={PROB_SERIES} fmt={(v) => pct(v, 0)} height={180} yRange={[0, 1]} ariaLabel="Model probability versus market-implied probability" />
        </div>
      </div>

      <div className="grid g3" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>
            Most influential variables <span className="tag model">model</span>
          </h2>
          {(s.explain ?? []).map((e: Any) => {
            const c = e.contribution_logodds ?? 0;
            const w = (Math.abs(c) / maxAbs) * 50;
            return (
              <div className="feat" key={e.feature}>
                <span title={e.feature}>{e.feature}</span>
                <div className="track">
                  <div style={{ left: c >= 0 ? "50%" : `${50 - w}%`, width: `${w}%` }} />
                </div>
                <span className="num small">{num(e.value, 2)}</span>
              </div>
            );
          })}
          <div className="lbl" style={{ marginTop: 6 }}>
            Bar: contribution to log-odds of Up (right = towards Up).
          </div>
        </div>
        <div className="card">
          <h2>Data feed health</h2>
          <table>
            <thead>
              <tr>
                <th>Feed</th>
                <th>Status</th>
                <th>Age</th>
                <th>Latency</th>
                <th>Reconn.</th>
              </tr>
            </thead>
            <tbody>
              {(s.health ?? []).map((h: Any) => (
                <tr key={h.name}>
                  <td>{h.name}</td>
                  <td className={h.connected ? "st-ok" : "st-bad"}>{h.connected ? "● up" : "✕ down"}</td>
                  <td>{num(h.age_s, 1)} s</td>
                  <td>{h.latency_median_s == null ? "—" : num(h.latency_median_s, 2) + " s"}</td>
                  <td>{h.reconnects}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="lbl" style={{ marginTop: 6 }}>
            Ledger: {health.ledger ? (health.ledger[0] ? `✓ chain intact (${health.ledger[1]} records)` : `✕ ${health.ledger[2]}`) : "—"}
            {(health.drift?.alerts ?? []).length ? ` · drift alerts: ${health.drift.alerts.join("; ")}` : ""}
          </div>
        </div>
        <div className="card">
          <h2>
            Live paper test <span className="tag hyp">hypothetical</span>
          </h2>
          <div className="hero sm num">
            {paper.settled_rounds ?? 0} / {paper.target_rounds ?? 1000}
          </div>
          <div className="lbl">unseen live rounds settled</div>
          <div className="bar">
            <div style={{ width: `${Math.min(100, (paper.progress ?? 0) * 100)}%` }} />
          </div>
          <div className="scroll" style={{ marginTop: 8 }}>
            <table>
              <thead>
                <tr>
                  <th>k</th>
                  <th>n</th>
                  <th>Brier model</th>
                  <th>Brier market</th>
                  <th>Hyp. PnL/sh</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(paper.by_checkpoint ?? {}).map(([k, v]: [string, Any]) => (
                  <tr key={k}>
                    <td>{k}s</td>
                    <td>{v.n}</td>
                    <td>{num(v.model?.brier, 4)}</td>
                    <td>{num(v.market?.brier, 4)}</td>
                    <td>{v.hypothetical ? cents(v.hypothetical.mean_pnl_per_share) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {paper.warning && <div className="lbl">{paper.warning}</div>}
        </div>
      </div>

      <div className="grid g2" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>
            Historical out-of-sample performance <span className="tag verified">walk-forward test</span>
          </h2>
          {wf ? (
            <div className="scroll">
              <div className="lbl">
                {wf.oos_rounds} unseen rounds · {new Date(wf.oos_range[0] * 1000).toISOString().slice(0, 10)} → {new Date(wf.oos_range[1] * 1000).toISOString().slice(0, 10)} · lower Brier is better
              </div>
              <table>
                <thead>
                  <tr>
                    <th>k</th>
                    <th>Acc. Gaussian</th>
                    <th>Brier Gaussian</th>
                    <th>Brier LightGBM</th>
                    <th>Brier market</th>
                    <th>Δ LGBM−market (95% CI)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(wf.by_checkpoint ?? {}).map(([k, v]: [string, Any]) => {
                    const vm = wf.vs_market?.[k];
                    const d = vm?.p_lightgbm;
                    return (
                      <tr key={k}>
                        <td>{k}s</td>
                        <td>{pct(v.p_gaussian_rw?.accuracy)}</td>
                        <td>{num(v.p_gaussian_rw?.brier, 4)}</td>
                        <td>{num(v.p_lightgbm?.brier, 4)}</td>
                        <td>{num(vm?.market_brier, 4)}</td>
                        <td>{d ? `${num(d.d_vs_market, 4)} [${num(d.ci95[0], 4)}, ${num(d.ci95[1], 4)}]` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="lbl">No evaluation report yet. Run `btc5m evaluate`.</div>
          )}
        </div>
        <div className="card">
          <h2>Recent ledger predictions</h2>
          <div className="scroll" style={{ maxHeight: 320, overflowY: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Round</th>
                  <th>k</th>
                  <th>P(Up)</th>
                  <th>Market</th>
                  <th>Paper side</th>
                  <th>Outcome</th>
                  <th>Hyp. PnL</th>
                </tr>
              </thead>
              <tbody>
                {ledger.map((r: Any) => (
                  <tr key={r.hash}>
                    <td>{new Date(r.round_start * 1000).toLocaleTimeString()}</td>
                    <td>{r.checkpoint}</td>
                    <td>{pct(r.p_up)}</td>
                    <td>{pct(r.market_mid)}</td>
                    <td>{r.would_trade ?? "—"}</td>
                    <td>{r.outcome_up == null ? "pending" : r.outcome_up ? "Up" : "Down"}</td>
                    <td>{r.hyp_pnl == null ? "—" : cents(r.hyp_pnl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <footer>
        Verified data = values received from Polymarket, Chainlink (via Polymarket RTDS), Coinbase and Kraken feeds. Model estimates are statistical outputs and may be wrong. Nothing here is investment advice;
        hypothetical results are not realised profits.
      </footer>
    </div>
  );
}
