import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";

export interface SeriesSpec {
  label: string;
  color: string; // CSS var name, e.g. "--s1"
  dash?: number[];
  width?: number;
}

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

interface Props {
  data: (number | null)[][]; // [x, ...series]
  series: SeriesSpec[];
  height?: number;
  yRange?: [number, number];
  fmt: (v: number) => string;
  ariaLabel: string;
}

/** Thin-line time-series chart with crosshair + live readout (the hover layer). */
export default function Chart({ data, series, height = 260, yRange, fmt, ariaLabel }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);
  const [readout, setReadout] = useState<string>("");
  const [theme, setTheme] = useState(0);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setTheme((t) => t + 1);
    mq.addEventListener("change", on);
    const obs = new MutationObserver(on);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", on);
      obs.disconnect();
    };
  }, []);

  useEffect(() => {
    if (!host.current) return;
    const grid = cssVar("--grid");
    const muted = cssVar("--muted");
    const axis = cssVar("--axis");
    const opts: uPlot.Options = {
      width: host.current.clientWidth,
      height,
      legend: { show: false },
      cursor: { drag: { x: false, y: false }, points: { size: 8 } },
      scales: { x: { time: true }, y: yRange ? { range: yRange } : { auto: true } },
      axes: [
        { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: axis, width: 1 } },
        { stroke: muted, grid: { stroke: grid, width: 1 }, ticks: { stroke: axis, width: 1 }, size: 70, values: (_u, vals) => vals.map((v) => fmt(v)) },
      ],
      series: [
        {},
        ...series.map((s) => ({ label: s.label, stroke: cssVar(s.color), width: s.width ?? 2, dash: s.dash, spanGaps: false, points: { show: false } })),
      ],
      hooks: {
        setCursor: [
          (u) => {
            const i = u.cursor.idx;
            if (i == null) return setReadout("");
            const t = new Date((u.data[0][i] as number) * 1000).toLocaleTimeString();
            const parts = series.map((s, j) => {
              const v = u.data[j + 1][i];
              return `${s.label}: ${v == null ? "—" : fmt(v as number)}`;
            });
            setReadout(`${t} · ${parts.join(" · ")}`);
          },
        ],
      },
    };
    plot.current?.destroy();
    plot.current = new uPlot(opts, data as uPlot.AlignedData, host.current);
    const ro = new ResizeObserver(() => host.current && plot.current?.setSize({ width: host.current.clientWidth, height }));
    ro.observe(host.current);
    return () => {
      ro.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme, series.length, height]);

  useEffect(() => {
    plot.current?.setData(data as uPlot.AlignedData);
  }, [data]);

  return (
    <div>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.label}>
            <span className={`sw${s.dash ? " dash" : ""}`} style={s.dash ? undefined : { background: `var(${s.color})` }} />
            {s.label}
          </span>
        ))}
      </div>
      <div ref={host} role="img" aria-label={ariaLabel} />
      <div className="readout">{readout || "Hover the chart for exact values"}</div>
    </div>
  );
}
