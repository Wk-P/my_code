export const fmt = (v, d = 4) =>
  v === null || v === undefined ? "—" : typeof v === "number" ? v.toFixed(d) : v;

export const pct = (v) =>
  v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%";

export const secToHuman = (s) => {
  if (s === null || s === undefined) return "—";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h${m}m`;
};

// HH:MM:SS, second-granularity — for a live-ticking elapsed-time display
// (secToHuman above intentionally drops seconds, too coarse for a clock).
export const secToClock = (s) => {
  if (s === null || s === undefined) return "—";
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${pad(m)}:${pad(sec)}`;
};
