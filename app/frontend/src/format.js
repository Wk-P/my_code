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
