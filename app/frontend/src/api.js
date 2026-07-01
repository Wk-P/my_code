export async function getResults() {
  const res = await fetch("/api/results");
  return res.json();
}

export async function getProgress() {
  const res = await fetch("/api/progress");
  return res.json();
}

export async function getHistory(scenario, algo) {
  const res = await fetch(`/api/history/${scenario}/${algo}`);
  return res.json();
}
