export async function getResults() {
  const res = await fetch("/api/results");
  return res.json();
}

export async function getProgress() {
  const res = await fetch("/api/progress");
  return res.json();
}

export async function getHistory(scenario, algo, branch) {
  const url = branch
    ? `/api/history/${scenario}/${algo}?branch=${encodeURIComponent(branch)}`
    : `/api/history/${scenario}/${algo}`;
  const res = await fetch(url);
  return res.json();
}

export async function getExperiments(branch) {
  const url = branch ? `/api/experiments?branch=${encodeURIComponent(branch)}` : "/api/experiments";
  const res = await fetch(url);
  return res.json();
}

export async function getBranch() {
  const res = await fetch("/api/branch");
  return res.json();
}
