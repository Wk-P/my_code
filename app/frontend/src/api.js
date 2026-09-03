export async function getResults(branch) {
  const url = branch ? `/api/results?branch=${encodeURIComponent(branch)}` : "/api/results";
  const res = await fetch(url);
  return res.json();
}

export async function getProgress() {
  const res = await fetch("/api/progress");
  return res.json();
}

export async function getBatchProgress(batchName) {
  const res = await fetch(`/api/batch_progress/${encodeURIComponent(batchName)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function getBatches() {
  const res = await fetch("/api/batches");
  if (!res.ok) return { batches: [] };
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

export async function getTags() {
  const res = await fetch("/api/tags");
  return res.json();
}

export async function getTagDoc(tag) {
  const res = await fetch(`/api/tags/${encodeURIComponent(tag)}/doc`);
  if (!res.ok) return null;
  return res.json();
}
