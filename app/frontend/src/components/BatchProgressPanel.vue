<script setup>
// Progress + results view for an ad-hoc parallel batch (many algos x many
// seeds running at once), e.g. scripts/eq_gt_migration_5seed.sh —
// deliberately separate from ProgressPanel.vue, which assumes one
// sequential process per scenario and shows "N/6 models done" against a
// fixed ALGO_ORDER. That model breaks down for a batch like this: 5 algos x
// 5 seeds run concurrently per scenario, so "which one is 'current'" is
// meaningless and the phase-marker log parsing never matches. This instead
// shows a table grouped by scenario -> algo -> seed, one row per run, with
// its status and (once done) its actual success_rate/AR — so it's always
// clear which runs belong together and what they found, without clicking
// into the separate Results Summary tree.
import { ref, onMounted, onUnmounted } from "vue";
import { getBatchProgress } from "../api.js";
import { secToClock } from "../format.js";

const props = defineProps({ batchName: { type: String, required: true } });

const data = ref(null);
let pollTimer = null;

// Elapsed time ticks every second on the client so it reads like a live
// clock, but is resynced to the server's elapsed_seconds (derived from the
// batch's oldest log file) on every 5s poll — client-side drift never
// accumulates for more than one poll interval.
const displaySeconds = ref(null);
let tickTimer = null;
let serverElapsedAtLastPoll = null;
let clientTimeAtLastPoll = null;

async function load() {
  data.value = await getBatchProgress(props.batchName);
  if (data.value?.elapsed_seconds != null) {
    serverElapsedAtLastPoll = data.value.elapsed_seconds;
    clientTimeAtLastPoll = Date.now();
    displaySeconds.value = serverElapsedAtLastPoll;
  }
}

onMounted(() => {
  load();
  pollTimer = setInterval(load, 5000);
  tickTimer = setInterval(() => {
    if (serverElapsedAtLastPoll != null) {
      displaySeconds.value = serverElapsedAtLastPoll + (Date.now() - clientTimeAtLastPoll) / 1000;
    }
  }, 1000);
});
onUnmounted(() => {
  clearInterval(pollTimer);
  clearInterval(tickTimer);
});

const SEED_ORDER = [1, 2, 3, 4, 5];
function algosFor(scenarioBlock) {
  const seen = [];
  for (const r of scenarioBlock?.runs || []) {
    if (!seen.includes(r.algo)) seen.push(r.algo);
  }
  return seen;
}
// A (scenario, algo, seed) cell can have more than one run behind it when a
// crashed/interrupted attempt was retried under a fresh exp_id (see
// scripts/eq_gt_migration_5seed.sh — each launch mints its own exp_id, so a
// retry doesn't overwrite the dead attempt's log). Picking array order would
// surface a stale "queued"/dead entry ahead of the real completed one, so
// rank by status instead: a finished run always wins, then a live one, and
// only fall back to "queued" when nothing better exists.
const STATUS_RANK = { done: 0, running: 1, queued: 2 };
function runFor(scenarioBlock, algo, seed) {
  const matches = (scenarioBlock?.runs || []).filter((r) => r.algo === algo && r.seed === seed);
  if (matches.length === 0) return undefined;
  return matches.reduce((best, r) => (STATUS_RANK[r.status] < STATUS_RANK[best.status] ? r : best));
}
function pct(x) {
  return x === undefined || x === null ? "—" : (x * 100).toFixed(1) + "%";
}
function fixed(x, n = 4) {
  return x === undefined || x === null ? "—" : x.toFixed(n);
}
</script>

<template>
  <div v-if="data" class="card batch-panel">
    <div class="title">
      Batch: {{ data.batch_name }} — {{ data.done }}/{{ data.total_runs }} runs done ({{ data.overall_pct }}%)
      <span class="batch-elapsed" :title="'Time since the batch\'s first run was launched'">
        ⏱ {{ secToClock(displaySeconds) }}
      </span>
    </div>
    <div class="bar-bg">
      <div class="bar-fill batch" :style="{ width: data.overall_pct + '%' }"></div>
    </div>

    <div v-for="(block, scenario) in data.by_scenario" :key="scenario" class="batch-scenario-block">
      <div class="batch-scenario-label">
        {{ scenario }} — {{ block.done }} done · {{ block.running }} running · {{ block.queued }} queued
      </div>
      <table class="batch-table">
        <thead>
          <tr>
            <th>algo</th>
            <th v-for="seed in SEED_ORDER" :key="seed">seed {{ seed }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="algo in algosFor(block)" :key="algo">
            <td class="batch-algo-name">{{ algo }}</td>
            <td v-for="seed in SEED_ORDER" :key="seed" class="batch-cell">
              <template v-if="runFor(block, algo, seed)?.status === 'done'">
                <span class="batch-cell-status batch-cell-status--done">done</span>
                <span class="batch-cell-metric">succ {{ pct(runFor(block, algo, seed).success_rate) }}</span>
                <span class="batch-cell-metric">AR {{ fixed(runFor(block, algo, seed).ar_mean) }}</span>
              </template>
              <span
                v-else
                class="batch-cell-status"
                :class="`batch-cell-status--${runFor(block, algo, seed)?.status ?? 'queued'}`"
              >{{ runFor(block, algo, seed)?.status ?? "queued" }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="batch-legend">
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--done"></span> done</span>
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--running"></span> running</span>
      <span class="batch-legend-item"><span class="batch-seed-dot batch-seed-dot--queued"></span> queued</span>
      <span>rows = algorithm · columns = seed · succ/AR shown once a run finishes</span>
    </div>
  </div>
</template>
