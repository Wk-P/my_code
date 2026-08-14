<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import { getProgress, getBranch } from "../api.js";
import { secToHuman } from "../format.js";

const SCENARIOS = ["eq", "gt", "lt"];
const data = ref({});
const branchInfo = ref(null);
let timer = null;

async function load() {
  // The dashboard server resolves "current branch" fresh on every
  // /api/progress call (see app/backend/main.py's _results_root docstring),
  // so a live process's progress is only found under results/<branch>/...
  // for whichever branch is checked out *right now* — re-fetch alongside
  // progress on every poll instead of once at mount, so a `git checkout` in
  // another terminal is reflected here without reloading the page.
  [data.value, branchInfo.value] = await Promise.all([getProgress(), getBranch()]);
}

onMounted(() => {
  load();
  timer = setInterval(load, 5000);
});
onUnmounted(() => clearInterval(timer));

function classifyLine(line) {
  if (/\[train\]/.test(line)) return "train";
  if (/^===/.test(line.trim())) return "phase";
  if (/\[cache\]/.test(line)) return "cache";
  if (/^\s*$/.test(line)) return "blank";
  return "plain";
}

function displayLine(line) {
  const trimmed = line.trim();
  if (/^===/.test(trimmed)) return trimmed.replace(/^=+\s*/, "").replace(/\s*=+$/, "");
  return line;
}
</script>

<template>
  <div class="progress-branch-banner" v-if="branchInfo">
    Tracking branch: <strong>{{ branchInfo.current ?? "unknown" }}</strong>
    <span class="progress-branch-hint">
      (results/{{ branchInfo.current ?? "unknown" }}/&lt;scenario&gt;/&lt;algo&gt;/ — progress below is scoped to this branch's on-disk results)
    </span>
  </div>
  <div class="progress-grid">
    <div v-for="sc in SCENARIOS" :key="sc" class="progress-card">
      <template v-if="!data[sc]?.running">
        <div class="title" :class="`scenario-${sc}`">{{ sc }}</div>
        <div class="stopped">No algorithm currently running</div>
      </template>
      <template v-else>
        <div class="title" :class="`scenario-${sc}`">
          {{ sc }} — overall progress {{ data[sc].models_done ?? 0 }}/{{ data[sc].models_total ?? 6 }}
          ({{ data[sc].overall_pct ?? 0 }}%)
          <span v-if="data[sc].monitor_status === 'stuck'" class="status-badge status-badge--stuck">
            stuck {{ secToHuman(data[sc].stalled_seconds) }}
          </span>
          <span v-else-if="data[sc].monitor_status === 'unknown'" class="status-badge status-badge--unknown">
            watchdog not running
          </span>
        </div>
        <div class="bar-bg">
          <div class="bar-fill" :class="sc" :style="{ width: (data[sc].overall_pct ?? 0) + '%' }"></div>
        </div>
        <div class="meta">Completed: {{ (data[sc].completed_algos || []).join(", ") || "none" }}</div>
        <div class="meta">
          Current: {{ data[sc].current_algo ?? "?" }} · PID {{ data[sc].pid }} · core {{ data[sc].core }}
          · CPU {{ data[sc].cpu_percent?.toFixed(0) }}% · elapsed {{ secToHuman(data[sc].elapsed_seconds) }}
        </div>
        <template v-if="data[sc].latest_progress">
          <div class="meta">
            <span class="exp-id" :title="'This whole batch (all scenarios launched together) shares exp_id ' + data[sc].latest_progress.exp_id">
              EXP_ID: {{ data[sc].latest_progress.exp_id ?? "pending" }}
            </span>
            <span class="branch-tag" :title="'Code running for this experiment is from branch ' + (branchInfo?.current ?? 'unknown')">
              branch: {{ branchInfo?.current ?? "unknown" }}
            </span>
            <span v-if="data[sc].latest_progress.phase"> · phase: {{ data[sc].latest_progress.phase }}</span>
            <span v-if="data[sc].latest_progress.round !== undefined"> · round {{ data[sc].latest_progress.round }}/{{ data[sc].latest_progress.total_rounds }}</span>
            <span v-if="data[sc].latest_progress.eval_label"> · eval {{ data[sc].latest_progress.eval_label }} ({{ data[sc].latest_progress.eval_step }}/{{ data[sc].latest_progress.eval_total }})</span>
          </div>
          <div class="bar-bg">
            <div class="bar-fill" :class="sc" :style="{ width: (data[sc].latest_progress.pct ?? 0) + '%', opacity: 0.6 }"></div>
          </div>
          <div v-if="data[sc].latest_progress.step !== undefined" class="meta">
            Current model step {{ data[sc].latest_progress.step.toLocaleString() }}/{{ (data[sc].latest_progress.total_steps ?? 0).toLocaleString() }}
            ({{ data[sc].latest_progress.pct }}%) · {{ data[sc].latest_progress.episodes }} episodes
            · {{ (data[sc].latest_progress.steps_per_sec ?? 0).toLocaleString() }} steps/s
          </div>
        </template>
        <div v-else class="meta">Current model has not hit its first progress checkpoint yet</div>
        <div class="log-tail">
          <div class="log-tail-header">live log</div>
          <div class="log-body">
            <div
              v-for="(line, i) in (data[sc].log_tail || [])"
              :key="i"
              class="log-line"
              :class="`log-line--${classifyLine(line)}`"
            >{{ displayLine(line) || " " }}</div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>
