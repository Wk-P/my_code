<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import { getProgress } from "../api.js";
import { secToHuman } from "../format.js";

const SCENARIOS = ["eq", "gt", "lt"];
const data = ref({});
let timer = null;

async function load() {
  data.value = await getProgress();
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
          </div>
          <div class="bar-bg">
            <div class="bar-fill" :class="sc" :style="{ width: data[sc].latest_progress.pct + '%', opacity: 0.6 }"></div>
          </div>
          <div class="meta">
            Current model step {{ data[sc].latest_progress.step.toLocaleString() }}/{{ data[sc].latest_progress.total_steps.toLocaleString() }}
            ({{ data[sc].latest_progress.pct }}%) · {{ data[sc].latest_progress.episodes }} episodes
            · {{ data[sc].latest_progress.steps_per_sec.toLocaleString() }} steps/s
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
