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
        <div class="log-tail">{{ (data[sc].log_tail || []).join("\n") }}</div>
      </template>
    </div>
  </div>
</template>
