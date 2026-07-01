<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { getResults } from "../api.js";
import { fmt, pct } from "../format.js";

const emit = defineEmits(["show-history", "show-image"]);

const rows = ref([]);
let timer = null;

async function load() {
  rows.value = await getResults();
}

const sortedRows = computed(() =>
  [...rows.value].sort((a, b) => (a.scenario + a.algo).localeCompare(b.scenario + b.algo))
);

function hasViolation(r) {
  return !!(r.test_cap_viol_total || r.test_conflict_viol_total);
}

onMounted(() => {
  load();
  timer = setInterval(load, 30000);
});
onUnmounted(() => clearInterval(timer));
</script>

<template>
  <div v-if="!rows.length" class="empty">
    No results yet — the first results.json will show up here once training finishes.
  </div>
  <table v-else>
    <tr>
      <th>Scenario</th><th>Algo</th><th>N/M</th><th>train/test count</th>
      <th>ILP AR</th><th>Test AR</th><th>Viol Rate (total)</th><th>Cap Violations</th><th>Conflict Violations</th>
      <th>Train AR (last50)</th><th>Train steps</th><th>Plots</th><th>History</th><th>EXP_ID (hash)</th>
    </tr>
    <tr v-for="r in sortedRows" :key="r.scenario + r.algo">
      <td :class="`scenario-${r.scenario}`">{{ r.scenario }}</td>
      <td>{{ r.algo }}</td>
      <td>{{ r.N ?? "—" }}/{{ r.M ?? "—" }}</td>
      <td>{{ r.train_count ?? "—" }}/{{ r.test_count ?? "—" }}</td>
      <td>{{ fmt(r.ilp_ar) }}</td>
      <td>
        <span
          v-if="hasViolation(r)"
          class="ar-warn"
          title="This run has capacity/conflict violations — its AR isn't directly comparable to ILP's constraint-respecting optimum"
        >{{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }} ⚠</span>
        <span v-else>{{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }}</span>
      </td>
      <td>{{ pct(r.test_viol_rate) }}</td>
      <td>{{ r.test_cap_viol_total ?? "—" }}</td>
      <td>{{ r.test_conflict_viol_total ?? "—" }}</td>
      <td>{{ fmt(r.train_ar_last50) }}</td>
      <td>{{ r.train_steps ? r.train_steps.toLocaleString() : "—" }}</td>
      <td>
        <span class="thumb-link" @click="emit('show-image', r.scenario, r.algo, r.run, 'training_curve.png')">curve</span>
        /
        <span class="thumb-link" @click="emit('show-image', r.scenario, r.algo, r.run, 'comparison.png')">compare</span>
      </td>
      <td><span class="hist-link" @click="emit('show-history', r.scenario, r.algo)">history</span></td>
      <td><span class="run-id">{{ r.run }}</span></td>
    </tr>
  </table>
</template>
