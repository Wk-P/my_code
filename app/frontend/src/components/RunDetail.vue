<script setup>
import { ref, watch } from "vue";
import { getHistory } from "../api.js";
import { fmt, pct } from "../format.js";

const props = defineProps({
  branch:   { type: String, required: true },
  scenario: { type: String, required: true },
  algo:     { type: String, required: true },
  run:      { type: String, required: true },
});

const row = ref(null);

watch(
  () => [props.branch, props.scenario, props.algo, props.run],
  async ([branch, scenario, algo, run]) => {
    row.value = null;
    const rows = await getHistory(scenario, algo, branch);
    row.value = rows.find((r) => r.run === run) ?? null;
  },
  { immediate: true }
);
</script>

<template>
  <a class="thumb-link" href="#/">&larr; back to dashboard</a>
  <h1>{{ branch }} / {{ scenario }} / {{ row?.display_algo ?? algo }} / {{ run }}</h1>

  <div v-if="!row" class="empty">Run not found (or still loading).</div>
  <template v-else>
    <table style="max-width: 600px">
      <tr><th>N/M</th><td>{{ row.N ?? "—" }}/{{ row.M ?? "—" }}</td></tr>
      <tr><th>train/test count</th><td>{{ row.train_count ?? "—" }}/{{ row.test_count ?? "—" }}</td></tr>
      <tr><th>ILP AR</th><td>{{ fmt(row.ilp_ar) }}</td></tr>
      <tr><th>Test AR</th><td>{{ fmt(row.test_ar_mean) }} ± {{ fmt(row.test_ar_std, 3) }}</td></tr>
      <tr><th>Success Rate</th><td>{{ pct(row.test_success_rate) }}</td></tr>
      <tr><th>Cap Viol Rate</th><td>{{ pct(row.test_cap_viol_rate) }}</td></tr>
      <tr><th>Conflict Viol Rate</th><td>{{ pct(row.test_conflict_viol_rate) }}</td></tr>
      <tr><th>Cap Violations (total)</th><td>{{ row.test_cap_viol_total ?? "—" }}</td></tr>
      <tr><th>Conflict Violations (total)</th><td>{{ row.test_conflict_viol_total ?? "—" }}</td></tr>
      <tr><th>Train AR (last50)</th><td>{{ fmt(row.train_ar_last50) }}</td></tr>
      <tr><th>Train steps</th><td>{{ row.train_steps ? row.train_steps.toLocaleString() : "—" }}</td></tr>
    </table>

    <h2>Training curve</h2>
    <img :src="`/api/results/${scenario}/${algo}/${run}/training_curve.png?branch=${encodeURIComponent(branch)}`" style="max-width: 90vw; border: 1px solid #444">

    <h2>Comparison</h2>
    <img :src="`/api/results/${scenario}/${algo}/${run}/comparison.png?branch=${encodeURIComponent(branch)}`" style="max-width: 90vw; border: 1px solid #444">
  </template>
</template>
