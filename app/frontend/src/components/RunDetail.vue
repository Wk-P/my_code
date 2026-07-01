<script setup>
import { ref, watch } from "vue";
import { getHistory } from "../api.js";
import { fmt, pct } from "../format.js";

const props = defineProps({
  scenario: { type: String, required: true },
  algo: { type: String, required: true },
  run: { type: String, required: true },
});

const row = ref(null);

watch(
  () => [props.scenario, props.algo, props.run],
  async ([scenario, algo, run]) => {
    row.value = null;
    const rows = await getHistory(scenario, algo);
    row.value = rows.find((r) => r.run === run) ?? null;
  },
  { immediate: true }
);
</script>

<template>
  <a class="thumb-link" href="#/">&larr; back to dashboard</a>
  <h1>{{ scenario }} / {{ algo }} / {{ run }}</h1>

  <div v-if="!row" class="empty">Run not found (or still loading).</div>
  <template v-else>
    <table style="max-width: 600px">
      <tr><th>N/M</th><td>{{ row.N ?? "—" }}/{{ row.M ?? "—" }}</td></tr>
      <tr><th>train/test count</th><td>{{ row.train_count ?? "—" }}/{{ row.test_count ?? "—" }}</td></tr>
      <tr><th>ILP AR</th><td>{{ fmt(row.ilp_ar) }}</td></tr>
      <tr><th>Test AR</th><td>{{ fmt(row.test_ar_mean) }} ± {{ fmt(row.test_ar_std, 3) }}</td></tr>
      <tr><th>Viol Rate (total)</th><td>{{ pct(row.test_viol_rate) }}</td></tr>
      <tr><th>Cap Violations</th><td>{{ row.test_cap_viol_total ?? "—" }}</td></tr>
      <tr><th>Conflict Violations</th><td>{{ row.test_conflict_viol_total ?? "—" }}</td></tr>
      <tr><th>Train AR (last50)</th><td>{{ fmt(row.train_ar_last50) }}</td></tr>
      <tr><th>Train steps</th><td>{{ row.train_steps ? row.train_steps.toLocaleString() : "—" }}</td></tr>
    </table>

    <h2>Training curve</h2>
    <img :src="`/api/results/${scenario}/${algo}/${run}/training_curve.png`" style="max-width: 90vw; border: 1px solid #444">

    <h2>Comparison</h2>
    <img :src="`/api/results/${scenario}/${algo}/${run}/comparison.png`" style="max-width: 90vw; border: 1px solid #444">
  </template>
</template>
