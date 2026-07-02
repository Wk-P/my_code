<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import { getExperiments } from "../api.js";
import { fmt, pct } from "../format.js";

const SCENARIO_ORDER = ["eq", "gt", "lt"];
const ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"];

const rows = ref([]);
let timer = null;

async function load() {
  rows.value = await getExperiments();
}

onMounted(() => {
  load();
  timer = setInterval(load, 30000);
});
onUnmounted(() => clearInterval(timer));

function orderIndex(order, key) {
  const i = order.indexOf(key);
  return i === -1 ? order.length : i;
}

const expGroups = computed(() => {
  const byExp = new Map();
  for (const r of rows.value) {
    if (!byExp.has(r.run)) byExp.set(r.run, { expId: r.run, latest: r.created_at ?? "", byScenario: new Map() });
    const g = byExp.get(r.run);
    if ((r.created_at ?? "") > g.latest) g.latest = r.created_at ?? "";
    if (!g.byScenario.has(r.scenario)) g.byScenario.set(r.scenario, []);
    g.byScenario.get(r.scenario).push(r);
  }

  const groups = [...byExp.values()].sort((a, b) => b.latest.localeCompare(a.latest));
  for (const g of groups) {
    g.scenarios = [...g.byScenario.entries()]
      .sort((a, b) => orderIndex(SCENARIO_ORDER, a[0]) - orderIndex(SCENARIO_ORDER, b[0]))
      .map(([scenario, algoRows]) => ({
        scenario,
        algoRows: algoRows.sort((a, b) => orderIndex(ALGO_ORDER, a.algo) - orderIndex(ALGO_ORDER, b.algo)),
      }));
  }
  return groups;
});

function hasViolation(r) {
  return !!(r.test_cap_viol_total || r.test_conflict_viol_total);
}

function successClass(r) {
  if (r.test_success_rate == null) return "";
  return r.test_success_rate >= 0.999 ? "success-good" : "success-bad";
}
</script>

<template>
  <div v-if="!expGroups.length" class="empty">
    No results yet — the first results.json will show up here once training finishes.
  </div>
  <div v-else class="exp-tree">
    <details v-for="(g, i) in expGroups" :key="g.expId" class="card exp-group" :open="i === 0">
      <summary>
        <span class="run-id">{{ g.expId }}</span>
        <span class="summary-meta">{{ g.scenarios.length }} scenario{{ g.scenarios.length > 1 ? "s" : "" }} · {{ g.latest || "—" }}</span>
      </summary>

      <details v-for="sc in g.scenarios" :key="sc.scenario" class="scenario-group" :open="i === 0">
        <summary>
          <span :class="`scenario-${sc.scenario}`">{{ sc.scenario }}</span>
          <span class="summary-meta">{{ sc.algoRows.length }} model{{ sc.algoRows.length > 1 ? "s" : "" }}</span>
        </summary>

        <table>
          <tr>
            <th>Algo</th><th>N/M</th><th>ILP AR</th><th>Test AR</th>
            <th>Success Rate</th><th>Cap Viol Rate</th><th>Conflict Viol Rate</th><th>Train steps</th>
          </tr>
          <tr v-for="r in sc.algoRows" :key="r.algo">
            <td style="text-align:left">
              <a class="thumb-link" :href="`#/run/${r.scenario}/${r.algo}/${r.run}`">{{ r.algo }}</a>
            </td>
            <td>{{ r.N ?? "—" }}/{{ r.M ?? "—" }}</td>
            <td>{{ fmt(r.ilp_ar) }}</td>
            <td>
              <span v-if="hasViolation(r)" class="ar-warn" title="Capacity/conflict violations present — not directly comparable to ILP's constraint-respecting optimum">
                {{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }} ⚠
              </span>
              <span v-else>{{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }}</span>
            </td>
            <td :class="successClass(r)" title="Share of the 40 test scenarios where all M services were placed with zero capacity/conflict violations">{{ pct(r.test_success_rate) }}</td>
            <td>{{ pct(r.test_cap_viol_rate) }}</td>
            <td>{{ pct(r.test_conflict_viol_rate) }}</td>
            <td>{{ r.train_steps ? r.train_steps.toLocaleString() : "—" }}</td>
          </tr>
        </table>
      </details>
    </details>
  </div>
</template>
