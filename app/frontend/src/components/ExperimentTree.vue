<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from "vue";
import { getExperiments } from "../api.js";
import { fmt, pct } from "../format.js";

const props = defineProps({
  // Current git branch + full branch list from /api/branch, forwarded by
  // App.vue. One tab per real branch; each tab fetches that branch's own
  // results/<branch>/ via ?branch=, not an is_bc filter on the current
  // branch's data — a tab labeled "main" must show main's actual results,
  // not an empty filter over whatever's currently checked out.
  currentBranch: { type: String, default: "" },
  branches: { type: Array, default: () => [] },
});

const SCENARIO_ORDER = ["eq", "gt", "lt"];
const ALGO_ORDER = ["ppo", "ppo_mask", "ppo_lagrangian", "ppo_opt", "dqn", "ddqn"];

const activeBranch = ref("");
watch(
  () => props.currentBranch,
  (b) => { if (b && !activeBranch.value) activeBranch.value = b; },
  { immediate: true }
);

const rows = ref([]);
let timer = null;

async function load() {
  if (!activeBranch.value) return;
  rows.value = await getExperiments(activeBranch.value);
}

onMounted(() => {
  load();
  timer = setInterval(load, 30000);
});
onUnmounted(() => clearInterval(timer));
watch(activeBranch, load);

function orderIndex(order, key) {
  const i = order.indexOf(key);
  return i === -1 ? order.length : i;
}

const expGroups = computed(() => {
  // Group by the real exp_id (one eq+gt+lt batch), not by run_dir.name —
  // run_all_bc.py's run dir is "<exp_id>_bc", which would otherwise split
  // a BC run off into its own fake batch instead of joining the baseline
  // run it was compared against.
  const byExp = new Map();
  for (const r of rows.value) {
    const key = r.exp_id ?? r.run;
    if (!byExp.has(key)) byExp.set(key, { expId: key, latest: r.created_at ?? "", byScenario: new Map() });
    const g = byExp.get(key);
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

function successClass(r) {
  if (r.test_success_rate == null) return "";
  return r.test_success_rate >= 0.999 ? "success-good" : "success-bad";
}

const openInfo = ref(null);
const INFO_TEXT = {
  success: "Of the 40 test scenarios, the fraction where all services were placed successfully with zero violations.",
};
function toggleInfo(key) {
  openInfo.value = openInfo.value === key ? null : key;
}
</script>

<template>
  <div v-if="branches.length > 1" class="tab-bar">
    <button
      v-for="b in branches"
      :key="b"
      class="tab-btn"
      :class="{ active: activeBranch === b }"
      @click="activeBranch = b"
    >
      {{ b }}
      <span v-if="b === currentBranch" class="tab-count" title="currently checked out">•</span>
    </button>
  </div>

  <div v-if="!expGroups.length" class="empty">
    No results yet on branch "{{ activeBranch }}" — the first results.json will show up here once training finishes.
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
            <th>Algo</th><th>N/M</th><th>ILP AR</th>
            <th>Test AR</th>
            <th class="info-th">
              Success Rate
              <button class="info-btn" @click="toggleInfo('success')">i</button>
              <div v-if="openInfo === 'success'" class="info-popup">{{ INFO_TEXT.success }}</div>
            </th>
            <th>Cap Viol Rate</th><th>Conflict Viol Rate</th><th>Train steps</th>
          </tr>
          <tr v-for="r in sc.algoRows" :key="r.algo + r.run" :class="r.is_bc ? 'row-bc' : 'row-baseline'">
            <td style="text-align:left">
              <span class="variant-dot" :class="r.is_bc ? 'variant-bc' : 'variant-baseline'"></span>
              <a class="thumb-link" :href="`#/run/${activeBranch}/${r.scenario}/${r.algo}/${r.run}`">{{ r.algo }}</a>
              <span v-if="r.is_bc" class="variant-badge variant-badge--bc">BC pre-train</span>
              <span v-else class="variant-badge variant-badge--baseline">baseline</span>
            </td>
            <td>{{ r.N ?? "—" }}/{{ r.M ?? "—" }}</td>
            <td>{{ fmt(r.ilp_ar) }}</td>
            <td>{{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }}</td>
            <td :class="successClass(r)">{{ pct(r.test_success_rate) }}</td>
            <td :class="r.test_cap_viol_rate ? 'ar-warn' : ''">{{ pct(r.test_cap_viol_rate) }}</td>
            <td :class="r.test_conflict_viol_rate ? 'ar-warn' : ''">{{ pct(r.test_conflict_viol_rate) }}</td>
            <td>{{ r.train_steps ? r.train_steps.toLocaleString() : "—" }}</td>
          </tr>
        </table>
      </details>
    </details>
  </div>
</template>
