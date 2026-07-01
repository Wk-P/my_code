<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import ProgressPanel from "./components/ProgressPanel.vue";
import ExperimentTree from "./components/ExperimentTree.vue";
import RunDetail from "./components/RunDetail.vue";

// #/run/<scenario>/<algo>/<run> routes to a standalone run detail page;
// anything else (including "" and "#/") shows the normal dashboard.
const hash = ref(window.location.hash);
function onHashChange() {
  hash.value = window.location.hash;
}
onMounted(() => window.addEventListener("hashchange", onHashChange));
onUnmounted(() => window.removeEventListener("hashchange", onHashChange));

const runRoute = computed(() => {
  const m = hash.value.match(/^#\/run\/([^/]+)\/([^/]+)\/([^/]+)$/);
  return m ? { scenario: m[1], algo: m[2], run: m[3] } : null;
});
</script>

<template>
  <RunDetail v-if="runRoute" v-bind="runRoute" />

  <template v-else>
    <h1>my-code Experiment Dashboard</h1>
    <div class="sub">Read-only view, does not affect any training process · auto-scanned from results/&lt;scenario&gt;/&lt;algo&gt;/</div>

    <h2>Live Training Progress</h2>
    <ProgressPanel />

    <h2>Results Summary</h2>
    <ExperimentTree />
  </template>
</template>
