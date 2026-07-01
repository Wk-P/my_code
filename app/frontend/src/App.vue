<script setup>
import { ref } from "vue";
import ProgressPanel from "./components/ProgressPanel.vue";
import ResultsTable from "./components/ResultsTable.vue";
import HistoryModal from "./components/HistoryModal.vue";
import ImageModal from "./components/ImageModal.vue";

const activeHistory = ref(null); // { scenario, algo } | null
const activeImage = ref(null);   // { scenario, algo, run, file } | null

function openHistory(scenario, algo) {
  activeHistory.value = { scenario, algo };
}
function openImage(scenario, algo, run, file) {
  activeImage.value = { scenario, algo, run, file };
}
</script>

<template>
  <h1>my-code Experiment Dashboard</h1>
  <div class="sub">Read-only view, does not affect any training process · auto-scanned from results/&lt;scenario&gt;/&lt;algo&gt;/</div>

  <h2>Live Training Progress</h2>
  <ProgressPanel />

  <h2>Results Summary</h2>
  <ResultsTable @show-history="openHistory" @show-image="openImage" />

  <ImageModal :active="activeImage" @close="activeImage = null" />
  <HistoryModal :active="activeHistory" @close="activeHistory = null" />
</template>
