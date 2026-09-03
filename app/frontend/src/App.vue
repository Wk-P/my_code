<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import ProgressPanel from "./components/ProgressPanel.vue";
import BatchProgressPanel from "./components/BatchProgressPanel.vue";
import ExperimentTree from "./components/ExperimentTree.vue";
import RunDetail from "./components/RunDetail.vue";
import VersionList from "./components/VersionList.vue";
import VersionDetail from "./components/VersionDetail.vue";
import PaperDraft from "./components/PaperDraft.vue";
import { getBranch, getBatches } from "./api.js";

// #/run/<branch>/<scenario>/<algo>/<run> routes to a standalone run detail
// page; anything else (including "" and "#/") shows the normal dashboard.
// branch is part of the route (not just "whatever's checked out") because
// ExperimentTree lets you browse any branch's tab without a real `git
// checkout` — drilling into a run has to keep querying that same branch,
// not silently fall back to the currently-checked-out one.
const hash = ref(window.location.hash);
function onHashChange() {
  hash.value = window.location.hash;
}
onMounted(() => window.addEventListener("hashchange", onHashChange));
onUnmounted(() => window.removeEventListener("hashchange", onHashChange));

const runRoute = computed(() => {
  const m = hash.value.match(/^#\/run\/([^/]+)\/([^/]+)\/([^/]+)\/([^/]+)$/);
  return m ? { branch: m[1], scenario: m[2], algo: m[3], run: m[4] } : null;
});

// #/versions is the doc-index page (every tag, one line each); #/versions/<tag>
// drills into that tag's full doc — two separate pages/routes, not an inline
// expand-in-place list, so each has its own linkable/bookmarkable URL.
const versionTagRoute = computed(() => {
  const m = hash.value.match(/^#\/versions\/(.+)$/);
  return m ? m[1] : null;
});
const isVersionsIndexRoute = computed(() => hash.value === "#/versions");

// #/paper is a single standalone page (no sub-routes) -- a paper-draft-style
// writeup of the add_states branch's reward-engineering findings, kept
// separate from the raw version/vX.Y.Z.md dump in #/versions so it can read
// as a coherent narrative instead of one changelog entry per tag.
const isPaperRoute = computed(() => hash.value === "#/paper");

// Auto-detects the checked-out git branch and re-polls so a manual
// `git checkout` elsewhere shows up here without reloading the page.
const branch = ref({ current: null, branches: [], bc_supported: false });
let branchTimer = null;
async function loadBranch() {
  branch.value = await getBranch();
}
onMounted(() => {
  loadBranch();
  branchTimer = setInterval(loadBranch, 10000);
});
onUnmounted(() => clearInterval(branchTimer));

// Auto-discovers every ad-hoc parallel batch under scripts/logs/ (see
// app/backend/main.py's /api/batches) instead of hardcoding batch names
// here — a new batch script (like scripts/gt_full_5M_rerun) shows up on its
// own without a frontend change. Re-polled on the same cadence as branch
// detection since a new batch can start at any time.
const batchNames = ref([]);
let batchesTimer = null;
async function loadBatches() {
  const { batches } = await getBatches();
  batchNames.value = batches.map((b) => b.batch_name);
}
onMounted(() => {
  loadBatches();
  batchesTimer = setInterval(loadBatches, 10000);
});
onUnmounted(() => clearInterval(batchesTimer));
</script>

<template>
  <RunDetail v-if="runRoute" v-bind="runRoute" />

  <VersionDetail v-else-if="versionTagRoute" :tag="versionTagRoute" />

  <PaperDraft v-else-if="isPaperRoute" />

  <template v-else-if="isVersionsIndexRoute">
    <a class="back-btn" href="#/">&larr; Back to dashboard</a>
    <h1>Version History</h1>
    <div class="sub">Every git tag, newest first · summary from version/VERSION.md · click a row for the full version/vX.Y.Z.md doc when one exists</div>
    <VersionList />
  </template>

  <template v-else>
    <h1>
      my-code Experiment Dashboard
      <span v-if="branch.current" class="branch-badge" :title="'Other branches: ' + (branch.branches.filter(b => b !== branch.current).join(', ') || 'none')">
        {{ branch.current }}
      </span>
    </h1>
    <div class="sub">Read-only view, does not affect any training process · auto-scanned from results/&lt;branch&gt;/&lt;scenario&gt;/&lt;algo&gt;/ · badge above tracks the checked-out branch, tabs below can browse any branch</div>

    <div class="intro-card">
      <h3>What am I looking at?</h3>
      <p>
        This is a read-only dashboard over <code>results/&lt;branch&gt;/&lt;scenario&gt;/&lt;algo&gt;/</code> —
        it never starts, stops, or otherwise touches any training run. "Live Training Progress" below shows
        any <code>run_all*.py</code> process currently running on this machine; "Results Summary" shows the
        latest saved result per scenario/algo, click any row to drill into its full history and training
        curve. Everything is scoped to the branch badge above — switch branches on disk (<code>git checkout</code>)
        and this page picks it up automatically within a few seconds, no reload needed.
      </p>
      <div class="intro-nav">
        <a href="#/versions">📜 Version History — every git tag's changelog</a>
        <a href="#/paper">📄 Paper Draft — narrative writeup of the add_states findings</a>
      </div>
    </div>

    <h2>Live Training Progress</h2>
    <BatchProgressPanel v-for="name in batchNames" :key="name" :batch-name="name" />
    <ProgressPanel />

    <h2>Results Summary</h2>
    <ExperimentTree :current-branch="branch.current" :branches="branch.branches" />
  </template>
</template>
