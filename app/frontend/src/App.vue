<script setup>
import { ref, computed, onMounted, onUnmounted } from "vue";
import ProgressPanel from "./components/ProgressPanel.vue";
import ExperimentTree from "./components/ExperimentTree.vue";
import RunDetail from "./components/RunDetail.vue";
import VersionList from "./components/VersionList.vue";
import VersionDetail from "./components/VersionDetail.vue";
import { getBranch } from "./api.js";

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
</script>

<template>
  <RunDetail v-if="runRoute" v-bind="runRoute" />

  <VersionDetail v-else-if="versionTagRoute" :tag="versionTagRoute" />

  <template v-else-if="isVersionsIndexRoute">
    <h1>
      Version History
      <a href="#/" class="branch-badge">&larr; back to dashboard</a>
    </h1>
    <div class="sub">Every git tag, newest first · summary from version/VERSION.md · click a row for the full version/vX.Y.Z.md doc when one exists</div>
    <VersionList />
  </template>

  <template v-else>
    <h1>
      my-code Experiment Dashboard
      <span v-if="branch.current" class="branch-badge" :title="'Other branches: ' + (branch.branches.filter(b => b !== branch.current).join(', ') || 'none')">
        {{ branch.current }}
      </span>
      <a href="#/versions" class="branch-badge branch-badge--link">version history</a>
    </h1>
    <div class="sub">Read-only view, does not affect any training process · auto-scanned from results/&lt;branch&gt;/&lt;scenario&gt;/&lt;algo&gt;/ · badge above tracks the checked-out branch, tabs below can browse any branch</div>

    <h2>Live Training Progress</h2>
    <ProgressPanel />

    <h2>Results Summary</h2>
    <ExperimentTree :current-branch="branch.current" :branches="branch.branches" />
  </template>
</template>
