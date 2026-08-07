<script setup>
import { ref, onMounted } from "vue";
import { getTags, getTagDoc } from "../api.js";

// Read-only changelog viewer: lists every git tag with the one-line summary
// from version/VERSION.md, and lazily fetches the full version/vX.Y.Z.md
// body (when one exists) only when a row is expanded — most tags never get
// opened, no reason to pull every doc up front.
const tags = ref([]);
const openTag = ref(null);
const docCache = ref({});
const docLoading = ref(null);

onMounted(async () => {
  tags.value = await getTags();
});

async function toggle(tag) {
  if (openTag.value === tag.tag) {
    openTag.value = null;
    return;
  }
  openTag.value = tag.tag;
  if (tag.has_doc && !docCache.value[tag.tag]) {
    docLoading.value = tag.tag;
    const doc = await getTagDoc(tag.tag);
    if (doc) docCache.value[tag.tag] = doc.content;
    docLoading.value = null;
  }
}
</script>

<template>
  <div class="version-history card">
    <div v-if="!tags.length" class="empty">no tags found</div>
    <div v-for="t in tags" :key="t.tag" class="version-row">
      <div class="version-row-head" @click="toggle(t)">
        <span class="version-tag exp-id">{{ t.tag }}</span>
        <span class="version-date">{{ t.date }}</span>
        <span class="version-summary">{{ t.summary }}</span>
        <span v-if="t.has_doc" class="version-doc-badge">doc</span>
        <span class="version-toggle">{{ openTag === t.tag ? "▾" : "▸" }}</span>
      </div>
      <div v-if="openTag === t.tag" class="version-detail">
        <div v-if="docLoading === t.tag" class="empty">loading…</div>
        <pre v-else-if="docCache[t.tag]" class="version-doc-body">{{ docCache[t.tag] }}</pre>
        <div v-else class="version-doc-body version-doc-body--none">{{ t.summary }}</div>
      </div>
    </div>
  </div>
</template>
