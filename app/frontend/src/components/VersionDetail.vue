<script setup>
import { ref, watch } from "vue";
import { getTags, getTagDoc } from "../api.js";

const props = defineProps({
  tag: { type: String, required: true },
});

const meta = ref(null);   // this tag's row from /api/tags (summary/date/has_doc)
const doc = ref(null);    // full doc body, only fetched when has_doc is true
const loading = ref(true);

watch(
  () => props.tag,
  async (tag) => {
    meta.value = null;
    doc.value = null;
    loading.value = true;
    const tags = await getTags();
    meta.value = tags.find((t) => t.tag === tag) ?? null;
    if (meta.value?.has_doc) {
      doc.value = await getTagDoc(tag);
    }
    loading.value = false;
  },
  { immediate: true }
);
</script>

<template>
  <a class="thumb-link" href="#/versions">&larr; back to version history</a>
  <h1>{{ tag }}</h1>

  <div v-if="loading" class="empty">loading…</div>
  <div v-else-if="!meta" class="empty">Tag not found.</div>
  <template v-else>
    <div class="sub">{{ meta.date }}</div>
    <div class="version-doc-body version-summary-full">{{ meta.summary }}</div>
    <pre v-if="doc" class="version-doc-body">{{ doc.content }}</pre>
    <div v-else class="empty">No standalone doc for this tag — see the summary above (from version/VERSION.md).</div>
  </template>
</template>
