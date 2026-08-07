<script setup>
import { ref, onMounted } from "vue";
import { getTags } from "../api.js";

// Doc index page (#/versions): every git tag as a row linking to its own
// detail page (#/versions/<tag>) — no inline expansion here, that lives in
// VersionDetail.vue now that the two are separate routes/pages.
const tags = ref([]);
onMounted(async () => {
  tags.value = await getTags();
});
</script>

<template>
  <div class="version-history card">
    <div v-if="!tags.length" class="empty">no tags found</div>
    <a v-for="t in tags" :key="t.tag" class="version-row version-row--link" :href="`#/versions/${t.tag}`">
      <div class="version-row-head">
        <span class="version-tag exp-id">{{ t.tag }}</span>
        <span class="version-date">{{ t.date }}</span>
        <span class="version-summary">{{ t.summary }}</span>
        <span v-if="t.has_doc" class="version-doc-badge">doc</span>
        <span class="version-toggle">&rsaquo;</span>
      </div>
    </a>
  </div>
</template>
