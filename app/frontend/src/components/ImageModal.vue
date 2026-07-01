<script setup>
const props = defineProps({
  active: { type: Object, default: null }, // { scenario, algo, run, file } | null
});
const emit = defineEmits(["close"]);
</script>

<template>
  <div
    id="modal"
    :style="{ display: active ? 'flex' : 'none' }"
    @click="emit('close')"
  >
    <img v-if="active" :src="`/api/results/${active.scenario}/${active.algo}/${active.run}/${active.file}`">
  </div>
</template>
