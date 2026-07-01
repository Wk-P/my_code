<script setup>
import { ref, watch } from "vue";
import { getHistory } from "../api.js";
import { fmt, pct } from "../format.js";

const props = defineProps({
  active: { type: Object, default: null }, // { scenario, algo } | null
});
const emit = defineEmits(["close"]);

const rows = ref([]);

watch(
  () => props.active,
  async (val) => {
    if (!val) return;
    rows.value = await getHistory(val.scenario, val.algo);
  }
);
</script>

<template>
  <div
    id="hist-modal"
    :style="{ display: active ? 'flex' : 'none' }"
    @click="(e) => { if (e.target.id === 'hist-modal') emit('close'); }"
  >
    <div class="box">
      <div v-if="!rows.length" class="empty">No history yet</div>
      <template v-else-if="active">
        <h3 style="margin-top:0">{{ active.scenario }} / {{ active.algo }} history runs</h3>
        <table>
          <tr>
            <th>EXP_ID (hash)</th><th>Test AR</th><th>Viol Rate (total)</th><th>Cap Violations</th>
            <th>Conflict Violations</th><th>Train AR (last50)</th><th>steps</th>
          </tr>
          <tr v-for="r in rows" :key="r.run">
            <td style="text-align:left">
              <a class="thumb-link run-id" :href="`#/run/${active.scenario}/${active.algo}/${r.run}`">{{ r.run }}</a>
            </td>
            <td>{{ fmt(r.test_ar_mean) }} ± {{ fmt(r.test_ar_std, 3) }}</td>
            <td>{{ pct(r.test_viol_rate) }}</td>
            <td>{{ r.test_cap_viol_total ?? "—" }}</td>
            <td>{{ r.test_conflict_viol_total ?? "—" }}</td>
            <td>{{ fmt(r.train_ar_last50) }}</td>
            <td>{{ r.train_steps ? r.train_steps.toLocaleString() : "—" }}</td>
          </tr>
        </table>
      </template>
    </div>
  </div>
</template>
