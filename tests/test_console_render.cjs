// Run the real table renderer without browser or model API requests.
// Usage: node tests/test_console_render.cjs [scale-metadata.json]
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../app/static/js/console.js"), "utf8");
const start = source.indexOf("  function providerOptions(");
const end = source.indexOf("  async function reloadScales()", start);
assert.ok(start >= 0 && end > start, "Console renderer functions must be present");

const rows = [];
const body = {
  set innerHTML(value) { rows.length = 0; },
  appendChild(row) { rows.push(row); },
};
const providers = ["deepseek", "openai", "qwen", "gemini"];
const context = vm.createContext({
  scales: [], providers,
  modelCache: Object.fromEntries(providers.map((p) => [p, ["fixture-model"]])),
  $: (id) => { assert.equal(id, "scales-body"); return body; },
  normalizeProvider: (value) => providers.includes(value) ? value : "deepseek",
  loadModels: async () => ["fixture-model"],
  prefetchAllModels: async () => {},
  syncSelectAllState() {},
  log(message) { throw new Error(message); },
  document: {
    createElement(tag) {
      assert.equal(tag, "tr");
      const controls = new Map();
      return {
        dataset: {}, innerHTML: "",
        querySelector(selector) {
          if (selector === ".mji-form") return null;
          assert.ok([".en-provider", ".en-model", ".ch-provider", ".ch-model", ".row-enabled"].includes(selector));
          if (!controls.has(selector)) controls.set(selector, { value: "deepseek", addEventListener() {} });
          return controls.get(selector);
        },
      };
    },
  },
});
vm.runInContext(source.slice(start, end), context, { filename: "console-renderer.js" });

const items = process.argv[2] ? JSON.parse(fs.readFileSync(process.argv[2], "utf8")) :
  ["整行乱序", "限定范围乱序", "固定顺序"].map((status, index) => ({
    name: `Fixture ${index}`, shuffle_status: status,
    en_architecture: { kind: "专属架构", ready: true, summary: "English <task>", detail: "detail", parser: "parser" },
    ch_architecture: { ready: false, summary: "中文任务", detail: "结构错误" },
  }));

(async () => {
  // Missing metadata is also a valid input during loading.
  assert.ok(context.architectureHtml(null, undefined).includes("未提供结构信息"));
  await context.renderScales(items);
  assert.equal(rows.length, items.length);
  rows.forEach((row, index) => {
    assert.ok(row.innerHTML.includes(context.escapeHtml(items[index].shuffle_status)));
    assert.equal((row.innerHTML.match(/architecture-card /g) || []).length, 2);
    assert.ok(!row.innerHTML.includes("row-shuffle"));
    assert.ok(!row.innerHTML.includes("architecture-rule"));
  });
  console.log(`Rendered ${rows.length} scale rows and ${rows.length * 2} architecture cards successfully.`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
