// Exercise the complete console script with its actual template IDs and mock APIs.
// No browser, real configuration file or model request is used.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = path.join(__dirname, "..");
const source = fs.readFileSync(path.join(root, "app/static/js/console.js"), "utf8");
const template = fs.readFileSync(path.join(root, "app/templates/console.html"), "utf8");

class Element {
  constructor(value = "") {
    this.value = value; this.checked = false; this.disabled = false;
    this.textContent = ""; this.children = []; this.events = {}; this.style = {};
  }
  set value(value) { this._value = String(value ?? ""); }
  get value() { return this._value; }
  addEventListener(name, callback) { (this.events[name] ??= []).push(callback); }
  async fire(name) {
    for (const callback of this.events[name] || []) await callback({ target: this, currentTarget: this });
  }
  append(...children) { this.children.push(...children); }
  appendChild(child) { this.append(child); }
  replaceChildren(...children) { this.children = children; }
}
const elements = new Map();
for (const match of template.matchAll(/\bid="([^"]+)"/g)) {
  assert.ok(!elements.has(match[1]), `Duplicate template ID: ${match[1]}`);
  const control = new Element();
  control.id = match[1];
  elements.set(match[1], control);
}
const element = (id) => {
  assert.ok(elements.has(id), `JavaScript references missing template ID: ${id}`);
  return elements.get(id);
};
const rows = [new Map(), new Map()];
rows.forEach((row) => {
  for (const cls of ["en-count", "ch-count", "en-provider", "en-model", "ch-provider", "ch-model", "row-enabled"]) {
    row.set("." + cls, new Element());
  }
  row.get(".en-count").value = "17";
  row.get(".ch-count").value = "23";
  row.get(".row-enabled").checked = true;
  row.get(".en-provider").value = row.get(".ch-provider").value = "deepseek";
  row.get(".en-model").value = row.get(".ch-model").value = "fixture-model";
});
const document = {
  getElementById: element,
  createElement: () => new Element(),
  querySelector(selector) {
    const index = /data-idx="(\d+)"/.exec(selector)?.[1];
    return index == null ? null : { querySelector: (cls) => rows[Number(index)].get(cls) || null };
  },
  querySelectorAll(selector) {
    if (selector === ".en-count, .ch-count") return rows.flatMap((row) => [row.get(".en-count"), row.get(".ch-count")]);
    if (selector === ".en-count" || selector === ".ch-count") return rows.map((row) => row.get(selector));
    return [];
  },
};
const main = {
  id: "free_response_v1", name: "main", condition_mode: "natural",
  generation_config: { temperature: 0, top_p: 1, thinking_mode: "disabled", stream: false },
  prompt_config: { contract: "unconstrained", cultural_identity: "none" },
  execution_config: { batch_en: 100, batch_ch: 100, concurrency: 5, scale_concurrency: 2,
    max_retries: 3, max_group_attempts: 200, random_seed: 42, sample_mode: "planned",
    capture_network: true, capture_model_catalog: true, network_guard: "stop" },
};
const china = { ...main, id: "china_identity_v1", name: "china", condition_mode: "controlled",
  prompt_config: { contract: "unconstrained", cultural_identity: "china" } };
const scores = { ...main, id: "free_response_scores_only_v1", name: "scores", condition_mode: "controlled",
  prompt_config: { contract: "free_scores_only", cultural_identity: "none" } };
const presets = [main, china, scores];
let failSave = false;
let lastSaved;
let previewReply = async () => response({ prompt: "exact request text", sections: [{ label: "Source", text: "exact request text" }] });
const response = (value, status = 200) => ({ ok: status === 200, status, statusText: "mock failure", json: async () => value });
const context = vm.createContext({ document, URLSearchParams, alert() {},
  fetch: async (url, options) => {
    if (url === "/api/presets/apply") {
      const id = JSON.parse(options.body).preset_id;
      return response({ preset: presets.find((item) => item.id === id) });
    }
    if (url === "/api/config") {
      if (failSave) return response({ error: "mock save failure" }, 400);
      lastSaved = JSON.parse(options.body);
      return response({ ok: true });
    }
    if (url === "/api/prompt-preview") return previewReply();
    throw new Error("Unexpected API call: " + url);
  },
});
const boot = source.indexOf("  // boot");
assert.ok(boot > 0);
vm.runInContext(source.slice(0, boot) + `
  globalThis.consoleTest = {
    collectConfig, handlePresetSelectionChange, handleConfigurationEdit, previewPrompt, invalidatePromptPreview,
    setState(list, id) { presets = list; activePresetId = id; scales = [{name: "A"}, {name: "B"}]; },
  };
})();`, context, { filename: "console.js" });
const consoleTest = context.consoleTest;
for (const [id, value] of Object.entries({ temperature: 0, top_p: 1, batch_en: 17, batch_ch: 23,
  concurrency: 7, scale_concurrency: 3, max_retries: 2, max_group_attempts: 0,
  network_guard: "warn", condition_mode: "natural", prompt_contract: "unconstrained",
  cultural_identity: "none", provider: "deepseek", model: "fixture-model",
  data_dir: "./中文 量表", results_dir: "./results", "preview-scale": "A", "preview-language": "ch" })) {
  element(id).value = value;
}
consoleTest.setState(presets, main.id);

(async () => {
  // Selection alone never changes the currently applied experiment.
  element("preset_id").value = china.id;
  consoleTest.handlePresetSelectionChange();
  assert.equal(consoleTest.collectConfig().preset_id, main.id);
  assert.equal(consoleTest.collectConfig().prompt_config.cultural_identity, "none");
  await element("btn-apply-preset").fire("click");
  assert.equal(lastSaved.preset_id, china.id);
  assert.equal(lastSaved.prompt_config.cultural_identity, "china");
  assert.equal(lastSaved.concurrency, 5);
  assert.equal(lastSaved.scale_vars.A.ch_count, 100);

  // Editing generation settings retains identity but marks the condition custom.
  element("top_p").value = "0.8";
  consoleTest.handleConfigurationEdit({ currentTarget: element("top_p") });
  assert.equal(consoleTest.collectConfig().condition_mode, "custom");
  assert.equal(consoleTest.collectConfig().prompt_config.cultural_identity, "china");

  await consoleTest.previewPrompt();
  assert.equal(element("btn-copy-prompt").disabled, false);
  assert.equal(element("preview-sections").children[1].textContent, "exact request text");

  // A failed save restores parameters, per-row counts, identity and provenance.
  const before = JSON.stringify(consoleTest.collectConfig());
  failSave = true;
  element("preset_id").value = scores.id;
  await element("btn-apply-preset").fire("click");
  assert.equal(JSON.stringify(consoleTest.collectConfig()), before);
  assert.equal(element("btn-copy-prompt").disabled, true);
  assert.equal(element("preview-sections").children.length, 0);
  failSave = false;
  await element("btn-apply-preset").fire("click");
  assert.equal(lastSaved.prompt_config.contract, "free_scores_only");
  assert.equal(lastSaved.prompt_config.cultural_identity, "none");

  // Discard a preview reply that arrives after settings changed.
  let resolveReply;
  previewReply = () => new Promise((resolve) => { resolveReply = resolve; });
  const pending = consoleTest.previewPrompt();
  consoleTest.invalidatePromptPreview();
  resolveReply(response({ prompt: "stale", sections: [{ text: "stale" }] }));
  await pending;
  assert.equal(element("btn-copy-prompt").disabled, true);
  assert.equal(element("preview-sections").children.length, 0);

  previewReply = async () => response({}, 404);
  await consoleTest.previewPrompt();
  assert.ok(element("preview-note").textContent.includes("重新启动"));
  element("preset_id").value = "custom";
  consoleTest.handlePresetSelectionChange();
  assert.equal(element("btn-apply-preset").disabled, true);
  console.log("Console template IDs, preset selection/apply/rollback, identity edits and preview races passed.");
})().catch((error) => { console.error(error); process.exitCode = 1; });
