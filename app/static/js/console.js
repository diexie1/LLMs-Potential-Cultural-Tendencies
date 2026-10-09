(() => {
  const $ = (id) => document.getElementById(id);
  const providers = ["deepseek", "openai", "qwen", "gemini"];
  const modelCache = {};
  let scales = [];
  let presets = [];
  let applyingPreset = false;
  let activePresetId = "custom";
  let randomSeed = null;
  let sse = null;
  let previewText = "";
  let previewRequest = 0;
  let answerRows = [];
  let answerResultsDir = "";
  let answerRequest = 0;

  const GENERATION_CONTROL_IDS = new Set(["temperature", "top_p"]);
  const PROMPT_CONTRACT_LABELS = {
    strict: "量表原文",
    free: "量表原文",
    unconstrained: "量表原文",
    free_scores_only: "量表原文 + 只答分数提示",
  };

  function log(msg, cls = "info") {
    const box = $("log-box");
    const line = document.createElement("div");
    line.className = cls;
    line.textContent = msg;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  function logClass(msg) {
    if (msg.includes("[成功") || msg.includes("[已保存") || msg.includes("完成") || msg.includes("已加载")) return "ok";
    if (msg.includes("[失败") || msg.includes("[重试") || msg.includes("错误") || msg.includes("失败"))
      return "err";
    return "info";
  }

  async function api(url, opts = {}) {
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
      ...opts,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const missingEndpoint = res.status === 404 || res.status === 405;
      throw new Error(data.error || (missingEndpoint
        ? "当前后台未提供此功能，请关闭旧平台后台并重新启动，再刷新页面。"
        : res.statusText || "请求失败"));
    }
    return data;
  }

  function activePreset() {
    return presets.find((item) => item.id === activePresetId) || null;
  }

  function appliedPresetId() {
    return activePresetId || "custom";
  }

  function handlePresetSelectionChange() {
    const select = $("preset_id");
    if (!select) return;
    const selectedId = select.value || "custom";
    if (selectedId === "custom") {
      activePresetId = "custom";
      $("condition_mode").value = "custom";
    }
    // Choosing another named scheme only previews it. The current run stays
    // tied to activePresetId until the explicit apply-and-save action.
    updatePresetButton();
  }

  function comparable(value) {
    if (value === undefined || value === null || value === "") return null;
    if (Array.isArray(value)) return value.map(comparable);
    if (typeof value === "object") {
      return Object.fromEntries(
        Object.keys(value).sort().map((key) => [key, comparable(value[key])]),
      );
    }
    return value;
  }

  function valuesMatch(left, right) {
    return JSON.stringify(comparable(left)) === JSON.stringify(comparable(right));
  }

  function currentExecutionConfig() {
    return {
      batch_en: Number($("batch_en").value || 0),
      batch_ch: Number($("batch_ch").value || 0),
      concurrency: Number($("concurrency").value || 5),
      scale_concurrency: Number($("scale_concurrency").value || 2),
      max_retries: Number($("max_retries").value || 1),
      max_group_attempts: Number($("max_group_attempts").value || 0),
      sample_mode: "planned",
      random_seed: randomSeed,
      strict_capabilities: false,
      capture_network: Boolean($("capture_network").checked),
      capture_model_catalog: Boolean($("capture_model_catalog").checked),
      network_guard: $("network_guard").value || "stop",
    };
  }

  function updateConditionModeFromPreset() {
    if (applyingPreset || activePresetId === "custom") return;
    const preset = activePreset();
    if (!preset) return;
    try {
      const currentGeneration = collectGenerationConfig();
      const contractMatches =
        !preset.prompt_config?.contract
        || $("prompt_contract")?.value === preset.prompt_config.contract;
      const identityMatches = ($("cultural_identity")?.value || "none") === (preset.prompt_config?.cultural_identity || "none");
      const matchesPreset = contractMatches && identityMatches && Object.keys(currentGeneration).every(
        (key) => valuesMatch(currentGeneration[key], preset.generation_config?.[key]),
      );
      $("condition_mode").value = matchesPreset ? preset.condition_mode : "custom";
    } catch (_) {
      // Keep the last valid label while a parameter is being edited.
    }
  }

  function updatePresetButton() {
    $("btn-apply-preset").disabled = ($("preset_id").value || "custom") === "custom";
  }

  function cancelUnappliedPreviewOnEdit() {
    if (applyingPreset) return;
    const select = $("preset_id");
    if (select && select.value !== "custom" && select.value !== activePresetId) {
      select.value = activePresetId;
    }
  }

  function handleConfigurationEdit(event) {
    if (applyingPreset) return;
    invalidatePromptPreview();
    cancelUnappliedPreviewOnEdit();
    if (GENERATION_CONTROL_IDS.has(event.currentTarget.id)) updateConditionModeFromPreset();
    updatePresetButton();
  }

  function bindPresetDirtyTracking() {
    const ids = [
      ...GENERATION_CONTROL_IDS,
      "capture_network", "capture_model_catalog", "sample_mode", "max_retries",
      "max_group_attempts", "concurrency", "scale_concurrency", "batch_en", "batch_ch",
      "data_dir",
      "network_guard",
    ];
    ids.forEach((id) => {
      const el = $(id);
      if (!el) return;
      el.addEventListener("change", handleConfigurationEdit);
      el.addEventListener("input", handleConfigurationEdit);
    });
    $("scales-body")?.addEventListener("input", (event) => {
      if (event.target.matches(".en-count, .ch-count")) handleConfigurationEdit(event);
    });
  }

  async function loadPresets() {
    const data = await api("/api/presets");
    presets = Array.isArray(data.presets) ? data.presets : [];
    const select = $("preset_id");
    if (select) {
      select.innerHTML = '<option value="custom">自定义（当前配置）</option>';
      presets.forEach((item) => {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = item.name;
        select.appendChild(option);
      });
      select.addEventListener("change", handlePresetSelectionChange);
    }
    updatePresetButton();
  }

  async function loadModels(provider, force = false) {
    const p = normalizeProvider(provider);
    if (!force && modelCache[p]?.length) return modelCache[p];
    const data = await api(`/api/models?provider=${encodeURIComponent(p)}`);
    modelCache[p] = data.models || [];
    return modelCache[p];
  }

  function normalizeProvider(p) {
    const v = String(p || "").trim();
    return providers.includes(v) ? v : "deepseek";
  }

  async function prefetchAllModels() {
    await Promise.all(providers.map((p) => loadModels(p, true)));
  }

  async function fillGlobalModels() {
    const provider = normalizeProvider($("provider").value);
    if ($("provider").value !== provider) $("provider").value = provider;
    const sel = $("model");
    const prev = sel.value;
    sel.innerHTML = "";
    const loading = document.createElement("option");
    loading.textContent = "加载中…";
    loading.disabled = true;
    sel.appendChild(loading);
    try {
      const models = await loadModels(provider, true);
      if ($("provider").value !== provider) return;
      sel.innerHTML = "";
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        sel.appendChild(opt);
      });
      if (prev && models.includes(prev)) sel.value = prev;
      else if (models.length) sel.value = models[0];
    } catch (e) {
      if ($("provider").value !== provider) return;
      sel.innerHTML = "";
      log(`加载 ${provider} 模型失败: ${e.message || e}`, "err");
    }
  }

  function collectScaleVars() {
    const vars = {};
    scales.forEach((s, idx) => {
      const row = document.querySelector(`tr[data-idx="${idx}"]`);
      if (!row) return;
      vars[s.name] = {
        enabled: row.querySelector(".row-enabled").checked,
        en_provider: row.querySelector(".en-provider").value,
        en_model: row.querySelector(".en-model").value,
        ch_provider: row.querySelector(".ch-provider").value,
        ch_model: row.querySelector(".ch-model").value,
        en_count: Number(row.querySelector(".en-count").value || 0),
        ch_count: Number(row.querySelector(".ch-count").value || 0),
        ...(row.querySelector(".mji-form")
          ? { mji_form: row.querySelector(".mji-form").value || "all" }
          : {}),
      };
    });
    return vars;
  }

  function setAllEnabled(enabled) {
    const rows = [...document.querySelectorAll("tr[data-idx]")];
    rows.forEach((row) => {
      const control = row.querySelector(".row-enabled");
      if (control) control.checked = enabled;
    });
    const selectAll = $("select-all-scales");
    if (selectAll) {
      selectAll.checked = enabled;
      selectAll.indeterminate = false;
    }
    log(`已${enabled ? "全选" : "取消选择"} ${rows.length} 个量表。点击“保存配置”后将保留该设置。`, "ok");
  }

  function syncSelectAllState() {
    const rows = [...document.querySelectorAll("tr[data-idx] .row-enabled")];
    const selectAll = $("select-all-scales");
    if (!selectAll || !rows.length) return;
    const checked = rows.filter((el) => el.checked).length;
    selectAll.checked = checked === rows.length;
    selectAll.indeterminate = checked > 0 && checked < rows.length;
  }

  function optionalNumber(id, label) {
    const el = $(id);
    if (!el) return null;
    const raw = String(el.value ?? "").trim();
    if (!raw) return null;
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`${label} 必须是数字`);
    return value;
  }

  function collectGenerationConfig() {
    return {
      temperature: optionalNumber("temperature", "Temperature") ?? 0.7,
      top_p: optionalNumber("top_p") ?? 1,
      thinking_mode: "disabled",
      stream: false,
    };
  }

  function setOptionalValue(id, value) {
    const el = $(id);
    if (el) el.value = value == null ? "" : value;
  }

  function fillGenerationConfig(generation) {
    const g = generation || {};
    setOptionalValue("temperature", g.temperature);
    setOptionalValue("top_p", g.top_p ?? 1);
  }

  function collectConfig() {
    return {
      data_dir: $("data_dir").value.trim() || "./data",
      results_dir: $("results_dir").value.trim() || "./results",
      provider: $("provider").value,
      model: $("model").value,
      preset_id: appliedPresetId(),
      temperature: optionalNumber("temperature", "Temperature"),
      condition_mode: $("condition_mode").value || "natural",
      prompt_config: {
        contract: $("prompt_contract")?.value || "unconstrained",
        cultural_identity: $("cultural_identity")?.value || "none",
      },
      ...currentExecutionConfig(),
      scale_vars: collectScaleVars(),
      generation_config: collectGenerationConfig(),
    };
  }

  function providerOptions(selected) {
    return providers
      .map((p) => `<option value="${p}" ${p === selected ? "selected" : ""}>${p}</option>`)
      .join("");
  }

  function modelOptions(models, selected) {
    const list = Array.isArray(models) ? models : [];
    let cur = selected || "";
    if (!cur || !list.includes(cur)) cur = list[0] || "";
    if (!list.length) {
      return `<option value="" disabled selected>暂无可用模型</option>`;
    }
    return list
      .map((m) => `<option value="${escapeHtml(m)}" ${m === cur ? "selected" : ""}>${escapeHtml(m)}</option>`)
      .join("");
  }

  async function fillRowModels(providerSel, modelSel, preferredModel) {
    const provider = normalizeProvider(providerSel.value);
    if (providerSel.value !== provider) providerSel.value = provider;
    const prev = preferredModel != null ? preferredModel : modelSel.value;
    modelSel.innerHTML = `<option disabled selected>加载中…</option>`;
    try {
      const list = await loadModels(provider, true);
      if (providerSel.value !== provider) return;
      modelSel.innerHTML = modelOptions(list, prev);
    } catch (e) {
      if (providerSel.value !== provider) return;
      modelSel.innerHTML = `<option value="" disabled selected>加载失败</option>`;
      log(`加载 ${provider} 模型失败: ${e.message || e}`, "err");
    }
  }

  function bindProviderModel(providerSel, modelSel) {
    providerSel.addEventListener("change", () => {
      fillRowModels(providerSel, modelSel, "");
    });
  }

  async function renderScales(items) {
    scales = items;
    const body = $("scales-body");
    body.innerHTML = "";

    // 预加载全部提供方模型，切换公司时下拉框可立即填充
    try {
      await prefetchAllModels();
    } catch (e) {
      log(`预加载模型列表失败: ${e.message || e}`, "err");
    }

    for (let idx = 0; idx < items.length; idx++) {
      const s = items[idx];
      const tr = document.createElement("tr");
      tr.dataset.idx = String(idx);
      const enP = normalizeProvider(s.en_provider);
      const chP = normalizeProvider(s.ch_provider);
      const enModels = modelCache[enP] || [];
      const chModels = modelCache[chP] || [];
      const profileErrors = Object.entries(s.profile_errors || {})
        .map(([lang, msg]) => `${lang}: ${msg}`)
        .join("\n");
      const profileLabel = String(s.profile_label || "").trim();
      const profileNote = [profileLabel, profileErrors].filter(Boolean).join("\n");
      const profileBadge = profileLabel
        ? `<div style="font-size:11px;opacity:.72;margin-top:3px" title="${escapeHtml(profileNote)}">${escapeHtml(profileLabel)}${profileErrors ? " ⚠" : ""}</div>`
        : "";
      const architecture = architectureHtml(s.en_architecture, s.ch_architecture);
      const mjiOptions = Array.isArray(s.mji_form_options) && s.mji_form_options.length
        ? `<label class="mji-form-control">施测形式<select class="mji-form" aria-label="${escapeHtml(s.name)} 施测形式">${s.mji_form_options.map((option) => `<option value="${escapeHtml(option.value)}" ${option.value === (s.mji_form || "all") ? "selected" : ""}>${escapeHtml(option.label)}</option>`).join("")}</select></label>`
        : "";

      tr.innerHTML = `
        <td class="check"><input class="row-enabled" type="checkbox" ${s.enabled ? "checked" : ""} aria-label="启用 ${escapeHtml(s.name)}" /></td>
        <td class="name" title="${escapeHtml(profileNote)}">${escapeHtml(s.name)}${profileBadge}</td>
        <td><div class="language-config"><input class="en-count" type="number" min="0" step="1" value="${Number(s.en_count) || 0}" aria-label="${escapeHtml(s.name)} 英文次数" /><details class="scale-api-settings"><summary>API / 模型</summary><select class="en-provider" aria-label="${escapeHtml(s.name)} 英文 API">${providerOptions(enP)}</select><select class="en-model" aria-label="${escapeHtml(s.name)} 英文模型">${modelOptions(enModels, s.en_model)}</select></details></div></td>
        <td><div class="language-config"><input class="ch-count" type="number" min="0" step="1" value="${Number(s.ch_count) || 0}" aria-label="${escapeHtml(s.name)} 中文次数" /><details class="scale-api-settings"><summary>API / 模型</summary><select class="ch-provider" aria-label="${escapeHtml(s.name)} 中文 API">${providerOptions(chP)}</select><select class="ch-model" aria-label="${escapeHtml(s.name)} 中文模型">${modelOptions(chModels, s.ch_model)}</select></details></div></td>
        <td class="check"><span class="shuffle-status">${escapeHtml(s.shuffle_status || (s.shuffle_items ? "乱序" : "固定顺序"))}</span></td>
        <td class="settings-cell"><details class="scale-settings"><summary>查看</summary><div class="scale-settings-body"><div class="scale-stats">题数：英 ${s.en_n} / 中 ${s.ch_n}　附图：英 ${Number(s.en_images) || 0} / 中 ${Number(s.ch_images) || 0}</div>${mjiOptions}${architecture}</div></details></td>
      `;
      body.appendChild(tr);
      const enProv = tr.querySelector(".en-provider");
      const enModel = tr.querySelector(".en-model");
      const chProv = tr.querySelector(".ch-provider");
      const chModel = tr.querySelector(".ch-model");
      bindProviderModel(enProv, enModel);
      bindProviderModel(chProv, chModel);
      // 若预加载为空（例如密钥未就绪），切换/进入时再拉一次
      if (!enModels.length) fillRowModels(enProv, enModel, s.en_model);
      if (!chModels.length) fillRowModels(chProv, chModel, s.ch_model);
      tr.querySelector(".row-enabled").addEventListener("change", syncSelectAllState);
      tr.querySelector(".mji-form")?.addEventListener("change", handleConfigurationEdit);
    }
    syncSelectAllState();
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function architectureHtml(enMeta, chMeta) {
    const card = (language, meta) => {
      const item = meta || {};
      const ready = item.ready !== false;
      const isSpecial = item.kind === "专属架构";
      const tone = !ready ? "is-error" : (isSpecial ? "is-special" : "is-default");
      const kind = item.kind || "加载中";
      const summary = item.summary || "未提供结构信息";
      const parser = item.parser ? `解析：${item.parser}` : "";
      const detail = item.detail || "";
      const title = [detail, parser].filter(Boolean).join("\n");
      return `
        <div class="architecture-card ${tone}" title="${escapeHtml(title)}">
          <div class="architecture-top"><span class="architecture-language">${language}</span><span class="architecture-kind">${escapeHtml(kind)}</span></div>
          <div class="architecture-summary">${escapeHtml(summary)}</div>
          ${!ready && detail ? `<div class="architecture-error">⚠ ${escapeHtml(detail)}</div>` : ""}
        </div>`;
    };
    return `${card("英", enMeta)}${card("中", chMeta)}`;
  }

  async function reloadScales() {
    const dataDir = $("data_dir").value.trim() || "./data";
    const data = await api(`/api/scales?data_dir=${encodeURIComponent(dataDir)}`);
    await renderScales(data.scales || []);
    fillPreviewScales();
    invalidatePromptPreview();
    updatePresetButton();
    log(`已加载 ${(data.scales || []).length} 个量表（${data.resolved}）`, "ok");
  }

  async function loadConfig() {
    const cfg = await api("/api/config");
    await loadPresets();
    $("cultural_identity").value = cfg.prompt_config?.cultural_identity || "none";
    $("data_dir").value = cfg.data_dir || "./data";
    $("results_dir").value = cfg.results_dir || "./results";
    $("provider").value = cfg.provider || "deepseek";
    randomSeed = cfg.random_seed ?? null;
    activePresetId = [...$("preset_id").options].some((o) => o.value === (cfg.preset_id || "custom"))
      ? (cfg.preset_id || "custom")
      : "custom";
    $("condition_mode").value = cfg.condition_mode || "custom";
    if ($("prompt_contract")) {
      const savedContract = cfg.prompt_config?.contract || "unconstrained";
      $("prompt_contract").value = PROMPT_CONTRACT_LABELS[savedContract] ? savedContract : "unconstrained";
    }
    $("preset_id").value = activePresetId;
    $("temperature").value = cfg.temperature ?? 0.7;
    $("concurrency").value = cfg.concurrency ?? 5;
    $("scale_concurrency").value = cfg.scale_concurrency ?? 2;
    $("batch_en").value = cfg.batch_en ?? 100;
    $("batch_ch").value = cfg.batch_ch ?? 100;
    fillGenerationConfig(cfg.generation_config || {});
    $("capture_network").checked = cfg.capture_network !== false;
    $("capture_model_catalog").checked = cfg.capture_model_catalog !== false;
    $("network_guard").value = cfg.network_guard || "stop";
    $("max_retries").value = cfg.max_retries ?? 10;
    $("max_group_attempts").value = cfg.max_group_attempts ?? 0;
    $("sample_mode").value = "planned";
    await fillGlobalModels();
    if (cfg.model) {
      const sel = $("model");
      if ([...sel.options].some((o) => o.value === cfg.model)) sel.value = cfg.model;
    }
    await reloadScales();
    updatePresetButton();
  }

  async function refreshNetwork() {
    $("net-ip").textContent = "公网 IP: 检测中…";
    $("net-proxy").textContent = "系统代理: 检测中…";
    try {
      const n = await api("/api/network");
      const org = n.org ? `  |  运营商: ${n.org}` : "";
      $("net-ip").textContent = `公网 IP: ${n.ip || "未知"}  |  位置: ${n.location || "未知"}${org}`;
      $("net-proxy").textContent = n.proxy
        ? `系统代理: ${n.proxy}`
        : "系统代理: 未检测到（将直连）";
      log(n.text || "网络检测完成", "info");
    } catch (e) {
      $("net-ip").textContent = `公网 IP: 查询失败`;
      log(String(e.message || e), "err");
    }
  }

  function setProgress(done, total) {
    const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
    $("progress-fill").style.width = `${pct}%`;
    $("progress-text").textContent = `${done} / ${total}`;
  }

  function setRunning(running) {
    $("btn-start").disabled = running;
    $("btn-stop").disabled = !running;
  }

  function ensureSSE() {
    if (sse) return;
    sse = new EventSource("/api/run/logs");
    sse.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "log" && data.message) {
          log(data.message, logClass(data.message));
        } else if (data.type === "ping") {
          setProgress(data.done || 0, data.total || 0);
          setRunning(!!data.running);
        }
      } catch (_) {}
    };
    sse.onerror = () => {
      // browser will retry; no-op
    };
  }

  function setInspectorOptions(id, options, preferred = "") {
    const select = $(id);
    select.replaceChildren();
    options.forEach(({ value, label }) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
    if (options.some((option) => option.value === preferred)) select.value = preferred;
    select.disabled = options.length === 0;
  }

  async function copyInspectorText(text) {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch (_) {
      const field = document.createElement("textarea");
      field.value = text;
      field.style.position = "fixed";
      field.style.left = "-10000px";
      document.body.appendChild(field);
      field.select();
      const copied = document.execCommand("copy");
      field.remove();
      if (!copied) throw new Error("复制失败，请手动选择文本复制。");
    }
  }

  function invalidatePromptPreview() {
    previewRequest += 1;
    previewText = "";
    $("btn-copy-prompt").disabled = true;
    $("preview-sections").replaceChildren();
    $("preview-images").replaceChildren();
    $("preview-note").textContent = "按当前配置预览，不调用模型。点击生成预览；来源标签不会发送给模型。";
  }

  function fillPreviewScales() {
    setInspectorOptions("preview-scale", scales.map((scale) => ({ value: scale.name, label: scale.name })), $("preview-scale").value);
    $("btn-preview-prompt").disabled = scales.length === 0;
  }

  async function previewPrompt() {
    invalidatePromptPreview();
    const token = previewRequest;
    $("preview-note").textContent = "正在生成预览…";
    try {
      const data = await api("/api/prompt-preview", {
        method: "POST", body: JSON.stringify({
          config: collectConfig(), scale_name: $("preview-scale").value,
          language: $("preview-language").value,
        }),
      });
      if (token !== previewRequest) return;
      previewText = data.prompt || "";
      (data.sections || []).forEach((section) => {
        const label = document.createElement("div");
        label.className = "inspector-label";
        label.textContent = section.label;
        const content = document.createElement("pre");
        content.className = "text-inspector";
        content.textContent = section.text;
        $("preview-sections").append(label, content);
      });
      (data.images || []).forEach((item) => {
        if (!String(item.src || "").startsWith("data:image/")) return;
        const image = document.createElement("img");
        image.src = item.src;
        image.alt = `量表附图 ${item.anchor || ""}`;
        $("preview-images").appendChild(image);
      });
      $("preview-note").textContent = `${data.note || ""} 来源标签不发送给模型。`;
      $("btn-copy-prompt").disabled = !previewText;
    } catch (error) {
      if (token === previewRequest) $("preview-note").textContent = error.message || String(error);
    }
  }

  function answerQuery() {
    return new URLSearchParams({ results_dir: answerResultsDir, run: $("answer-run").value });
  }

  async function showTrialAnswer() {
    const token = ++answerRequest;
    $("answer-text").textContent = "";
    $("btn-copy-answer").disabled = true;
    const row = answerRows.find((item) => item.id === $("answer-trial").value);
    if (!row) {
      $("answer-note").textContent = "没有符合条件的试次。";
      return;
    }
    if (!row.source) {
      $("answer-note").textContent = row.status_label;
      return;
    }
    $("answer-note").textContent = "正在读取完整回答…";
    try {
      const query = answerQuery();
      query.set("source", row.source);
      query.set("kind", row.source_kind);
      const data = await api(`/api/trials/detail?${query}`);
      if (token !== answerRequest) return;
      $("answer-text").textContent = data.raw_response || "";
      $("answer-note").textContent = `${row.experiment} · ${row.provider} / ${row.model} · ${data.status_label} · ${row.answer_length} 字符`;
      $("btn-copy-answer").disabled = !data.raw_response;
    } catch (error) {
      if (token === answerRequest) $("answer-note").textContent = error.message || String(error);
    }
  }

  function filterTrialAnswers() {
    const scale = $("answer-scale").value;
    const languages = [...new Set(answerRows.filter((row) => row.scale_name === scale).map((row) => row.language))];
    setInspectorOptions("answer-language", languages.map((language) => ({ value: language, label: language === "ch" ? "中文" : "英文" })), $("answer-language").value);
    const language = $("answer-language").value;
    const rows = answerRows.filter((row) => row.scale_name === scale && row.language === language);
    setInspectorOptions("answer-trial", rows.map((row) => ({
      value: row.id,
      label: `#${row.order_id} · ${row.model} · ${row.status_label}${$("answer-run").value === "." ? ` · ${row.run_id || "旧结果"}` : ""}`,
    })), $("answer-trial").value);
    showTrialAnswer();
  }

  async function loadTrialAnswers() {
    ++answerRequest;
    answerRows = [];
    $("answer-text").textContent = "";
    $("btn-copy-answer").disabled = true;
    $("btn-export-answers").disabled = true;
    $("answer-note").textContent = "正在加载试次记录…";
    const query = answerQuery();
    const token = answerRequest;
    try {
      const data = await api(`/api/trials?${query}`);
      if (token !== answerRequest) return;
      answerRows = data.trials || [];
      const names = [...new Set(answerRows.map((row) => row.scale_name))];
      setInspectorOptions("answer-scale", names.map((name) => ({ value: name, label: name })), $("answer-scale").value);
      $("btn-export-answers").disabled = !answerRows.length;
      filterTrialAnswers();
    } catch (error) {
      if (token === answerRequest) $("answer-note").textContent = error.message || String(error);
    }
  }

  async function refreshTrialRuns() {
    ++answerRequest;
    const token = answerRequest;
    answerResultsDir = $("results_dir").value.trim() || "./results";
    $("answer-text").textContent = "";
    $("btn-copy-answer").disabled = true;
    $("btn-export-answers").disabled = true;
    $("answer-note").textContent = "正在加载运行记录…";
    try {
      const query = new URLSearchParams({ results_dir: answerResultsDir });
      const data = await api(`/api/trials/runs?${query}`);
      if (token !== answerRequest) return;
      setInspectorOptions("answer-run", (data.runs || []).map((run) => ({ value: run.id, label: run.label })), $("answer-run").value);
      if (!data.runs?.length) {
        answerRows = [];
        setInspectorOptions("answer-scale", []);
        setInspectorOptions("answer-trial", []);
        $("answer-note").textContent = "当前结果目录还没有运行记录。";
        return;
      }
      await loadTrialAnswers();
    } catch (error) {
      if (token === answerRequest) $("answer-note").textContent = error.message || String(error);
    }
  }

  $("prompt-preview-panel").addEventListener("toggle", () => {
    if ($("prompt-preview-panel").open) { fillPreviewScales(); previewPrompt(); }
  });
  $("preview-scale").addEventListener("change", previewPrompt);
  $("preview-language").addEventListener("change", previewPrompt);
  $("btn-preview-prompt").addEventListener("click", previewPrompt);
  $("btn-copy-prompt").addEventListener("click", () => copyInspectorText(previewText).catch((error) => { $("preview-note").textContent = error.message; }));
  $("trial-answers-panel").addEventListener("toggle", () => { if ($("trial-answers-panel").open) refreshTrialRuns(); });
  $("btn-refresh-answers").addEventListener("click", refreshTrialRuns);
  $("answer-run").addEventListener("change", loadTrialAnswers);
  $("answer-scale").addEventListener("change", filterTrialAnswers);
  $("answer-language").addEventListener("change", filterTrialAnswers);
  $("answer-trial").addEventListener("change", showTrialAnswer);
  $("btn-copy-answer").addEventListener("click", () => copyInspectorText($("answer-text").textContent).catch((error) => { $("answer-note").textContent = error.message; }));
  $("btn-export-answers").addEventListener("click", async () => {
    $("btn-export-answers").disabled = true;
    try {
      const response = await fetch("/api/trials/export", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ results_dir: answerResultsDir, run: $("answer-run").value }),
      });
      if (!response.ok) { const data = await response.json(); throw new Error(data.error || "导出失败"); }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "试次完整回答.xlsx";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      $("answer-note").textContent = "完整回答 Excel 已生成，原 JSON 和逐题 CSV 继续保留。";
    } catch (error) {
      $("answer-note").textContent = error.message || String(error);
    } finally {
      $("btn-export-answers").disabled = !answerRows.length;
    }
  });

  // events
  $("provider").addEventListener("change", () => fillGlobalModels());
  $("btn-refresh-models").addEventListener("click", () => fillGlobalModels().then(() => log("模型列表已刷新", "ok")));
  $("btn-reload-scales").addEventListener("click", () => reloadScales().catch((e) => log(e.message, "err")));
  $("btn-refresh-net").addEventListener("click", () => refreshNetwork());

  $("btn-apply-preset").addEventListener("click", async () => {
    if (applyingPreset) return;
    const presetId = $("preset_id").value;
    if (!presetId || presetId === "custom") {
      alert("自定义配置不会被预设覆盖；请直接保存或运行当前配置。");
      return;
    }
    try {
      applyingPreset = true;
      const data = await api("/api/presets/apply", {
        method: "POST",
        body: JSON.stringify({ preset_id: presetId }),
      });
      const preset = data.preset || {};
      const previousActiveId = activePresetId;
      const previousSeed = randomSeed;
      const controls = [
        "temperature", "top_p", "batch_en", "batch_ch", "concurrency", "scale_concurrency",
        "max_retries", "max_group_attempts", "sample_mode", "capture_network",
        "capture_model_catalog", "network_guard", "cultural_identity", "prompt_contract", "condition_mode",
      ].map($).filter(Boolean).concat([...document.querySelectorAll(".en-count, .ch-count")]);
      const previousValues = controls.map((el) => ({ el, value: el.value, checked: el.checked }));
      invalidatePromptPreview();
      fillGenerationConfig(preset.generation_config || {});
      const execution = preset.execution_config || {};
      randomSeed = execution.random_seed ?? null;
      if (execution.batch_en != null) $("batch_en").value = execution.batch_en;
      if (execution.batch_ch != null) $("batch_ch").value = execution.batch_ch;
      if (execution.concurrency != null) $("concurrency").value = execution.concurrency;
      if (execution.scale_concurrency != null) $("scale_concurrency").value = execution.scale_concurrency;
      if (execution.max_retries != null) $("max_retries").value = execution.max_retries;
      if (execution.max_group_attempts != null) $("max_group_attempts").value = execution.max_group_attempts;
      if (execution.sample_mode) $("sample_mode").value = "planned";
      if (execution.capture_network != null) $("capture_network").checked = Boolean(execution.capture_network);
      if (execution.capture_model_catalog != null) $("capture_model_catalog").checked = Boolean(execution.capture_model_catalog);
      if (execution.network_guard) $("network_guard").value = execution.network_guard;
      $("cultural_identity").value = preset.prompt_config?.cultural_identity || "none";
      const presetContract = preset.prompt_config?.contract;
      if (presetContract && $("prompt_contract")) $("prompt_contract").value = presetContract;
      // A formal preset targets the requested per-language sample size. It
      // does not enable/disable scales or change their API/model assignments.
      document.querySelectorAll(".en-count").forEach((el) => (el.value = execution.batch_en ?? el.value));
      document.querySelectorAll(".ch-count").forEach((el) => (el.value = execution.batch_ch ?? el.value));
      const presetSelect = $("preset_id");
      activePresetId = presetId;
      if (preset.condition_mode) $("condition_mode").value = preset.condition_mode;
      presetSelect.value = presetId;
      try {
        const cfg = collectConfig();
        await api("/api/config", { method: "POST", body: JSON.stringify(cfg) });
      } catch (saveError) {
        activePresetId = previousActiveId;
        randomSeed = previousSeed;
        previousValues.forEach(({ el, value, checked }) => { el.value = value; el.checked = checked; });
        presetSelect.value = presetId;
        throw saveError;
      }
      updatePresetButton();
      invalidatePromptPreview();
      log(`已应用并保存预设：${preset.name || presetId}`, "ok");
      (preset.warnings || []).forEach((warning) => log(`预设提示：${warning}`, "info"));
    } catch (e) {
      log(e.message || String(e), "err");
      alert(e.message || String(e));
    } finally {
      applyingPreset = false;
      updatePresetButton();
    }
  });

  async function browseFolder(kind) {
    try {
      const data = await api("/api/pick-folder", {
        method: "POST",
        body: JSON.stringify({ kind }),
      });
      if (!data.path) return;
      if (kind === "results") {
        $("results_dir").value = data.path;
        log(`已选择结果目录: ${data.path}`, "ok");
      } else {
        $("data_dir").value = data.path;
        log(`已选择数据目录: ${data.path}`, "ok");
      }
      // 选完立即存盘：仅保留所选文件夹内的文件，不延续旧目录/旧量表
      await api("/api/config", { method: "POST", body: JSON.stringify(collectConfig()) });
      if (kind === "data") await reloadScales();
    } catch (e) {
      if (String(e.message || e).includes("取消")) return;
      log(e.message || String(e), "err");
      alert(e.message || String(e));
    }
  }
  $("btn-browse-data").addEventListener("click", () => browseFolder("data"));
  $("btn-browse-results").addEventListener("click", () => browseFolder("results"));
  $("btn-apply-counts").addEventListener("click", () => {
    const en = Number($("batch_en").value || 0);
    const ch = Number($("batch_ch").value || 0);
    document.querySelectorAll(".en-count").forEach((el) => (el.value = en));
    document.querySelectorAll(".ch-count").forEach((el) => (el.value = ch));
    updatePresetButton();
    log(`已批量设置次数：英文=${en}，中文=${ch}`, "ok");
  });
  $("btn-select-all").addEventListener("click", () => setAllEnabled(true));
  $("btn-select-none").addEventListener("click", () => setAllEnabled(false));
  $("select-all-scales").addEventListener("change", (event) => setAllEnabled(event.target.checked));

  $("btn-apply-api-all").addEventListener("click", async () => {
    const provider = $("provider").value;
    const model = $("model").value;
    if (!model) return alert("请先选择默认模型");
    const models = await loadModels(provider, true);
    const fillPair = (pSel, mSel) => {
      pSel.value = provider;
      mSel.innerHTML = modelOptions(models, model);
      mSel.value = model;
    };
    document.querySelectorAll("tr[data-idx]").forEach((row) => {
      fillPair(row.querySelector(".en-provider"), row.querySelector(".en-model"));
      fillPair(row.querySelector(".ch-provider"), row.querySelector(".ch-model"));
    });
    log(`已将默认 API 应用到全部量表的中英文：${provider} / ${model}`, "ok");
  });
  $("btn-save-config").addEventListener("click", async () => {
    try {
      await api("/api/config", { method: "POST", body: JSON.stringify(collectConfig()) });
      log("配置已保存", "ok");
    } catch (e) {
      log(e.message, "err");
    }
  });

  $("btn-start").addEventListener("click", async () => {
    try {
      ensureSSE();
      setRunning(true);
      const cfg = collectConfig();
      await api("/api/run/start", { method: "POST", body: JSON.stringify(cfg) });
      log("已发送启动请求", "info");
    } catch (e) {
      setRunning(false);
      alert(e.message);
      log(e.message, "err");
    }
  });
  $("btn-stop").addEventListener("click", async () => {
    try {
      await api("/api/run/stop", { method: "POST", body: "{}" });
    } catch (e) {
      log(e.message, "err");
    }
  });

  $("btn-open-results").addEventListener("click", async () => {
    await api("/api/config", { method: "POST", body: JSON.stringify(collectConfig()) });
    await api("/api/open-folder", { method: "POST", body: JSON.stringify({ kind: "results" }) });
  });
  $("btn-open-logs").addEventListener("click", async () => {
    await api("/api/open-folder", { method: "POST", body: JSON.stringify({ kind: "logs" }) });
  });

  // keys modal
  $("btn-keys").addEventListener("click", async () => {
    const keys = await api("/api/keys");
    $("key-deepseek").value = keys.deepseek?.api_key || "";
    $("url-deepseek").value = keys.deepseek?.base_url || "";
    $("key-openai").value = keys.openai?.api_key || "";
    $("url-openai").value = keys.openai?.base_url || "";
    $("key-qwen").value = keys.qwen?.api_key || "";
    $("url-qwen").value = keys.qwen?.base_url || "";
    $("key-gemini").value = keys.gemini?.api_key || "";
    $("url-gemini").value = keys.gemini?.base_url || "";
    $("keys-modal").classList.add("open");
  });
  $("btn-keys-cancel").addEventListener("click", () => $("keys-modal").classList.remove("open"));
  $("btn-keys-save").addEventListener("click", async () => {
    await api("/api/keys", {
      method: "POST",
      body: JSON.stringify({
        deepseek: { api_key: $("key-deepseek").value, base_url: $("url-deepseek").value },
        openai: { api_key: $("key-openai").value, base_url: $("url-openai").value },
        qwen: { api_key: $("key-qwen").value, base_url: $("url-qwen").value },
        gemini: { api_key: $("key-gemini").value, base_url: $("url-gemini").value },
      }),
    });
    $("keys-modal").classList.remove("open");
    log("API 密钥已保存", "ok");
    await fillGlobalModels();
  });

  // boot
  bindPresetDirtyTracking();
  ensureSSE();
  loadConfig()
    .then(() => refreshNetwork())
    .catch((e) => log(e.message, "err"));
})();
