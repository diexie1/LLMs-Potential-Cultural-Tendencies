(() => {
  const $ = (id) => document.getElementById(id);
  const providers = ["deepseek", "openai", "qwen", "gemini"];
  const modelCache = {};
  let scales = [];
  let sse = null;

  function log(msg, cls = "info") {
    const box = $("log-box");
    const line = document.createElement("div");
    line.className = cls;
    line.textContent = msg;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  function logClass(msg) {
    if (msg.includes("[成功") || msg.includes("完成") || msg.includes("已加载")) return "ok";
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
    if (!res.ok) throw new Error(data.error || res.statusText || "请求失败");
    return data;
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
      };
    });
    return vars;
  }

  function collectConfig() {
    return {
      data_dir: $("data_dir").value.trim() || "./data",
      results_dir: $("results_dir").value.trim() || "./results",
      provider: $("provider").value,
      model: $("model").value,
      temperature: Number($("temperature").value || 0.7),
      concurrency: Number($("concurrency").value || 5),
      batch_en: Number($("batch_en").value || 0),
      batch_ch: Number($("batch_ch").value || 0),
      scale_vars: collectScaleVars(),
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
      modelSel.innerHTML = modelOptions(list, prev);
    } catch (e) {
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

      tr.innerHTML = `
        <td class="check"><input class="row-enabled" type="checkbox" ${s.enabled ? "checked" : ""} /></td>
        <td class="name">${escapeHtml(s.name)}</td>
        <td><select class="en-provider">${providerOptions(enP)}</select></td>
        <td><select class="en-model">${modelOptions(enModels, s.en_model)}</select></td>
        <td><select class="ch-provider">${providerOptions(chP)}</select></td>
        <td><select class="ch-model">${modelOptions(chModels, s.ch_model)}</select></td>
        <td>${s.en_n}</td>
        <td>${s.ch_n}</td>
        <td><input class="en-count" type="number" min="0" step="1" value="${Number(s.en_count) || 0}" style="width:78px" /></td>
        <td><input class="ch-count" type="number" min="0" step="1" value="${Number(s.ch_count) || 0}" style="width:78px" /></td>
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
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function reloadScales() {
    const dataDir = $("data_dir").value.trim() || "./data";
    const data = await api(`/api/scales?data_dir=${encodeURIComponent(dataDir)}`);
    await renderScales(data.scales || []);
    log(`已加载 ${(data.scales || []).length} 个量表（${data.resolved}）`, "ok");
  }

  async function loadConfig() {
    const cfg = await api("/api/config");
    $("data_dir").value = cfg.data_dir || "./data";
    $("results_dir").value = cfg.results_dir || "./results";
    $("provider").value = cfg.provider || "deepseek";
    $("temperature").value = cfg.temperature ?? 0.7;
    $("concurrency").value = cfg.concurrency ?? 5;
    $("batch_en").value = cfg.batch_en ?? 100;
    $("batch_ch").value = cfg.batch_ch ?? 100;
    await fillGlobalModels();
    if (cfg.model) {
      const sel = $("model");
      if ([...sel.options].some((o) => o.value === cfg.model)) sel.value = cfg.model;
    }
    await reloadScales();
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

  // events
  $("provider").addEventListener("change", () => fillGlobalModels());
  $("btn-refresh-models").addEventListener("click", () => fillGlobalModels().then(() => log("模型列表已刷新", "ok")));
  $("btn-reload-scales").addEventListener("click", () => reloadScales().catch((e) => log(e.message, "err")));
  $("btn-refresh-net").addEventListener("click", () => refreshNetwork());

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
        await reloadScales();
      }
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
    log(`已批量设置次数：英文=${en}，中文=${ch}`, "ok");
  });
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
  ensureSSE();
  loadConfig()
    .then(() => refreshNetwork())
    .catch((e) => log(e.message, "err"));
})();
