(() => {
  const $ = (id) => document.getElementById(id);
  const modelCache = {};
  const history = [];
  let abortCtrl = null;
  let streaming = false;

  async function api(url, options) {
    const resp = await fetch(url, {
      headers: { "Content-Type": "application/json", ...(options?.headers || {}) },
      ...options,
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
    return data;
  }

  async function loadModels(provider, force = false) {
    if (!force && modelCache[provider]?.length) return modelCache[provider];
    const data = await api(`/api/models?provider=${encodeURIComponent(provider)}`);
    modelCache[provider] = data.models || [];
    return modelCache[provider];
  }

  async function fillModels(force = false) {
    const provider = $("provider").value;
    const sel = $("model");
    const cur = sel.value;
    sel.innerHTML = "";
    const loading = document.createElement("option");
    loading.textContent = "加载中…";
    loading.disabled = true;
    sel.appendChild(loading);
    try {
      const models = await loadModels(provider, force);
      if ($("provider").value !== provider) return;
      sel.innerHTML = "";
      models.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        sel.appendChild(opt);
      });
      if (cur && models.includes(cur)) sel.value = cur;
      else if (models.length) sel.value = models[0];
    } catch (e) {
      if ($("provider").value !== provider) return;
      sel.innerHTML = "";
      console.error(e);
    }
  }

  function optionalNumber(id) {
    const raw = String($(id)?.value ?? "").trim();
    if (!raw) return null;
    const value = Number(raw);
    if (!Number.isFinite(value)) throw new Error(`${id} 必须是数字`);
    return value;
  }

  function collectGenerationConfig() {
    return {
      temperature: optionalNumber("temperature") ?? 0.7,
      top_p: optionalNumber("top_p") ?? 1,
      thinking_mode: "disabled",
      stream: true,
    };
  }

  function appendBubble(role, text) {
    const box = $("messages");
    const el = document.createElement("div");
    el.className = `chat-bubble ${role}`;
    const meta = document.createElement("div");
    meta.className = "chat-meta";
    meta.textContent = role === "user" ? "你" : "助手";
    const body = document.createElement("div");
    body.className = "chat-text";
    body.textContent = text || "";
    el.appendChild(meta);
    el.appendChild(body);
    box.appendChild(el);
    box.scrollTop = box.scrollHeight;
    return body;
  }

  function setStreaming(on) {
    streaming = on;
    $("btn-send").disabled = on;
    $("btn-stop").disabled = !on;
    $("input").disabled = on;
  }

  async function send() {
    if (streaming) return;
    const text = $("input").value.trim();
    if (!text) return;
    const provider = $("provider").value;
    const model = $("model").value;
    if (!model) {
      alert("请先选择模型");
      return;
    }

    $("input").value = "";
    history.push({ role: "user", content: text });
    appendBubble("user", text);
    const assistantEl = appendBubble("assistant", "");
    assistantEl.classList.add("streaming");

    const temperature = Number($("temperature").value || 0.7);
    abortCtrl = new AbortController();
    setStreaming(true);

    let full = "";
    try {
      const resp = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          model,
          temperature,
          generation_config: collectGenerationConfig(),
          messages: history,
        }),
        signal: abortCtrl.signal,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || `HTTP ${resp.status}`);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const lines = part.split("\n");
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw) continue;
            let ev;
            try {
              ev = JSON.parse(raw);
            } catch {
              continue;
            }
            if (ev.type === "delta" && ev.content) {
              full += ev.content;
              assistantEl.textContent = full;
              $("messages").scrollTop = $("messages").scrollHeight;
            } else if (ev.type === "error") {
              throw new Error(ev.message || "流式调用失败");
            }
          }
        }
      }
      if (!full) full = "（无内容返回）";
      assistantEl.textContent = full;
      history.push({ role: "assistant", content: full });
    } catch (err) {
      if (err.name === "AbortError") {
        const note = full ? `${full}\n\n[已停止]` : "[已停止]";
        assistantEl.textContent = note;
        if (full) history.push({ role: "assistant", content: full });
      } else {
        assistantEl.textContent = `错误: ${err.message || err}`;
        assistantEl.classList.add("err");
        history.pop();
      }
    } finally {
      assistantEl.classList.remove("streaming");
      setStreaming(false);
      abortCtrl = null;
    }
  }

  function clearChat() {
    if (streaming) return;
    history.length = 0;
    $("messages").innerHTML = "";
  }

  $("provider").addEventListener("change", () => fillModels(true));
  $("btn-refresh-models").addEventListener("click", () => fillModels(true));
  $("btn-clear").addEventListener("click", clearChat);
  $("btn-send").addEventListener("click", send);
  $("btn-stop").addEventListener("click", () => {
    if (abortCtrl) abortCtrl.abort();
  });
  $("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  api("/api/config")
    .then((cfg) => {
      if (cfg.provider) $("provider").value = cfg.provider;
      if (cfg.temperature != null) $("temperature").value = cfg.temperature;
      if (cfg.generation_config?.temperature != null) $("temperature").value = cfg.generation_config.temperature;
      $("top_p").value = cfg.generation_config?.top_p ?? 1;
      return fillModels(true).then(() => {
        if (cfg.model) {
          const sel = $("model");
          if ([...sel.options].some((o) => o.value === cfg.model)) sel.value = cfg.model;
        }
      });
    })
    .catch(() => fillModels(true));
})();
