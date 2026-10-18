"use strict";
const byId = (id) => document.getElementById(id);
let lastText = "";
let modelReady = false;
let busy = false;
const descriptions = {
  rnn: "A simple recurrent state: the baseline for next-word prediction.",
  birnn: "Reads the past context in both directions. The next word stays hidden.",
  lstm: "Input, forget, and output gates control a separate memory state.",
  gru: "Gated memory with a compact recurrent state."
};
function updateModelDescription() { byId("model-description").textContent = descriptions[byId("model").value]; }
function showError(message) { byId("error").textContent = message; byId("error").hidden = !message; }
function cell(text) { const element = document.createElement("td"); element.textContent = text; return element; }
async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Could not load the experiment. Refresh to try again.");
    const data = await response.json();
    modelReady = data.ready;
    byId("connection").classList.toggle("ready", data.ready);
    byId("connection").lastChild.textContent = data.ready ? `${data.models.length} models available · local` : "No trained models yet";
    byId("setup").hidden = data.ready;
    byId("generate").disabled = !data.ready || busy;
    for (const option of byId("model").options) option.disabled = !data.models.includes(option.value);
    if (data.ready && !data.models.includes(byId("model").value)) byId("model").value = data.recommended_model || data.models[0];
    updateModelDescription();
    byId("run-detail").textContent = `Run: ${data.run_name} · Status: ${data.status.replaceAll("_", " ")}${data.device ? ` · Trained on ${data.device.toUpperCase()}` : ""}`;
    byId("stats").replaceChildren();
    if (data.dataset) {
      const entries = [["Corpus tokens", data.dataset.total_tokens.toLocaleString()], ["Vocabulary", data.dataset.vocabulary_size.toLocaleString()], ["Context window", `${data.dataset.context_length} tokens`], ["Train / Val / Test", "80 / 10 / 10"]];
      for (const [label, value] of entries) {
        const box = document.createElement("div"); box.className = "stat";
        const title = document.createElement("span"); title.textContent = label;
        const number = document.createElement("strong"); number.textContent = value;
        box.append(title, number); byId("stats").append(box);
      }
    }
    byId("comparison-body").replaceChildren();
    for (const row of data.comparison) {
      const tr = document.createElement("tr");
      if (row.model === data.recommended_model) tr.className = "recommended";
      const name = data.model_names[row.model] + (row.model === data.recommended_model ? " · selected" : "");
      tr.append(...[name, row.best_epoch, row.validation_loss.toFixed(3), row.test_perplexity.toFixed(1), `${(100 * row.test_accuracy).toFixed(1)}%`, `${row.training_seconds.toFixed(1)}s`].map(cell));
      byId("comparison-body").append(tr);
    }
    byId("no-metrics").hidden = data.comparison.length > 0;
    byId("plots").hidden = !data.plots_ready;
    if (data.plots_ready) { byId("loss-plot").src = "/api/plots/training_curves.png"; byId("temperature-plot").src = "/api/plots/temperature_effects.png"; }
  } catch (error) { showError(error.message); byId("connection").lastChild.textContent = "Connection unavailable"; }
}
document.querySelectorAll("[data-view]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-view]").forEach((item) => { item.classList.toggle("active", item === button); item.setAttribute("aria-pressed", String(item === button)); });
  for (const view of ["studio", "experiments"]) byId(view).hidden = view !== button.dataset.view;
}));
document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { byId("prompt").value = button.dataset.prompt; byId("prompt").focus(); }));
byId("model").addEventListener("change", updateModelDescription);
byId("temperature").addEventListener("input", () => { byId("temperature-value").textContent = Number(byId("temperature").value).toFixed(1); });
byId("refresh").addEventListener("click", loadStatus);
byId("generation-form").addEventListener("submit", async (event) => {
  event.preventDefault(); if (busy || !modelReady) return;
  busy = true; showError(""); byId("generate").disabled = true; byId("generate").textContent = "Writing the next lines…";
  try {
    const payload = { prompt: byId("prompt").value, model: byId("model").value, new_tokens: Number(byId("new-tokens").value), temperature: Number(byId("temperature").value), top_k: Number(byId("top-k").value), top_p: Number(byId("top-p").value), seed: Number(byId("seed").value), greedy: byId("greedy").checked };
    const response = await fetch("/api/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const result = await response.json(); if (!response.ok) throw new Error(result.error || "Generation failed. Try again.");
    const prompt = document.createElement("span"); prompt.className = "seed-text"; prompt.textContent = result.prompt;
    byId("generated-text").replaceChildren(prompt, document.createTextNode(result.text.slice(result.prompt.length)));
    lastText = result.text; byId("empty-output").hidden = true; byId("result").hidden = false;
    byId("output-meta").textContent = `${byId("model").selectedOptions[0].text} / ${payload.greedy ? "GREEDY" : `T ${payload.temperature.toFixed(1)}`}`;
    byId("output-count").textContent = `${result.generated_tokens} new tokens · seed ${result.seed}`;
    byId("unknown-words").textContent = result.unknown_prompt_tokens.length ? `Outside the training vocabulary: ${result.unknown_prompt_tokens.join(", ")}. These were encoded as unknown words.` : "";
    byId("copy").disabled = false; byId("download").disabled = false;
  } catch (error) { showError(error.message); }
  finally { busy = false; byId("generate").disabled = !modelReady; byId("generate").textContent = "Continue the story ↗"; }
});
byId("copy").addEventListener("click", async () => { try { await navigator.clipboard.writeText(lastText); byId("copy").textContent = "Copied"; setTimeout(() => { byId("copy").textContent = "Copy text"; }, 1600); } catch { showError("Copy is unavailable in this browser. Use Save .txt instead."); } });
byId("download").addEventListener("click", () => { const url = URL.createObjectURL(new Blob([lastText], { type: "text/plain;charset=utf-8" })); const link = document.createElement("a"); link.href = url; link.download = "baker-street-story.txt"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); });
loadStatus();
