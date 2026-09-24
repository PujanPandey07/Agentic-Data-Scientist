import { useEffect, useState } from "react";
import axiosInstance from "../api/axiosstance";

const PROVIDER_LABELS = {
  openai: "OpenAI",
  anthropic: "Anthropic",
  gemini: "Gemini",
};

function ApiKeySettings() {
  const [availableModels, setAvailableModels] = useState({});
  const [status, setStatus] = useState(null); // {configured, provider, model_name}
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // Form state for "add/replace key" (always provider + key + model together)
  const [provider, setProvider] = useState("openai");
  const [apiKey, setApiKey] = useState("");
  const [modelName, setModelName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Form state for "change model only" (same provider, no key needed)
  const [newModel, setNewModel] = useState("");
  const [changingModel, setChangingModel] = useState(false);

  const [deleting, setDeleting] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setErrorMsg("");
      try {
        const [modelsRes, statusRes] = await Promise.all([
          axiosInstance.get("/api/api-keys/available-models"),
          axiosInstance.get("/api/api-keys"),
        ]);
        setAvailableModels(modelsRes.data);
        setStatus(statusRes.data);
        if (statusRes.data.configured) {
          setNewModel(statusRes.data.model_name);
        }
      } catch (error) {
        setErrorMsg(error.response?.data?.detail || "Could not load API key settings.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Keep the "add/replace" model dropdown valid whenever provider changes —
  // each provider has a different model list, so a stale selection from the
  // previous provider would otherwise submit an invalid model_name.
  useEffect(() => {
    const models = availableModels[provider] || [];
    if (models.length > 0 && !models.includes(modelName)) {
      setModelName(models[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, availableModels]);

  async function handleSubmitKey(e) {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");
    if (!apiKey.trim()) {
      setErrorMsg("Please enter your API key.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await axiosInstance.put("/api/api-keys", {
        provider,
        api_key: apiKey.trim(),
        model_name: modelName,
      });
      setStatus(res.data);
      setNewModel(res.data.model_name);
      setApiKey(""); // never keep the raw key in memory longer than needed
      setSuccessMsg("Your API key has been saved.");
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Could not save API key.");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleChangeModel(e) {
    e.preventDefault();
    setErrorMsg("");
    setSuccessMsg("");
    setChangingModel(true);
    try {
      const res = await axiosInstance.patch("/api/api-keys/model", {
        model_name: newModel,
      });
      setStatus(res.data);
      setSuccessMsg("Model updated.");
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Could not update model.");
    } finally {
      setChangingModel(false);
    }
  }

  async function handleDelete() {
    setErrorMsg("");
    setSuccessMsg("");
    setDeleting(true);
    try {
      await axiosInstance.delete("/api/api-keys");
      setStatus({ configured: false });
      setConfirmingDelete(false);
      setSuccessMsg("Removed — you're back on the shared default model.");
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Could not remove API key.");
    } finally {
      setDeleting(false);
    }
  }

  if (loading) {
    return <div className="max-w-xl mx-auto px-6 py-12 text-muted">Loading...</div>;
  }

  const providerOptions = Object.keys(availableModels);
  const modelsForProvider = availableModels[provider] || [];

  return (
    <div className="max-w-xl mx-auto px-6 py-12">
      <h1 className="font-serif text-2xl text-ink mb-2">Your AI model</h1>
      <p className="text-muted text-sm mb-6">
        By default, everyone shares a limited free model. Add your own API key
        for OpenAI, Anthropic, or Gemini to avoid shared rate limits — it's
        used only for your own analyses, and your key is never shown again
        after saving.
      </p>

      {errorMsg && (
        <p className="text-clay text-sm mb-4 bg-clay/10 border border-clay/30 rounded-md px-3 py-2">
          {errorMsg}
        </p>
      )}
      {successMsg && (
        <p className="text-ink text-sm mb-4 bg-accent/10 border border-accent/30 rounded-md px-3 py-2">
          {successMsg}
        </p>
      )}

      {status?.configured ? (
        <div className="bg-white border border-muted/20 rounded-lg p-5 mb-6">
          <p className="text-sm text-muted mb-1">Currently using</p>
          <p className="text-ink font-medium mb-4">
            {PROVIDER_LABELS[status.provider] || status.provider} — {status.model_name}
          </p>

          <form onSubmit={handleChangeModel} className="flex gap-2 mb-4">
            <select
              value={newModel}
              onChange={(e) => setNewModel(e.target.value)}
              className="flex-1 px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            >
              {(availableModels[status.provider] || []).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button
              type="submit"
              disabled={changingModel || newModel === status.model_name}
              className="bg-ink text-paper px-4 py-2 rounded-md text-sm font-medium hover:bg-panel transition-colors disabled:opacity-50"
            >
              {changingModel ? "Updating..." : "Switch model"}
            </button>
          </form>
          <p className="text-xs text-muted mb-4">
            Switching models here stays on {PROVIDER_LABELS[status.provider] || status.provider} — no need to re-enter your key.
          </p>

          {confirmingDelete ? (
            <div className="flex items-center gap-3">
              <p className="text-sm text-clay">Remove this key and go back to the shared default?</p>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="bg-clay text-white px-3 py-1.5 rounded-md text-xs font-medium disabled:opacity-50"
              >
                {deleting ? "Removing..." : "Yes, remove"}
              </button>
              <button
                onClick={() => setConfirmingDelete(false)}
                className="text-xs text-muted hover:text-ink"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmingDelete(true)}
              className="text-xs text-clay hover:underline"
            >
              Remove key
            </button>
          )}
        </div>
      ) : (
        <p className="text-sm text-muted mb-6">
          You're currently using the shared default model. Add your own key below.
        </p>
      )}

      <div className="bg-white border border-muted/20 rounded-lg p-5">
        <p className="font-medium text-ink mb-4">
          {status?.configured ? "Replace with a different provider" : "Add your API key"}
        </p>
        <form onSubmit={handleSubmitKey} className="space-y-3">
          <div>
            <label className="block text-xs text-muted mb-1">Provider</label>
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            >
              {providerOptions.map((p) => (
                <option key={p} value={p}>{PROVIDER_LABELS[p] || p}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-muted mb-1">Model</label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            >
              {modelsForProvider.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-xs text-muted mb-1">API key</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              autoComplete="off"
              className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <p className="text-xs text-muted mt-1">
              Stored encrypted. It's write-only — never shown again after saving.
            </p>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full bg-ink text-paper py-2 rounded-md font-medium hover:bg-panel transition-colors disabled:opacity-50"
          >
            {submitting ? "Saving..." : status?.configured ? "Replace key" : "Save key"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default ApiKeySettings;