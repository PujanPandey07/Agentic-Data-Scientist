import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosstance";

const DEFAULT_MODELS = {
  gemini: ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
  openai: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
  anthropic: ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
  groq: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
};

const PROVIDER_INFO = {
  gemini: {
    label: "Google Gemini",
    description: "Recommended for high-speed agentic execution with generous quotas.",
    keyPlaceholder: "AIzaSy...",
  },
  openai: {
    label: "OpenAI",
    description: "Industry-standard reasoning with GPT-4o models.",
    keyPlaceholder: "sk-proj-...",
  },
  anthropic: {
    label: "Anthropic Claude",
    description: "High capability analysis with Claude 3.5 Sonnet.",
    keyPlaceholder: "sk-ant-...",
  },
  groq: {
    label: "Groq",
    description: "Ultra-fast inference for open-source Llama & Mixtral models.",
    keyPlaceholder: "gsk_...",
  },
};

export default function ModelSettings() {
  const navigate = useNavigate();
  const [status, setStatus] = useState({ configured: false, provider: null, model_name: null });
  const [availableModels, setAvailableModels] = useState(DEFAULT_MODELS);
  const [selectedProvider, setSelectedProvider] = useState("gemini");
  const [selectedModel, setSelectedModel] = useState("gemini-2.0-flash");
  const [apiKey, setApiKey] = useState("");
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [feedback, setFeedback] = useState({ type: null, message: null });

  useEffect(() => {
    async function loadData() {
      try {
        const [modelsRes, statusRes] = await Promise.all([
          axiosInstance.get("/api/api-keys/available-models").catch(() => ({ data: DEFAULT_MODELS })),
          axiosInstance.get("/api/api-keys").catch(() => ({ data: { configured: false } })),
        ]);

        const models = modelsRes.data || DEFAULT_MODELS;
        setAvailableModels(models);

        if (statusRes.data?.configured) {
          setStatus(statusRes.data);
          setSelectedProvider(statusRes.data.provider || "gemini");
          setSelectedModel(statusRes.data.model_name || (models[statusRes.data.provider] || [])[0]);
        } else {
          setSelectedModel(models["gemini"]?.[0] || "gemini-2.0-flash");
        }
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  function handleProviderChange(provider) {
    setSelectedProvider(provider);
    const models = availableModels[provider] || DEFAULT_MODELS[provider] || [];
    setSelectedModel(models[0] || "");
    setFeedback({ type: null, message: null });
  }

  async function handleSaveKey(e) {
    e.preventDefault();
    if (!apiKey.trim()) {
      setFeedback({ type: "error", message: "Please enter your API key." });
      return;
    }

    setSaving(true);
    setFeedback({ type: null, message: null });

    try {
      const res = await axiosInstance.put("/api/api-keys", {
        provider: selectedProvider,
        api_key: apiKey.trim(),
        model_name: selectedModel,
      });
      setStatus(res.data);
      setApiKey("");
      setFeedback({
        type: "success",
        message: `Successfully configured ${PROVIDER_INFO[selectedProvider]?.label} (${selectedModel}).`,
      });
    } catch (err) {
      setFeedback({
        type: "error",
        message: err.response?.data?.detail || "Failed to save API key. Please check your inputs.",
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleModelOnlySwitch(newModel) {
    if (!status.configured || status.provider !== selectedProvider) return;
    setSaving(true);
    setFeedback({ type: null, message: null });

    try {
      const res = await axiosInstance.patch("/api/api-keys/model", { model_name: newModel });
      setStatus(res.data);
      setSelectedModel(newModel);
      setFeedback({
        type: "success",
        message: `Model updated to ${newModel} (using existing ${status.provider} key).`,
      });
    } catch (err) {
      setFeedback({
        type: "error",
        message: err.response?.data?.detail || "Failed to switch model.",
      });
    } finally {
      setSaving(false);
    }
  }

  async function handleResetDefault() {
    if (!window.confirm("Remove your custom API key and revert to the default system model?")) return;
    setSaving(true);
    setFeedback({ type: null, message: null });

    try {
      await axiosInstance.delete("/api/api-keys");
      setStatus({ configured: false, provider: null, model_name: null });
      setFeedback({
        type: "success",
        message: "Reverted to shared system default model.",
      });
    } catch (err) {
      setFeedback({
        type: "error",
        message: err.response?.data?.detail || "Failed to reset model.",
      });
    } finally {
      setSaving(false);
    }
  }

  const currentModels = availableModels[selectedProvider] || DEFAULT_MODELS[selectedProvider] || [];

  return (
    <div className="h-full overflow-y-auto bg-paper px-6 py-10">
      <div className="max-w-2xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <button
              onClick={() => navigate(-1)}
              className="text-sm text-muted hover:text-ink mb-2 inline-flex items-center gap-1 font-medium"
            >
              ← Back
            </button>
            <h1 className="font-serif text-2xl text-ink">LLM Model Settings</h1>
            <p className="text-sm text-muted">
              Bring your own API key to use your preferred model provider, unlock custom rate limits, or seamlessly switch if tokens expire.
            </p>
          </div>
        </div>

        {/* Feedback Banner */}
        {feedback.message && (
          <div
            className={`p-4 rounded-lg text-sm border ${
              feedback.type === "success"
                ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                : "bg-red-50 border-red-200 text-red-800"
            }`}
          >
            {feedback.message}
          </div>
        )}

        {/* Current Active Status Card */}
        <div className="bg-white border border-muted/30 rounded-xl p-5 shadow-sm space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-xs uppercase tracking-wider font-semibold text-muted">Current Active Configuration</h2>
            {status.configured ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-600 animate-pulse" />
                Custom Key Active
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-panel/10 text-muted">
                Shared Default Model
              </span>
            )}
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-2">
            <div>
              <p className="text-lg font-serif text-ink">
                {status.configured ? (
                  <>
                    <span className="capitalize">{PROVIDER_INFO[status.provider]?.label || status.provider}</span>
                    <span className="text-muted font-sans text-sm ml-2">({status.model_name})</span>
                  </>
                ) : (
                  <>Google Gemini <span className="text-muted font-sans text-sm ml-2">(gemini-2.0-flash default)</span></>
                )}
              </p>
              <p className="text-xs text-muted mt-0.5">
                {status.configured
                  ? "All data science tasks in your chats will execute using your stored API key."
                  : "Using the application's built-in model quotas."}
              </p>
            </div>

            {status.configured && (
              <button
                type="button"
                onClick={handleResetDefault}
                disabled={saving}
                className="text-xs font-medium text-clay hover:underline disabled:opacity-50"
              >
                Reset to Default
              </button>
            )}
          </div>
        </div>

        {/* Configuration Form */}
        <form onSubmit={handleSaveKey} className="bg-white border border-muted/30 rounded-xl p-6 shadow-sm space-y-6">
          <h2 className="font-serif text-lg text-ink">Configure / Update Provider</h2>

          {/* Provider Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-ink">1. Select Provider</label>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
              {Object.keys(PROVIDER_INFO).map((pKey) => {
                const isSelected = selectedProvider === pKey;
                return (
                  <button
                    key={pKey}
                    type="button"
                    onClick={() => handleProviderChange(pKey)}
                    className={`px-3 py-3 rounded-lg border text-left transition-all ${
                      isSelected
                        ? "border-accent bg-accent/10 text-ink shadow-xs"
                        : "border-muted/30 hover:border-muted/60 text-muted"
                    }`}
                  >
                    <p className="text-xs font-semibold">{PROVIDER_INFO[pKey].label}</p>
                  </button>
                );
              })}
            </div>
            <p className="text-xs text-muted pt-1">{PROVIDER_INFO[selectedProvider]?.description}</p>
          </div>

          {/* Model Selection */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label htmlFor="modelSelect" className="text-sm font-medium text-ink">2. Select Model</label>
              {status.configured && status.provider === selectedProvider && selectedModel !== status.model_name && (
                <button
                  type="button"
                  onClick={() => handleModelOnlySwitch(selectedModel)}
                  disabled={saving}
                  className="text-xs text-accent font-medium hover:underline"
                >
                  Quick switch model without re-entering key →
                </button>
              )}
            </div>
            <select
              id="modelSelect"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full bg-paper border border-muted/30 rounded-lg px-3 py-2 text-sm text-ink focus:outline-none focus:border-accent"
            >
              {currentModels.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          {/* API Key Input */}
          <div className="space-y-2">
            <label htmlFor="apiKeyInput" className="text-sm font-medium text-ink">3. Enter API Key</label>
            <div className="relative">
              <input
                id="apiKeyInput"
                type={showKey ? "text" : "password"}
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder={PROVIDER_INFO[selectedProvider]?.keyPlaceholder || "Enter API Key"}
                className="w-full bg-paper border border-muted/30 rounded-lg px-3 py-2 pr-16 text-sm text-ink focus:outline-none focus:border-accent font-mono"
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted hover:text-ink font-sans"
              >
                {showKey ? "Hide" : "Show"}
              </button>
            </div>
            <p className="text-xs text-muted">
              🔒 Your key is encrypted via AES-256 before storage and is never exposed in responses or logs.
            </p>
          </div>

          <div className="pt-2">
            <button
              type="submit"
              disabled={saving}
              className="w-full bg-ink text-paper py-2.5 rounded-lg text-sm font-medium hover:bg-ink/90 transition-colors disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save and Activate Model"}
            </button>
          </div>
        </form>

        {/* Resuming After Token Limit Guide */}
        <div className="bg-panel/5 border border-panel/20 rounded-xl p-5 text-sm space-y-2">
          <p className="font-medium text-ink flex items-center gap-1.5">
            <span>💡</span> Switching Models Mid-Task
          </p>
          <p className="text-xs text-muted leading-relaxed">
            If your run stops because your model ran out of tokens or hit a rate limit, simply switch your model or provider here and save.
            Then navigate back to your workspace chat and type <strong className="text-ink">"continue"</strong> or <strong className="text-ink">"retry"</strong>.
            The pipeline will automatically resume from the exact step where it stopped, preserving all previous cleaning, EDA, and feature engineering steps without repeating them.
          </p>
        </div>
      </div>
    </div>
  );
}
