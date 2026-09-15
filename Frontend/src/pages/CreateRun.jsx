import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosstance";

function CreateRunPage() {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const uploadResponse = await axiosInstance.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const datasetId = uploadResponse.data.dataset_id;

      const runResponse = await axiosInstance.post("/api/runs", {
        dataset_id: datasetId,
        user_query: query,
      });

      navigate(`/chat/${runResponse.data.thread_id}`, {
        state: { runResult: runResponse.data, userQuery: query },
      });
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl mx-auto px-6 py-12">
      <h1 className="font-serif text-2xl text-ink mb-6">New Run</h1>
      <form onSubmit={handleSubmit} className="space-y-5">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            dragActive ? "border-accent bg-accent/5" : "border-muted/40"
          }`}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <input
            id="fileInput"
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files[0])}
            className="hidden"
          />
          {file ? (
            <p className="text-ink text-sm font-medium">{file.name}</p>
          ) : (
            <>
              <p className="text-ink text-sm font-medium mb-1">
                Drop your CSV here, or click to browse
              </p>
              <p className="text-muted text-xs">CSV files only</p>
            </>
          )}
        </div>

        <div>
          <label className="block text-sm text-ink mb-1">What do you want to do?</label>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink focus:outline-none focus:ring-2 focus:ring-accent"
            rows={3}
            required
          />
        </div>

        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}

        <button
          type="submit"
          disabled={loading || !file}
          className="w-full bg-ink text-paper py-2 rounded-md font-medium hover:bg-panel transition-colors disabled:opacity-50"
        >
          {loading ? "Starting..." : "Start Run"}
        </button>
      </form>
    </div>
  );
}

export default CreateRunPage;