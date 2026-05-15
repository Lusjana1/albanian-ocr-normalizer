import { useState } from "react";
import { processImage } from "./api";
import Header from "./components/Header";
import HistoryPanel from "./components/HistoryPanel";
import ResultPanel from "./components/ResultPanel";
import UploadZone from "./components/UploadZone";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [historyKey, setHistoryKey] = useState(0);

  function handleFileChange(selected) {
    setFile(selected);
    if (preview) URL.revokeObjectURL(preview);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  }

  async function handleSubmit() {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await processImage(file);
      setResult(data);
      setHistoryKey((k) => k + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleHistorySelect(item) {
    setResult({
      raw_text: item.raw_text,
      cleaned_text: item.cleaned_text,
      normalized_text: item.normalized_text,
      word_count: item.word_count,
      char_count: item.cleaned_text.length,
      processing_time_ms: item.processing_time_ms,
      ocr_engine: "cached",
    });
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <>
      <Header />
      <div className="app-body">
        <div className="main-grid">
          {/* Left column: upload */}
          <div>
            <UploadZone
              file={file}
              preview={preview}
              onFileChange={handleFileChange}
              onSubmit={handleSubmit}
              loading={loading}
            />
          </div>

          {/* Right column: results */}
          <div>
            {error && (
              <div className="error-box">
                <span>⚠️</span>
                <div><strong>Error:</strong> {error}</div>
              </div>
            )}
            <ResultPanel result={result} loading={loading} />
          </div>
        </div>

        <HistoryPanel refreshTrigger={historyKey} onSelect={handleHistorySelect} />
      </div>
    </>
  );
}
