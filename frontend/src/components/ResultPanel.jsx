import StepsIndicator from "./StepsIndicator";
import TextCard from "./TextCard";

export default function ResultPanel({ result, loading }) {
  const activeStep = loading ? 1 : result ? 3 : null;

  if (!result && !loading) {
    return (
      <div className="empty-state">
        <div className="empty-icon">🗒️</div>
        <p>Upload an image and click <strong>Extract &amp; Normalize Text</strong> to see results here.</p>
      </div>
    );
  }

  return (
    <div>
      <StepsIndicator activeStep={activeStep} />

      {result && (
        <>
          <div className="stats-bar">
            <div className="stat">
              <span className="stat-value">{result.word_count}</span>
              <span className="stat-label">Words</span>
            </div>
            <div className="stat">
              <span className="stat-value">{result.char_count}</span>
              <span className="stat-label">Characters</span>
            </div>
            <div className="stat">
              <span className="stat-value">{result.processing_time_ms.toFixed(0)}ms</span>
              <span className="stat-label">Processing time</span>
            </div>
            <div className="stat">
              <span className="stat-value" style={{ fontSize: 14 }}>{result.ocr_engine}</span>
              <span className="stat-label">OCR engine</span>
            </div>
          </div>

          <div className="results-grid">
            <TextCard
              title="🔍 Raw OCR Text"
              text={result.raw_text}
              variant="raw"
              badge="Step 1"
            />
            <TextCard
              title="🧹 Cleaned Text"
              text={result.cleaned_text}
              variant="clean"
              badge="Step 2"
            />
            <TextCard
              title="✨ Normalized Text"
              text={result.normalized_text}
              variant="norm"
              badge="Step 3 · AI"
            />
          </div>
        </>
      )}

      {loading && (
        <div className="empty-state">
          <div className="empty-icon">⚙️</div>
          <p>Running OCR, cleaning, and AI normalization…</p>
        </div>
      )}
    </div>
  );
}
