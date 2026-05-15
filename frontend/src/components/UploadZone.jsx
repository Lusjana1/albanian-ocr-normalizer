import { useRef, useState } from "react";

export default function UploadZone({ file, preview, onFileChange, onSubmit, loading }) {
  const inputRef = useRef();
  const [dragging, setDragging] = useState(false);

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) onFileChange(dropped);
  }

  function handleChange(e) {
    const selected = e.target.files[0];
    if (selected) onFileChange(selected);
  }

  return (
    <div className="card">
      <div className="card-header">📄 Upload Document</div>
      <div className="card-body">
        <div
          className={`upload-zone${dragging ? " drag-over" : ""}`}
          onClick={() => inputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <input ref={inputRef} type="file" accept="image/*" onChange={handleChange} />
          <div className="upload-icon">{file ? "🖼️" : "📤"}</div>
          {file ? (
            <>
              <div className="upload-label"><strong>{file.name}</strong></div>
              <div className="upload-hint">{(file.size / 1024).toFixed(1)} KB — click to change</div>
            </>
          ) : (
            <>
              <div className="upload-label">
                <strong>Click to browse</strong> or drag &amp; drop
              </div>
              <div className="upload-hint">JPEG · PNG · WEBP · TIFF · BMP — max 20 MB</div>
            </>
          )}
        </div>

        {preview && (
          <div className="preview-wrap">
            <img src={preview} alt="Document preview" />
          </div>
        )}

        <button
          className="process-btn"
          onClick={onSubmit}
          disabled={!file || loading}
        >
          {loading ? (
            <><span className="spinner" />Processing…</>
          ) : (
            "Extract &amp; Normalize Text"
          )}
        </button>
      </div>
    </div>
  );
}
