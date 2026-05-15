import { useState } from "react";

export default function TextCard({ title, text, variant, badge }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    });
  }

  function handleDownload() {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${variant}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={`card text-card text-card-${variant}`}>
      <div className="card-header">
        {title}
        {badge && <span className="meta-badge">{badge}</span>}
      </div>
      <div className="card-body">
        <div className="text-area-wrap">{text || "—"}</div>
        <div className="text-actions">
          <button
            className={`btn-sm${copied ? " copied" : ""}`}
            onClick={handleCopy}
            disabled={!text}
          >
            {copied ? "✓ Copied" : "📋 Copy"}
          </button>
          <button className="btn-sm" onClick={handleDownload} disabled={!text}>
            ⬇ Download
          </button>
        </div>
      </div>
    </div>
  );
}
