import { useEffect, useState } from "react";
import { deleteHistoryItem, fetchHistory } from "../api";

export default function HistoryPanel({ refreshTrigger, onSelect }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);

  async function load() {
    setLoading(true);
    try {
      const data = await fetchHistory();
      setItems(data);
    } catch (_) {
      // silently fail
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { load(); }, [refreshTrigger]);

  async function handleDelete(e, id) {
    e.stopPropagation();
    await deleteHistoryItem(id);
    setItems((prev) => prev.filter((i) => i.id !== id));
  }

  if (loading) return null;
  if (!items.length) return null;

  return (
    <div className="history-section">
      <div className="card-header" style={{ marginBottom: 12 }}>🕓 Recent Documents</div>
      <div className="history-list">
        {items.map((item) => (
          <div
            key={item.id}
            className="history-item"
            onClick={() => onSelect(item)}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div className="history-filename">{item.filename}</div>
              <div className="history-preview">
                {item.normalized_text.slice(0, 100)}{item.normalized_text.length > 100 ? "…" : ""}
              </div>
              <div className="history-meta">
                {item.word_count} words · {item.processing_time_ms.toFixed(0)}ms · {item.created_at.slice(0, 16)}
              </div>
            </div>
            <button
              className="history-delete"
              onClick={(e) => handleDelete(e, item.id)}
              title="Delete"
            >
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
