const BASE = "/api";

// Maximum time we wait for the backend to finish (milliseconds).
// OCR + normalization on a complex image can take up to ~2 minutes on CPU.
const REQUEST_TIMEOUT_MS = 150_000; // 2.5 minutes

function withTimeout(promise, ms) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  return { promise, controller, timer };
}

export async function processImage(file) {
  const form = new FormData();
  form.append("file", file);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${BASE}/process`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(
        "Processing timed out after 2.5 minutes. " +
        "The image may be too complex or the server is under load. " +
        "Try a smaller or clearer image."
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchHistory(limit = 20) {
  const res = await fetch(`${BASE}/history?limit=${limit}`);
  if (!res.ok) throw new Error("Failed to load history");
  return res.json();
}

export async function deleteHistoryItem(id) {
  const res = await fetch(`${BASE}/history/${id}`, { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete item");
  return res.json();
}
