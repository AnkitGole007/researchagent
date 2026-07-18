// Streaming client for the Research Agent backend.
//
// The backend's POST /api/search returns Server-Sent Events. EventSource only
// supports GET, so we read the response body as a stream and parse the
// `data: {...}` frames ourselves.

// Map a backend PaperOut onto the field names the UI components expect
// (the original mockup used `rel`, `tooNew`, `arxiv`).
function normalizePaper(p) {
  return {
    ...p,
    rel: p.relevance,
    tooNew: p.too_new,
    arxiv: p.arxiv_url,
    pdf: p.pdf_url,
  };
}

export async function streamSearch(req, { onStage, onDone, onError }) {
  let resp;
  try {
    resp = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    });
  } catch (e) {
    onError?.(`Could not reach the server: ${e.message}`);
    return;
  }

  if (!resp.ok || !resp.body) {
    onError?.(`Server error (HTTP ${resp.status})`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let sep;
    while ((sep = buffer.indexOf("\n\n")) >= 0) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);

      const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!dataLine) continue;

      let payload;
      try {
        payload = JSON.parse(dataLine.slice(5).trim());
      } catch {
        continue;
      }

      if (payload.type === "stage") onStage?.(payload);
      else if (payload.type === "done") {
        onDone?.({
          ...payload,
          papers: (payload.papers || []).map(normalizePaper),
        });
      } else if (payload.type === "error") onError?.(payload.message);
    }
  }
}
