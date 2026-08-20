/**
 * src/services/sse_client.js
 * ----------------------------------------------------
 * GENKIT AI v6.0 Enterprise Real-time SSE Token Stream Client
 * Reads Server-Sent Events (text/event-stream) using native Fetch ReadableStream.
 */

const API_BASE = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_URL)
  ? import.meta.env.VITE_API_URL.replace(/\/+$/, "")
  : "";

export async function streamChatResponse({
  message,
  sessionId,
  onStart,
  onToken,
  onEnd,
  onError,
}) {
  try {
    const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(
        errorData.detail || `Server returned status ${response.status}`,
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const payload = JSON.parse(line.replace(/^data:\s*/, ""));

            if (payload.event === "start" && onStart) {
              onStart(payload);
            } else if (payload.event === "token" && onToken) {
              onToken(payload.chunk);
            } else if (payload.event === "end" && onEnd) {
              onEnd(payload);
            }
          } catch (e) {
            console.warn("Failed to parse SSE line:", line, e);
          }
        }
      }
    }
  } catch (err) {
    if (onError) {
      onError(err);
    } else {
      console.error("SSE Streaming Error:", err);
    }
  }
}
