// ======================================================
// Genkit AI API Service v6.0
// ======================================================

const API_BASE = (typeof import.meta !== "undefined" && import.meta.env && import.meta.env.VITE_API_URL)
  ? import.meta.env.VITE_API_URL.replace(/\/+$/, "")
  : "";

// ------------------------------------------------------
// Session
// ------------------------------------------------------

export function getSessionId() {
  let session = localStorage.getItem("genkit_session");
  if (!session) {
    session = "session_" + Math.random().toString(36).substring(2, 12);
    localStorage.setItem("genkit_session", session);
  }
  return session;
}

export function saveSessionId(id) {
  if (!id) return;
  localStorage.setItem("genkit_session", id);
}

// ------------------------------------------------------
// Chat Streaming API
// ------------------------------------------------------

export async function sendChatMessage(message, onChunk, onComplete, onError) {
  try {
    const sessionId = getSessionId();
    const response = await fetch(`${API_BASE}/api/v1/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        session_id: sessionId,
      }),
    });

    if (!response.ok) {
      let errorDetail = "Failed to connect to server.";
      try {
        const errBody = await response.json();
        if (errBody?.detail) errorDetail = errBody.detail;
      } catch (_) {
        /* ignore */
      }
      throw new Error(errorDetail);
    }

    const session = response.headers.get("X-Session-ID") || sessionId;
    if (session) {
      saveSessionId(session);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let fullResponse = "";
    let sseBuffer = "";
    let metadata = { sources: [], intent: "General", grounded: true, confidence: 1.0 };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunkText = decoder.decode(value, { stream: true });
      sseBuffer += chunkText;

      // Parse SSE lines
      if (sseBuffer.includes("data: ")) {
        const lines = sseBuffer.split("\n\n");
        sseBuffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            try {
              const payload = JSON.parse(trimmed.replace(/^data:\s*/, ""));
              if (payload.event === "start") {
                if (payload.sources) metadata.sources = payload.sources;
                if (payload.intent) metadata.intent = payload.intent;
                if (payload.grounded !== undefined) metadata.grounded = payload.grounded;
              } else if (payload.event === "token" && payload.chunk) {
                fullResponse += payload.chunk;
                if (onChunk) onChunk(fullResponse, metadata);
              } else if (payload.event === "end") {
                if (payload.session_id) saveSessionId(payload.session_id);
                if (payload.confidence !== undefined) metadata.confidence = payload.confidence;
              }
            } catch (_) {
              const raw = trimmed.replace(/^data:\s*/, "");
              if (raw && !raw.startsWith("{")) {
                fullResponse += raw;
                if (onChunk) onChunk(fullResponse, metadata);
              }
            }
          }
        }
      } else {
        fullResponse += chunkText;
        if (onChunk) onChunk(fullResponse, metadata);
      }
    }

    if (onComplete) {
      onComplete(fullResponse, metadata);
    }

    return fullResponse;

  } catch (error) {
    console.error("[Genkit API Error]", error);
    if (onError) {
      onError(error.message || "Connection error.");
    }
    throw error;
  }
}

// ------------------------------------------------------
// Lead API
// ------------------------------------------------------

export async function submitLead(data) {
  const response = await fetch(`${API_BASE}/api/v1/leads`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ...data,
      session_id: getSessionId() || undefined,
    }),
  });

  if (!response.ok) {
    let errorDetail = "Unable to submit lead.";
    try {
      const errBody = await response.json();
      if (errBody?.detail) errorDetail = errBody.detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(errorDetail);
  }

  return await response.json();
}

// ------------------------------------------------------
// Health Check
// ------------------------------------------------------

export async function checkServer() {
  try {
    const response = await fetch(`${API_BASE}/api/v1/health`);
    return response.ok;
  } catch {
    return false;
  }
}
