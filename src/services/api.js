// ======================================================
// Genkit AI API Service
// ======================================================

// If using Vite proxy, keep this empty.
// For production you can change it:
// const API_BASE = "https://your-domain.com";
const API_BASE = "";


// ------------------------------------------------------
// Session
// ------------------------------------------------------

export function getSessionId() {
  return localStorage.getItem("genkit_session");
}

export function saveSessionId(id) {
  if (!id) return;
  localStorage.setItem("genkit_session", id);
}


// ------------------------------------------------------
// Chat Streaming API
// ------------------------------------------------------

export async function sendChatMessage(
  message,
  onChunk,
  onComplete,
  onError
) {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        q: message,
        session_id: getSessionId() || undefined
      })
    });

    if (!response.ok) {
      // Try to extract the server error message
      let errorDetail = "Failed to connect to server.";
      try {
        const errBody = await response.json();
        if (errBody?.detail) errorDetail = errBody.detail;
      } catch (_) { /* ignore */ }
      throw new Error(errorDetail);
    }

    // ✅ Fix: correct header name is X-Session-ID (capital ID)
    // Backend sends: "X-Session-ID" in StreamingResponse headers
    const session = response.headers.get("X-Session-ID");

    if (session) {
      saveSessionId(session);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let fullResponse = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      const chunk = decoder.decode(value, {
        stream: true
      });

      fullResponse += chunk;

      if (onChunk) {
        onChunk(fullResponse);
      }
    }

    if (onComplete) {
      onComplete(fullResponse);
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

  const response = await fetch(`${API_BASE}/lead`, {

    method: "POST",

    headers: {
      "Content-Type": "application/json"
    },

    body: JSON.stringify({

      ...data,

      session_id: getSessionId() || undefined

    })

  });

  if (!response.ok) {
    let errorDetail = "Unable to submit lead.";
    try {
      const errBody = await response.json();
      if (errBody?.detail) errorDetail = errBody.detail;
    } catch (_) { /* ignore */ }
    throw new Error(errorDetail);
  }

  return await response.json();

}



// ------------------------------------------------------
// Health Check
// ------------------------------------------------------

export async function checkServer() {

  try {

    const response = await fetch(`${API_BASE}/health`);

    return response.ok;

  } catch {

    return false;

  }

}