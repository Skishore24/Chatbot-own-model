// ── Session Management ────────────────────────────────────────────────────────
let sessionId = localStorage.getItem("genkit_session") || null;

// ── DOM Ready ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderWelcome();
  addSuggestions();
  autoGrowTextarea();
  
  // Open chat by default
  setTimeout(toggleChat, 500);
});

// ── Toggle Chat ───────────────────────────────────────────────────────────────
function toggleChat() {
  const chatBox = document.getElementById("chatBox");
  const icon    = document.getElementById("toggleIcon");
  const isOpen  = chatBox.classList.toggle("active");
  icon.textContent = isOpen ? "close" : "chat";
  if (isOpen) setTimeout(() => document.getElementById("input").focus(), 300);
}

// ── Welcome Message ───────────────────────────────────────────────────────────
function renderWelcome() {
  const text   = "👋 Hi! I'm the **Genkit AI Assistant**.\n\nAsk me anything about our services, portfolio, or how we can help your business!";
  const bubble = appendBotMessage("");
  typeWriter(text, bubble);
}

function typeWriter(text, element, speed = 14) {
  let i = 0;
  element.innerHTML = "";

  function type() {
    if (i < text.length) {
      element.innerText = text.substring(0, i + 1);
      i++;
      setTimeout(type, speed);
    } else {
      element.innerHTML = renderMarkdown(text);
      scrollToBottom();
    }
  }
  type();
}

// ── Quick Suggestions ─────────────────────────────────────────────────────────
function addSuggestions() {
  const msgBox = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "suggestions";

  ["What is Genkit?", "What tools do you use?", "Services offered", "Contact info"].forEach(text => {
    const btn     = document.createElement("button");
    btn.className = "suggestion-btn";
    btn.innerText = text;
    btn.onclick   = () => {
      document.getElementById("input").value = text;
      sendMessage();
    };
    row.appendChild(btn);
  });

  msgBox.appendChild(row);
  setTimeout(scrollToBottom, 100);
}

// ── Send Message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById("input");
  const text  = input.value.trim();
  if (!text) return;

  setInputState(true);
  appendUserMessage(text);
  input.value = "";
  input.style.height = "auto";   // reset auto-grow

  const typingIndicator = appendTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ q: text, session_id: sessionId || undefined }),
    });

    // Save session id from response header
    const newSessionId = res.headers.get("X-Session-Id");
    if (newSessionId) {
      sessionId = newSessionId;
      localStorage.setItem("genkit_session", sessionId);
    }

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData?.detail || "Server error");
    }

    typingIndicator.remove();
    const bubble = appendBotMessage("");
    let fullText = "";

    // ── READ STREAM ──────────────────────────────────────────────────────────
    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      fullText += chunk;
      
      // Update UI in real-time
      bubble.innerText = fullText;
      scrollToBottom();
    }

    // Final render with markdown
    bubble.innerHTML = renderMarkdown(fullText);
    scrollToBottom();

    // Show lead form if AI prompted for it
    if (fullText.includes("👉")) {
      setTimeout(() => showLeadForm(), 600);
    }


  } catch (err) {
    if (typingIndicator) typingIndicator.remove();
    appendBotMessage(`⚠️ ${err.message || "Server error. Please try again."}`);
    console.error("[Genkit Chat Error]", err);
  } finally {
    setInputState(false);
    document.getElementById("input").focus();
  }
}

// ── Append Messages ───────────────────────────────────────────────────────────
function appendUserMessage(text) {
  const msgBox = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "message-row user-row";
  row.innerHTML = `
    <div class="avatar user-avatar">
      <span class="material-symbols-rounded">person</span>
    </div>
    <div class="message">${escapeHTML(text)}</div>
  `;
  msgBox.appendChild(row);
  scrollToBottom();
}

function appendBotMessage(text) {
  const msgBox = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "message-row bot-row";

  const bubble  = document.createElement("div");
  bubble.className = "message";
  bubble.innerHTML = renderMarkdown(text);

  row.innerHTML = `
    <div class="avatar bot-avatar">
      <img src="/static/images/logo1.png" alt="Genkit" class="bot-logo">
    </div>
  `;
  row.appendChild(bubble);
  msgBox.appendChild(row);
  scrollToBottom();
  return bubble;
}

// ── Typing Indicator ──────────────────────────────────────────────────────────
function appendTypingIndicator() {
  const msgBox = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "message-row bot-row";
  row.innerHTML = `
    <div class="avatar bot-avatar">
      <img src="/static/images/logo1.png" class="bot-logo" alt="Genkit">
    </div>
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>
  `;
  msgBox.appendChild(row);
  scrollToBottom();
  return row;
}

// ── Markdown Renderer ─────────────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHTML(text);

  // Bold, italic, inline code
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g,     "<em>$1</em>");
  html = html.replace(/_(.+?)_/g,       "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g,     "<code>$1</code>");

  // Numbered lists  (1. item)
  html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");

  // Bullet lists  (- item or • item)
  html = html.replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>");

  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li>.*?<\/li>(\n|<br>)*)+/gs, match =>
    `<ul>${match.replace(/<br>/g, "")}</ul>`
  );

  // Line breaks
  html = html.replace(/\n/g, "<br>");

  return html;
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escapeHTML(str) {
  return str.replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  }[c]));
}

function setInputState(disabled) {
  document.getElementById("input").disabled   = disabled;
  document.getElementById("sendBtn").disabled = disabled;
}

function scrollToBottom() {
  const msgBox = document.getElementById("messages");
  msgBox.scrollTop = msgBox.scrollHeight;
}

// ── Auto-grow textarea ────────────────────────────────────────────────────────
function autoGrowTextarea() {
  const ta = document.getElementById("input");
  ta.addEventListener("input", () => {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 120) + "px";
  });
}

// ── Lead Capture Form ─────────────────────────────────────────────────────────
function showLeadForm() {
  // Don't show more than once
  if (document.getElementById("leadFormCard")) return;

  const msgBox = document.getElementById("messages");
  const card   = document.createElement("div");
  card.className = "lead-form-card";
  card.id        = "leadFormCard";
  card.innerHTML = `
    <p class="lead-form-title">✉️ Get a Free Quote</p>
    <input  id="leadName"  class="lead-input" type="text"  placeholder="Your Name"  autocomplete="name">
    <input  id="leadEmail" class="lead-input" type="email" placeholder="Your Email" autocomplete="email">
    <p id="leadError" class="lead-error" style="display:none;"></p>
    <button id="leadSubmitBtn" class="lead-submit" onclick="submitLead()">Send →</button>
  `;
  msgBox.appendChild(card);
  setTimeout(scrollToBottom, 100);
}

async function submitLead() {
  const name      = document.getElementById("leadName").value.trim();
  const email     = document.getElementById("leadEmail").value.trim();
  const errorEl   = document.getElementById("leadError");
  const submitBtn = document.getElementById("leadSubmitBtn");

  // Client-side validation
  if (!name) {
    showLeadError("Please enter your name."); return;
  }
  if (!email || !email.includes("@") || !email.split("@")[1]?.includes(".")) {
    showLeadError("Please enter a valid email address."); return;
  }

  errorEl.style.display = "none";
  submitBtn.disabled    = true;
  submitBtn.textContent = "Sending…";

  try {
    const res = await fetch("/lead", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ name, email, session_id: sessionId || undefined }),
    });

    if (!res.ok) throw new Error("Failed to submit.");

    // Remove form, show success message
    document.getElementById("leadFormCard")?.remove();
    appendBotMessage(`✅ Thanks **${escapeHTML(name)}**! We'll reach out to **${escapeHTML(email)}** soon.`);
    scrollToBottom();

  } catch (err) {
    showLeadError("Something went wrong. Please email us at genkit.tech@gmail.com.");
    submitBtn.disabled    = false;
    submitBtn.textContent = "Send →";
    console.error("[Lead Submit Error]", err);
  }
}

function showLeadError(msg) {
  const el = document.getElementById("leadError");
  if (!el) return;
  el.textContent    = msg;
  el.style.display  = "block";
}

// ── Keyboard Shortcut ─────────────────────────────────────────────────────────
document.addEventListener("keydown", e => {
  const input = document.getElementById("input");
  if (e.key === "Enter" && !e.shiftKey && document.activeElement === input) {
    e.preventDefault();
    sendMessage();
  }
});