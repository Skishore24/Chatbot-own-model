// ── Session Management ────────────────────────────────────────────────────────
let sessionId = localStorage.getItem("genkit_session") || null;
const API_BASE = window.location.port && window.location.port !== "8000" ? "http://127.0.0.1:8000" : "";

// ── DOM Ready ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  renderWelcome();
  addSuggestions();
  autoGrowTextarea();
  
  // Open chat by default
  setTimeout(toggleChat, 500);
});

// ── Toggle Chat ─────────────────────────────────────────────────────────────
function toggleChat() {
  const chatBox = document.getElementById("chatBox");
  const badge   = document.getElementById("chatBadge");
  const isOpen  = chatBox.classList.toggle("active");

  // Toggle button always shows the chat bubble icon
  // (header already has its own close ✕ button)
  if (isOpen && badge) badge.style.display = "none";
  if (isOpen) setTimeout(() => document.getElementById("input").focus(), 300);
}

// ── Welcome Message ───────────────────────────────────────────────────────────
function renderWelcome() {
  const text   = "👋 Hi! I'm the **Genkit AI Assistant**.\n\nAsk me anything about our services, portfolio, or how we can help your business!";
  const bubble = appendBotMessage("", false);
  typeWriter(text, bubble);
}

// ── Timestamp helper ──────────────────────────────────────────────────────────
function getTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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
    const res = await fetch(`${API_BASE}/chat`, {
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

  // avatar
  const av = document.createElement("div");
  av.className = "avatar user-avatar";
  av.innerHTML = `<span class="material-symbols-rounded">person</span>`;

  // bubble
  const bubble = document.createElement("div");
  bubble.className = "message user-message";
  bubble.textContent = text;

  // time
  const time = document.createElement("span");
  time.className = "msg-time user-time";
  time.textContent = getTime();

  // wrapper holds bubble + time, aligned right
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap user-wrap";
  wrap.appendChild(bubble);
  wrap.appendChild(time);

  row.appendChild(wrap);
  row.appendChild(av);
  msgBox.appendChild(row);
  scrollToBottom();
}

function appendBotMessage(text, showTime = true) {
  const msgBox = document.getElementById("messages");
  const row    = document.createElement("div");
  row.className = "message-row bot-row";

  // avatar
  const av = document.createElement("div");
  av.className = "avatar bot-avatar";
  av.innerHTML = `<img src="./images/logo1.png" alt="Genkit" class="bot-logo">`;

  // bubble
  const bubble = document.createElement("div");
  bubble.className = "message bot-message";
  bubble.innerHTML = renderMarkdown(text);

  // time
  const time = document.createElement("span");
  time.className = "msg-time";
  if (showTime) time.textContent = getTime();

  // wrapper holds bubble + time
  const wrap = document.createElement("div");
  wrap.className = "msg-wrap bot-wrap";
  wrap.appendChild(bubble);
  wrap.appendChild(time);

  row.appendChild(av);
  row.appendChild(wrap);
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
      <img src="./images/logo1.png" class="bot-logo" alt="Genkit">
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
// Rule: only render <ul> if 2 or more bullet/numbered items.
// A single bullet → strip the bullet and render as plain paragraph.
function renderMarkdown(text) {
  if (!text) return "";

  // ── 1. Count bullet/numbered lines BEFORE escaping ──────────────────
  const bulletLines = (text.match(/^([-•]|\d+\.)\s+.+/gm) || []);
  const isList = bulletLines.length >= 2;

  // ── 2. Strip single bullets into plain text ──────────────────────────
  let processed = text;
  if (!isList && bulletLines.length === 1) {
    // Remove the bullet prefix so it renders as a paragraph
    processed = text.replace(/^([-•]|\d+\.)\s+/gm, "");
  }

  let html = escapeHTML(processed);

  // ── 3. Inline formatting ─────────────────────────────────────────────
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g,     "<em>$1</em>");
  html = html.replace(/_(.+?)_/g,       "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g,     "<code>$1</code>");

  // ── 4. Lists (only when 2+ items) ───────────────────────────────────
  if (isList) {
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/^[-•]\s+(.+)$/gm,   "<li>$1</li>");
    // Wrap consecutive <li> in <ul>
    html = html.replace(/(<li>.*?<\/li>(\n|<br>)*)+/gs, m =>
      `<ul>${m.replace(/<br>/g, "")}</ul>`
    );
  }

  // ── 5. Line breaks ───────────────────────────────────────────────────
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
    const res = await fetch(`${API_BASE}/lead`, {
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