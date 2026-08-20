import { useEffect, useRef, useState, useCallback } from "react";

import { sendChatMessage, submitLead } from "../services/api";

import {
  DEFAULT_SUGGESTIONS,
  WELCOME_MESSAGE,
  getCurrentTime,
  generateId,
  isValidEmail,
} from "../utils/helpers";

import { renderMarkdown } from "../utils/markdown";

export default function useChat() {
  // ======================================================
  // Chat State
  // ======================================================

  const [messages, setMessages] = useState([]);
  const [typing, setTyping] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [showLeadForm, setShowLeadForm] = useState(false);
  const [sendingLead, setSendingLead] = useState(false);
  const [leadError, setLeadError] = useState("");
  const [leadData, setLeadData] = useState({
    name: "",
    email: "",
  });

  // ======================================================
  // References
  // ======================================================
  const messagesRef = useRef(null);
  const textareaRef = useRef(null);
  const isStreamingRef = useRef(false);

  // ======================================================
  // Suggestions
  // ======================================================

  const suggestions = DEFAULT_SUGGESTIONS;

  // ======================================================
  // Toggle Widget
  // ======================================================

  function toggleChat() {
    setIsOpen((prev) => !prev);
  }

  // ======================================================
  // Welcome Message
  // ======================================================

  useEffect(() => {
    const welcome = {
      id: generateId(),

      sender: "bot",

      html: renderMarkdown(WELCOME_MESSAGE),

      text: WELCOME_MESSAGE,

      time: getCurrentTime(),
    };

    setMessages([welcome]);
  }, []);

  // ======================================================
  // Auto Open Widget
  // ======================================================

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsOpen(true);
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  // ======================================================
  // Auto Scroll
  // ======================================================

  useEffect(() => {
    if (!messagesRef.current) return;

    messagesRef.current.scrollTop = messagesRef.current.scrollHeight;
  }, [messages, typing]);

  // ======================================================
  // Suggestion Click
  // ✅ Fix: auto-send after filling input (like tests/HTML)
  // ======================================================

  function handleSuggestion(text) {
    // Set input and immediately send
    setInput(text);

    // Use a short timeout so React can flush the input state
    // before sendMessage() reads it (or pass text directly)
    sendMessage(text);
  }

  // ======================================================
  // Lead Form Input
  // ======================================================

  function updateLead(field, value) {
    setLeadData((prev) => ({
      ...prev,

      [field]: value,
    }));
  }

  // ======================================================
  // Send Message
  // ✅ Fix: isTyping guard + proper typing/placeholder logic
  // ======================================================

  const sendMessage = useCallback(
    async (customMessage = null) => {
      // ✅ Guard: prevent double-send while streaming
      if (isStreamingRef.current) return;

      const text = (customMessage ?? input).trim();

      if (!text) return;

      // ✅ Max length guard (matches backend max_length=1000)
      if (text.length > 1000) return;

      // Set streaming guard immediately
      isStreamingRef.current = true;
      setIsStreaming(true);

      // -------------------------------
      // User Message
      // -------------------------------

      const userMessage = {
        id: generateId(),

        sender: "user",

        text,

        html: text,

        time: getCurrentTime(),
      };

      setMessages((prev) => [...prev, userMessage]);

      setInput("");

      // ✅ Fix: show typing indicator (not a placeholder bot message)
      // The placeholder bot message approach caused dual-render bugs.
      // We use the typing state for the TypingIndicator component,
      // and only add the real bot message once we have content.
      setTyping(true);

      const botId = generateId();
      let botMessageAdded = false;

      try {
        await sendChatMessage(
          text,

          // ===================================
          // Streaming Callback — first chunk
          // creates the bot message, subsequent
          // chunks update it in-place
          // ===================================

          (chunk) => {
            if (!botMessageAdded) {
              // First chunk: add bot message and hide typing indicator
              botMessageAdded = true;
              setTyping(false);
              setMessages((prev) => [
                ...prev,
                {
                  id: botId,
                  sender: "bot",
                  text: chunk,
                  html: renderMarkdown(chunk),
                  time: getCurrentTime(),
                },
              ]);
            } else {
              // Subsequent chunks: update existing bot message
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botId
                    ? {
                        ...msg,

                        text: chunk,

                        html: renderMarkdown(chunk),
                      }
                    : msg,
                ),
              );
            }
          },

          // ===================================
          // Completed
          // ===================================

          (finalText) => {
            setTyping(false);

            if (!botMessageAdded) {
              // Edge case: stream completed with no chunks (e.g., empty response)
              botMessageAdded = true;
              setMessages((prev) => [
                ...prev,
                {
                  id: botId,
                  sender: "bot",
                  text: finalText || "I couldn't find an answer to that.",
                  html: renderMarkdown(
                    finalText || "I couldn't find an answer to that.",
                  ),
                  time: getCurrentTime(),
                },
              ]);
            } else {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botId
                    ? {
                        ...msg,

                        text: finalText,

                        html: renderMarkdown(finalText),
                      }
                    : msg,
                ),
              );
            }

            // -------------------------------
            // Show Lead Form
            // ✅ Expanded trigger conditions
            // -------------------------------

            if (
              finalText.includes("👉") ||
              finalText.toLowerCase().includes("free quote") ||
              finalText.toLowerCase().includes("get in touch") ||
              finalText.toLowerCase().includes("contact us")
            ) {
              setShowLeadForm(true);
            }
          },

          // ===================================
          // Error
          // ===================================

          (error) => {
            setTyping(false);

            const errorText =
              "⚠️ " + (error || "Unable to connect to the server.");

            if (!botMessageAdded) {
              botMessageAdded = true;
              setMessages((prev) => [
                ...prev,
                {
                  id: botId,
                  sender: "bot",
                  text: errorText,
                  html: renderMarkdown(errorText),
                  time: getCurrentTime(),
                },
              ]);
            } else {
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botId
                    ? {
                        ...msg,

                        text: errorText,

                        html: renderMarkdown(errorText),
                      }
                    : msg,
                ),
              );
            }
          },
        );
      } catch (error) {
        setTyping(false);

        const errText = "⚠️ Unable to connect to the server.";

        if (!botMessageAdded) {
          setMessages((prev) => [
            ...prev,
            {
              id: botId,
              sender: "bot",
              text: errText,
              html: renderMarkdown(errText),
              time: getCurrentTime(),
            },
          ]);
        } else {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botId
                ? {
                    ...msg,

                    text: errText,

                    html: renderMarkdown(errText),
                  }
                : msg,
            ),
          );
        }

        console.error(error);
      } finally {
        // Release streaming guards
        isStreamingRef.current = false;
        setIsStreaming(false);
      }
    },
    [input],
  );

  // ======================================================
  // Submit Lead Form
  // ======================================================

  async function submitLeadForm() {
    setLeadError("");

    // -----------------------------
    // Validation
    // -----------------------------

    if (!leadData.name.trim()) {
      setLeadError("Please enter your name.");

      return;
    }

    if (!isValidEmail(leadData.email)) {
      setLeadError("Please enter a valid email address.");

      return;
    }

    setSendingLead(true);

    try {
      await submitLead({
        name: leadData.name.trim(),

        email: leadData.email.trim(),
      });

      // -----------------------------
      // Success Message
      // -----------------------------

      const successMessage = {
        id: generateId(),

        sender: "bot",

        text: `✅ Thanks ${leadData.name}! We'll contact you at ${leadData.email} soon.`,

        html: renderMarkdown(
          `✅ Thanks **${leadData.name}**! We'll contact you at **${leadData.email}** soon.`,
        ),

        time: getCurrentTime(),
      };

      setMessages((prev) => [...prev, successMessage]);

      // -----------------------------
      // Reset Form
      // -----------------------------

      setLeadData({
        name: "",

        email: "",
      });

      setShowLeadForm(false);
    } catch (error) {
      console.error(error);

      setLeadError("Unable to submit your request. Please try again.");
    } finally {
      setSendingLead(false);
    }
  }

  // ======================================================
  // Close Lead Form
  // ======================================================

  function closeLeadForm() {
    setShowLeadForm(false);

    setLeadError("");

    setLeadData({
      name: "",

      email: "",
    });
  }

  // ======================================================
  // Handle Enter Key
  // ======================================================

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      sendMessage();
    }
  }

  // ======================================================
  // Reset Chat
  // ======================================================

  function resetChat() {
    const welcome = {
      id: generateId(),

      sender: "bot",

      text: WELCOME_MESSAGE,

      html: renderMarkdown(WELCOME_MESSAGE),

      time: getCurrentTime(),
    };

    setMessages([welcome]);

    setInput("");

    setTyping(false);

    setShowLeadForm(false);

    setLeadData({
      name: "",

      email: "",
    });

    setLeadError("");

    isStreamingRef.current = false;
  }

  // ======================================================
  // Public API
  // ======================================================

  return {
    // -----------------------------
    // Chat
    // -----------------------------

    messages,

    input,

    setInput,

    sendMessage,

    typing,

    isOpen,

    toggleChat,

    resetChat,

    // Expose streaming state for disabling input
    isStreaming,

    // -----------------------------
    // Suggestions
    // -----------------------------

    suggestions,

    handleSuggestion,

    // -----------------------------
    // Lead Form
    // -----------------------------

    showLeadForm,

    leadData,

    updateLead,

    submitLeadForm,

    closeLeadForm,

    sendingLead,

    leadError,

    // -----------------------------
    // Refs
    // -----------------------------

    messagesRef,

    textareaRef,

    // -----------------------------
    // Keyboard
    // -----------------------------

    handleKeyDown,
  };
}
