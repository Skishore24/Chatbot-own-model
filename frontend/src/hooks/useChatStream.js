/**
 * src/hooks/useChatStream.js
 * ----------------------------------------------------
 * GENKIT AI v5.0 Custom React Hook for Real-time Token Streaming State
 */

import { useState, useCallback, useRef } from "react";
import { streamChatResponse } from "../services/sse_client";

export function useChatStream() {
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      role: "assistant",
      text: "Hello! I am Genkit AI Assistant. How can I help you with Genkit's AI, web development, or cloud services today?",
      timestamp: new Date().toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      }),
    },
  ]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId] = useState(
    () => "session_" + Math.random().toString(36).substring(2, 10),
  );

  const sendMessage = useCallback(
    async (userQuery) => {
      if (!userQuery.trim() || isStreaming) return;

      const userMsg = {
        id: "msg_" + Date.now(),
        role: "user",
        text: userQuery,
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };

      const assistantMsgId = "asst_" + Date.now();
      const initialAssistantMsg = {
        id: assistantMsgId,
        role: "assistant",
        text: "",
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };

      setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
      setIsStreaming(true);

      await streamChatResponse({
        message: userQuery,
        sessionId,
        onStart: () => {},
        onToken: (chunk) => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? { ...msg, text: msg.text + chunk }
                : msg,
            ),
          );
        },
        onEnd: () => {
          setIsStreaming(false);
        },
        onError: (err) => {
          setIsStreaming(false);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMsgId
                ? {
                    ...msg,
                    text: "I apologize, an error occurred while processing your request. Please try again.",
                  }
                : msg,
            ),
          );
        },
      });
    },
    [isStreaming, sessionId],
  );

  const clearHistory = useCallback(() => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        text: "History cleared. How else can I assist you today?",
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ]);
  }, []);

  return {
    messages,
    isStreaming,
    sessionId,
    sendMessage,
    clearHistory,
  };
}
