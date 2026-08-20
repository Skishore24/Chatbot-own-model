import React from "react";

export default function FloatingButton({ isOpen, toggleChat }) {
  return (
    <button
      className="chat-toggle"
      onClick={toggleChat}
      aria-label={isOpen ? "Close Chat" : "Open Chat"}
      aria-expanded={isOpen}
      type="button"
    >
      <span className="material-symbols-rounded">
        {isOpen ? "close" : "chat"}
      </span>

      {/* Badge only shown when chat is closed */}
      {!isOpen && <span className="badge" aria-hidden="true"></span>}
    </button>
  );
}
