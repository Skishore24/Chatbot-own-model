import React from "react";

import logo from "../../assets/images/logo1.png";

export default function Message({ message }) {
  const isUser = message.sender === "user";

  return (
    <div className={`message-row ${isUser ? "user-row" : "bot-row"}`}>
      {/* Bot Avatar */}

      {!isUser && (
        <div className="avatar bot-avatar">
          <img src={logo} alt="Genkit" className="bot-logo" />
        </div>
      )}

      {/* Bubble */}

      <div className={`msg-wrap ${isUser ? "user-wrap" : "bot-wrap"}`}>
        {isUser ? (
          // ✅ XSS-safe: user text rendered as plain text (no innerHTML)
          <div className="message user-message">{message.text}</div>
        ) : (
          // Bot messages use sanitized HTML from marked.parse()
          <div
            className="message bot-message"
            dangerouslySetInnerHTML={{
              __html: message.html,
            }}
          />
        )}

        <span className={`msg-time ${isUser ? "user-time" : ""}`}>
          {message.time}
        </span>
      </div>

      {/* User Avatar */}

      {isUser && (
        <div className="avatar user-avatar">
          <span className="material-symbols-rounded">person</span>
        </div>
      )}
    </div>
  );
}
