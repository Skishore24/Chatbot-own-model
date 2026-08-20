import React from "react";

import logo from "../../assets/images/logo1.png";

export default function ChatHeader({ toggleChat }) {
  return (
    <div className="chat-header">
      {/* Logo */}

      <div className="bot-logo-header">
        <img src={logo} alt="Genkit Logo" />
      </div>

      {/* Title */}

      <div className="header-info">
        <span className="header-title">Genkit Assistant</span>

        <span className="header-subtitle">Online · Ready to help</span>
      </div>

      {/* Close */}

      <button
        className="close-btn"
        onClick={toggleChat}
        aria-label="Close chat"
        type="button"
      >
        <span className="material-symbols-rounded">close</span>
      </button>
    </div>
  );
}
