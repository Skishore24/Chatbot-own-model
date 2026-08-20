import React from "react";

import logo from "../../assets/images/logo1.png";

export default function TypingIndicator() {
  return (
    <div className="message-row bot-row">
      <div className="avatar bot-avatar">
        <img src={logo} alt="Genkit" className="bot-logo" />
      </div>

      <div className="typing-indicator">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
}
