import React from "react";

import useAutoGrow from "../../hooks/useAutoGrow";

export default function ChatInput({

  value,

  onChange,

  onSend,

  onKeyDown,

  textareaRef,

  disabled = false

}) {

  useAutoGrow(textareaRef, value);

  function handleSend() {

    if (!value.trim() || disabled) return;

    onSend();

  }

  return (

    <div className="chat-input-container">

      <div className="chat-input-wrapper">

        <textarea

          ref={textareaRef}

          rows={1}

          value={value}

          placeholder="Ask Genkit anything..."

          onChange={(e) => onChange(e.target.value)}

          onKeyDown={onKeyDown}

          disabled={disabled}

          maxLength={1000}

          aria-label="Type your message here"

          aria-multiline="true"

          autocomplete="off"

          spellCheck="true"

        />

        <button

          className="send-btn"

          onClick={handleSend}

          disabled={!value.trim() || disabled}

          aria-label="Send message"

        >

          <span className="material-symbols-rounded">

            send

          </span>

        </button>

      </div>

      <div className="footer-text">

        Genkit AI · Powered by our own LLM · Responses may not always be accurate

      </div>

    </div>

  );

}