import FloatingButton from "./FloatingButton";

import ChatHeader from "./ChatHeader";
import ChatMessages from "./ChatMessages";
import ChatInput from "./ChatInput";
import Suggestions from "./Suggestions";
import LeadForm from "./LeadForm";

import useChat from "../../hooks/useChat";

export default function ChatWidget() {
  const {
    // Chat

    messages,
    input,
    setInput,
    sendMessage,

    typing,

    isOpen,
    toggleChat,

    // Streaming guard (disables input while bot is responding)
    isStreaming,

    // Suggestions

    suggestions,
    handleSuggestion,

    // Lead

    showLeadForm,
    leadData,
    updateLead,
    submitLeadForm,
    closeLeadForm,
    sendingLead,
    leadError,

    // Refs

    messagesRef,
    textareaRef,

    // Keyboard

    handleKeyDown,
  } = useChat();

  return (
    <>
      {/* Floating Button */}

      <FloatingButton isOpen={isOpen} toggleChat={toggleChat} />

      {/* Widget */}

      <div
        className={`chat-container ${isOpen ? "active" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label="Genkit AI Chat"
      >
        {/* Header */}

        <ChatHeader toggleChat={toggleChat} />

        {/* Messages — LeadForm lives inside the scrollable messages area */}

        <ChatMessages
          ref={messagesRef}
          messages={messages}
          typing={typing}
          showLeadForm={showLeadForm}
          leadData={leadData}
          updateLead={updateLead}
          submitLeadForm={submitLeadForm}
          closeLeadForm={closeLeadForm}
          sendingLead={sendingLead}
          leadError={leadError}
        />

        {/* Suggestions — only show when not streaming */}

        {!typing && !isStreaming && (
          <Suggestions suggestions={suggestions} onSelect={handleSuggestion} />
        )}

        {/* Input */}

        <ChatInput
          value={input}
          onChange={setInput}
          onSend={sendMessage}
          onKeyDown={handleKeyDown}
          textareaRef={textareaRef}
          disabled={typing || isStreaming}
        />
      </div>
    </>
  );
}
