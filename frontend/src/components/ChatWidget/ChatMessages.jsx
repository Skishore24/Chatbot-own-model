import React, { forwardRef } from "react";

import Message from "./Message";
import TypingIndicator from "./TypingIndicator";
import LeadForm from "./LeadForm";

const ChatMessages = forwardRef(
  (
    {
      messages,
      typing,
      showLeadForm,
      leadData,
      updateLead,
      submitLeadForm,
      closeLeadForm,
      sendingLead,
      leadError,
    },
    ref,
  ) => {
    return (
      <div
        className="chat-messages"
        ref={ref}
        role="log"
        aria-live="polite"
        aria-label="Chat messages"
      >
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}

        {typing && <TypingIndicator />}

        {/* ✅ LeadForm inside messages area so it scrolls with chat */}

        {showLeadForm && (
          <LeadForm
            data={leadData}
            updateLead={updateLead}
            submitLeadForm={submitLeadForm}
            closeLeadForm={closeLeadForm}
            sendingLead={sendingLead}
            error={leadError}
          />
        )}
      </div>
    );
  },
);

ChatMessages.displayName = "ChatMessages";

export default ChatMessages;
