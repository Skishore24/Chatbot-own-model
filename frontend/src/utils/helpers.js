// ======================================================
// Genkit AI — Helper Utilities
// ======================================================

// ------------------------------------------------------
// Unique ID
// ------------------------------------------------------

export function generateId() {
  return (
    Date.now().toString(36) +
    Math.random().toString(36).substring(2, 9)
  );
}

// ------------------------------------------------------
// Current Time
// ------------------------------------------------------

export function getCurrentTime() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit"
  });
}

// ------------------------------------------------------
// Email Validation
// ------------------------------------------------------

export function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// ------------------------------------------------------
// Default Suggestions
// ------------------------------------------------------

export const DEFAULT_SUGGESTIONS = [
  "What is Genkit?",
  "Services Offered",
  "AI Development",
  "Website Development",
  "Contact Information"
];

// ------------------------------------------------------
// Welcome Message
// ------------------------------------------------------

export const WELCOME_MESSAGE =
  "👋 Hi! I'm the **Genkit AI Assistant**.\n\nAsk me anything about our services, products, portfolio, AI solutions, or how we can help your business.";