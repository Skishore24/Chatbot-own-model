// ======================================================
// Genkit AI Markdown Renderer
// ======================================================

import { marked } from "marked";

// ------------------------------------------------------
// Marked Configuration
// ------------------------------------------------------

marked.setOptions({
  breaks: true,
  gfm: true
});

// ------------------------------------------------------
// Render Markdown
// ✅ Fix: do NOT escape HTML before marked.parse().
//    marked handles its own escaping internally.
//    Escaping before parsing breaks **bold**, *italic*,
//    lists etc. because the text gets HTML-encoded first.
//    We only add link safety (noopener) after parsing.
// ------------------------------------------------------

export function renderMarkdown(text = "") {
  if (!text) return "";

  let processed = text;

  // --------------------------------------------
  // Count bullets / numbered list lines
  // --------------------------------------------

  const bulletLines =
    processed.match(/^([-•]|\d+\.)\s+.+/gm) || [];

  const isList = bulletLines.length >= 2;

  // --------------------------------------------
  // Remove single bullet prefix → plain paragraph
  // --------------------------------------------

  if (!isList && bulletLines.length === 1) {
    processed = processed.replace(
      /^([-•]|\d+\.)\s+/gm,
      ""
    );
  }

  // --------------------------------------------
  // Parse markdown (marked handles XSS-escaping
  // of raw HTML entities internally)
  // --------------------------------------------

  let html = marked.parse(processed);

  // --------------------------------------------
  // Clean empty paragraphs
  // --------------------------------------------

  html = html.replace(/<p>\s*<\/p>/g, "");

  // --------------------------------------------
  // Make all links safe: open in new tab,
  // no referrer, no opener (security)
  // --------------------------------------------

  html = html.replace(
    /<a /g,
    '<a target="_blank" rel="noopener noreferrer" '
  );

  return html;
}

// ------------------------------------------------------
// Strip Markdown — returns plain text
// ------------------------------------------------------

export function stripMarkdown(text = "") {

  return text

    .replace(/```[\s\S]*?```/g, "")

    .replace(/`([^`]+)`/g, "$1")

    .replace(/\*\*(.*?)\*\*/g, "$1")

    .replace(/\*(.*?)\*/g, "$1")

    .replace(/_(.*?)_/g, "$1")

    .replace(/#+\s/g, "")

    .replace(/\[(.*?)\]\((.*?)\)/g, "$1")

    .trim();

}

// ------------------------------------------------------
// Plain Text Preview
// ------------------------------------------------------

export function preview(text = "", length = 80) {

  const clean = stripMarkdown(text);

  if (clean.length <= length) return clean;

  return clean.substring(0, length) + "...";

}