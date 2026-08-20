import { useEffect } from "react";

/**
 * =====================================================
 * Auto Grow Textarea Hook
 * =====================================================
 *
 * Automatically increases the height of a textarea
 * while typing.
 *
 * Usage:
 *
 * const textareaRef = useRef(null);
 * useAutoGrow(textareaRef, value);
 *
 */

export default function useAutoGrow(ref, value) {
  useEffect(() => {
    const textarea = ref.current;

    if (!textarea) return;

    textarea.style.height = "auto";

    textarea.style.height = Math.min(textarea.scrollHeight, 120) + "px";
  }, [value, ref]);
}
