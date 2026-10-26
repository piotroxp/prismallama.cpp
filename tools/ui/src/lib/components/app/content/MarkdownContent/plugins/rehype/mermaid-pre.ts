import { MERMAID_BLOCK_CLASS, MERMAID_LANGUAGE } from "$lib/constants";
import { createPreTransform } from "./pre-transform";

/**
 * Converts mermaid code blocks to <pre class="mermaid"> for client-side rendering.
 */
export const rehypeMermaidPre = createPreTransform(MERMAID_LANGUAGE, MERMAID_BLOCK_CLASS);
