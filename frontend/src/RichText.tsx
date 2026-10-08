/* Minimal, dependency-free rich-text renderer for bot/agent messages.

 * The LLM answers in markdown; rendering raw strings makes bold markers,
 * headings and bullets visible as literal characters. This renderer is a
 * tiny safe subset: it HTML-escapes everything first, then applies
 * **bold**, *italic*, `code`, auto-linked URLs, # headings, and
 * bullet/numbered lists. No raw HTML ever reaches the DOM.
 */

import type { ReactNode } from "react";

const ESC: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESC[c]);
}

/** Split raw text into plain segments + URLs (so hrefs keep their entities). */
function tokenize(src: string): Array<{ text: string; url?: string }> {
  const out: Array<{ text: string; url?: string }> = [];
  const re = /https?:\/\/[^\s<>"')]+/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(src))) {
    if (m.index > last) out.push({ text: src.slice(last, m.index) });
    out.push({ text: m[0], url: m[0] });
    last = m.index + m[0].length;
  }
  if (last < src.length) out.push({ text: src.slice(last) });
  return out;
}

/** Inline markers on already-escaped text: **b**, *i*, `code`. */
function inline(escaped: string, prefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(escaped))) {
    if (m.index > last) nodes.push(escaped.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      nodes.push(<strong key={`${prefix}-b${i++}`}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      nodes.push(<code key={`${prefix}-c${i++}`}>{tok.slice(1, -1)}</code>);
    } else {
      nodes.push(<em key={`${prefix}-i${i++}`}>{tok.slice(1, -1)}</em>);
    }
    last = m.index + tok.length;
  }
  if (last < escaped.length) nodes.push(escaped.slice(last));
  return nodes;
}

/** One line -> nodes (autolinks + inline markers; text escaped). */
function line(text: string, key: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  tokenize(text).forEach((tok, i) => {
    if (tok.url) {
      nodes.push(
        <a key={`${key}-u${i}`} href={tok.url} target="_blank" rel="noreferrer">
          {tok.url}
        </a>,
      );
    } else {
      nodes.push(
        <span key={`${key}-t${i}`}>{inline(escapeHtml(tok.text), key)}</span>,
      );
    }
  });
  return nodes;
}

/** Block-level rendering with safe text (HTML already escaped). */
export default function RichText({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let listKind: "ul" | "ol" | null = null;
  let items: Array<{ key: string; node: ReactNode }> = [];

  const flushList = () => {
    if (listKind === null || items.length === 0) return;
    const Tag = listKind;
    blocks.push(
      <Tag key={`list-${blocks.length}`}>
        {items.map((it) => (
          <li key={it.key}>{it.node}</li>
        ))}
      </Tag>,
    );
    items = [];
    listKind = null;
  };

  text.split("\n").forEach((raw, idx) => {
    const bullet = /^\s*[-*•]\s+(.*)$/.exec(raw);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(raw);
    const listMatch = bullet ?? numbered;
    if (listMatch) {
      const kind = bullet ? "ul" : "ol";
      if (listKind !== null && listKind !== kind) flushList();
      listKind = kind;
      items.push({
        key: `it${idx}`,
        node: <>{line(listMatch[1].trim(), `li${idx}`)}</>,
      });
      return;
    }
    flushList();
    const trimmed = raw.trim();
    if (!trimmed) return;
    const heading = /^(#{1,4})\s+(.*)$/.exec(trimmed);
    if (heading) {
      const level = heading[1].length;
      const content = heading[2];
      if (level <= 2) {
        blocks.push(<h3 key={`h${idx}`}>{line(content, `h${idx}`)}</h3>);
      } else if (level === 3) {
        blocks.push(<h4 key={`h${idx}`}>{line(content, `h${idx}`)}</h4>);
      } else {
        blocks.push(<h5 key={`h${idx}`}>{line(content, `h${idx}`)}</h5>);
      }
      return;
    }
    blocks.push(<p key={`p${idx}`}>{line(trimmed, `p${idx}`)}</p>);
  });
  flushList();

  return <>{blocks}</>;
}