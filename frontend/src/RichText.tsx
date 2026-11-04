/* Minimal, dependency-free rich-text renderer for bot/agent messages.

 * The LLM answers in markdown; rendering raw strings makes bold markers,
 * headings and bullets visible as literal characters. This renderer is a
 * tiny safe subset: it applies **bold**, *italic*, `code`, auto-linked
 * URLs, # headings, and bullet/numbered lists.
 *
 * Safety note: every piece of text ends up in a React text node, and React
 * escapes text children for us. The text must therefore NOT be escaped a
 * second time here — doing that printed apostrophes as the literal
 * "&#39;" in the chat bubble.
 */

import type { ReactNode } from "react";

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

/** Inline markers: **b**, *i*, `code`. */
function inline(text: string, prefix: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) nodes.push(text.slice(last, m.index));
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
  if (last < text.length) nodes.push(text.slice(last));
  return nodes;
}

/** One line -> nodes (autolinks + inline markers). */
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
      nodes.push(<span key={`${key}-t${i}`}>{inline(tok.text, key)}</span>);
    }
  });
  return nodes;
}

/** "| a | b |" -> ["a", "b"] (leading/trailing pipes dropped, cells trimmed). */
function tableCells(raw: string): string[] {
  return raw
    .replace(/^\s*\|/, "")
    .replace(/\|\s*$/, "")
    .split("|")
    .map((c) => c.trim());
}

/** A markdown separator row: every cell is only dashes (with optional colons). */
function isSeparatorRow(cells: string[]): boolean {
  return cells.length > 0 && cells.every((c) => /^:?-{2,}:?$/.test(c));
}

/** A full table row: starts and ends with a pipe and has content between. */
const TABLE_ROW_RE = /^\s*\|.+\|\s*$/;

/** Block-level rendering: paragraphs, headings, lists and tables. */
export default function RichText({ text }: { text: string }) {
  const blocks: ReactNode[] = [];
  let listKind: "ul" | "ol" | null = null;
  let items: Array<{ key: string; node: ReactNode }> = [];
  let tableLines: string[] = [];

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

  const flushTable = () => {
    if (tableLines.length === 0) return;
    const rows = tableLines
      .map(tableCells)
      .filter((cells) => !isSeparatorRow(cells));
    tableLines = [];
    if (rows.length === 0) return;
    const [header, ...body] = rows;
    const tid = `tbl-${blocks.length}`;
    blocks.push(
      <div className="msg-table-wrap" key={tid}>
        <table className="msg-table">
          <thead>
            <tr>
              {header.map((cell, i) => (
                <th key={`${tid}-h${i}`}>{line(cell, `${tid}-h${i}`)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {body.map((row, r) => (
              <tr key={`${tid}-r${r}`}>
                {row.map((cell, i) => (
                  <td key={`${tid}-r${r}c${i}`}>{line(cell, `${tid}-r${r}c${i}`)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>,
    );
  };

  text.split("\n").forEach((raw, idx) => {
    if (TABLE_ROW_RE.test(raw)) {
      flushList();
      tableLines.push(raw);
      return;
    }
    flushTable();

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
  flushTable();

  return <>{blocks}</>;
}