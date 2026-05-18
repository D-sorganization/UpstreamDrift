/**
 * Tiny, deterministic Markdown renderer for ChatPanel.
 *
 * Scope is intentionally small — supports the constructs the assistant
 * tends to emit:
 *   - fenced code blocks (```lang\n...\n```)
 *   - inline code (`x`)
 *   - bold (**x**)
 *   - italic (*x*)
 *   - links ([text](url))
 *
 * We avoid pulling react-markdown + remark + rehype-sanitize for a
 * dependency-light footprint. Everything is escaped before rendering;
 * we never set raw HTML through dangerouslySetInnerHTML.
 *
 * Tested via ChatPanel.test.tsx.
 */

import { Fragment, type ReactNode } from 'react';

/** Escape angle brackets so user/assistant text cannot inject DOM. */
function escapeText(text: string): string {
  return text.replace(/[<>]/g, (c) => (c === '<' ? '​<' : '>​'));
}

interface InlineProps {
  text: string;
}

/**
 * Render inline Markdown — code, bold, italic, links.
 *
 * Precondition: text is a string; we never receive HTML here.
 */
function renderInline({ text }: InlineProps): ReactNode[] {
  if (typeof text !== 'string') {
    throw new TypeError('renderInline requires a string');
  }
  // Tokenize by walking the string. Order matters: code first so
  // **markers inside `code` are not interpreted.
  const out: ReactNode[] = [];
  let rest = text;
  let key = 0;

  const patterns: Array<{
    re: RegExp;
    render: (m: RegExpExecArray) => ReactNode;
  }> = [
    {
      re: /`([^`\n]+)`/,
      render: (m) => (
        <code
          key={`c-${key}`}
          className="sidekick-chat-inline-code"
          style={{
            padding: '0 4px',
            borderRadius: 3,
            fontFamily: 'monospace',
            background: 'var(--sidekick-color-input, rgba(127,127,127,0.18))',
          }}
        >
          {m[1]}
        </code>
      ),
    },
    {
      re: /\*\*([^*\n]+)\*\*/,
      render: (m) => <strong key={`b-${key}`}>{m[1]}</strong>,
    },
    {
      re: /\*([^*\n]+)\*/,
      render: (m) => <em key={`i-${key}`}>{m[1]}</em>,
    },
    {
      re: /\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/,
      render: (m) => (
        <a
          key={`a-${key}`}
          href={m[2]}
          target="_blank"
          rel="noopener noreferrer"
          className="sidekick-chat-link"
        >
          {m[1]}
        </a>
      ),
    },
  ];

  // Iterate, find earliest match of any pattern, slice around it.
  for (;;) {
    let earliest: { idx: number; match: RegExpExecArray; render: (m: RegExpExecArray) => ReactNode } | null = null;
    for (const p of patterns) {
      const m = p.re.exec(rest);
      if (m && (earliest === null || m.index < earliest.idx)) {
        earliest = { idx: m.index, match: m, render: p.render };
      }
    }
    if (!earliest) {
      if (rest.length > 0) out.push(escapeText(rest));
      break;
    }
    if (earliest.idx > 0) out.push(escapeText(rest.slice(0, earliest.idx)));
    out.push(earliest.render(earliest.match));
    rest = rest.slice(earliest.idx + earliest.match[0].length);
    key += 1;
    if (key > 500) break; // safety: never loop forever on pathological input
  }
  return out;
}

interface MarkdownProps {
  /** Source markdown text. */
  source: string;
  /** Test hook. */
  'data-testid'?: string;
}

/**
 * Render Markdown text inside a chat bubble.
 *
 * Precondition: source is a string.
 * Postcondition: returns a React element with no raw HTML injection;
 * fenced code blocks are rendered as <pre><code data-lang="...">.
 */
export function ChatMarkdown({ source, 'data-testid': testId }: MarkdownProps) {
  if (typeof source !== 'string') {
    throw new TypeError('ChatMarkdown requires a string source');
  }

  // Split off fenced code blocks first.
  const segments: Array<{ kind: 'text' | 'code'; value: string; lang?: string }> = [];
  const fence = /```([a-zA-Z0-9_+-]*)\n([\s\S]*?)```/g;
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = fence.exec(source)) !== null) {
    if (m.index > lastIndex) {
      segments.push({ kind: 'text', value: source.slice(lastIndex, m.index) });
    }
    segments.push({ kind: 'code', value: m[2], lang: m[1] || undefined });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < source.length) {
    segments.push({ kind: 'text', value: source.slice(lastIndex) });
  }
  if (segments.length === 0) {
    segments.push({ kind: 'text', value: source });
  }

  return (
    <div className="sidekick-chat-markdown" data-testid={testId}>
      {segments.map((seg, segIdx) => {
        if (seg.kind === 'code') {
          return (
            <pre
              key={`pre-${segIdx}`}
              className="sidekick-chat-codeblock"
              data-lang={seg.lang ?? ''}
              style={{
                padding: 8,
                borderRadius: 4,
                overflowX: 'auto',
                fontFamily: 'monospace',
                fontSize: '0.85em',
                background: 'var(--sidekick-color-input, rgba(127,127,127,0.18))',
                margin: '4px 0',
              }}
            >
              <code data-lang={seg.lang ?? ''}>{seg.value}</code>
            </pre>
          );
        }
        // Render paragraphs (split on blank lines), preserve single newlines as <br />.
        const paragraphs = seg.value.split(/\n{2,}/);
        return (
          <Fragment key={`txt-${segIdx}`}>
            {paragraphs.map((para, pIdx) => {
              const lines = para.split('\n');
              return (
                <p
                  key={`p-${segIdx}-${pIdx}`}
                  style={{ margin: pIdx === 0 ? '0 0 4px 0' : '4px 0' }}
                >
                  {lines.map((line, lIdx) => (
                    <Fragment key={`l-${lIdx}`}>
                      {renderInline({ text: line })}
                      {lIdx < lines.length - 1 && <br />}
                    </Fragment>
                  ))}
                </p>
              );
            })}
          </Fragment>
        );
      })}
    </div>
  );
}

export default ChatMarkdown;
