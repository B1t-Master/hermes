/* Memphis decoration: hand-drawn shapes, no icon library.
 *
 * Everything here is plain SVG — blobs, squiggles, sparkles and rings drawn
 * with the same 3px ink stroke as the rest of the interface. */

export default function Memphis({
  className = "memphis",
  variant = "default",
}: {
  className?: string;
  variant?: "default" | "chat";
}) {
  if (variant === "chat") {
    return (
      <svg
        className={className}
        viewBox="0 0 160 120"
        aria-hidden="true"
        focusable="false"
      >
        {/* squiggle */}
        <path
          d="M8 92c14-22 28 14 42-8s28 14 42-8 28 14 42-8"
          fill="none"
          stroke="currentColor"
          strokeWidth="5"
          strokeLinecap="round"
        />
        {/* sparkle */}
        <path
          d="M120 20c3 12 10 19 22 22-12 3-19 10-22 22-3-12-10-19-22-22 12-3 19-10 22-22Z"
          fill="currentColor"
        />
        {/* ring */}
        <circle
          cx="26"
          cy="26"
          r="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="6"
        />
        {/* triangle */}
        <path d="M92 96 116 56l24 40Z" fill="currentColor" opacity="0.35" />
      </svg>
    );
  }

  return (
    <svg
      className={className}
      viewBox="0 0 320 320"
      aria-hidden="true"
      focusable="false"
    >
      {/* big pastel blob */}
      <path
        d="M62 44c34-26 92-30 128-2s30 84 6 118-88 46-126 20S28 70 62 44Z"
        fill="var(--teal)"
        stroke="var(--ink)"
        strokeWidth="6"
      />
      {/* squiggle */}
      <path
        d="M22 232c16-26 32 18 48-8s32 18 48-8 32 18 48-8"
        fill="none"
        stroke="var(--ink)"
        strokeWidth="7"
        strokeLinecap="round"
      />
      {/* sparkle */}
      <path
        d="M244 58c5 20 17 32 37 37-20 5-32 17-37 37-5-20-17-32-37-37 20-5 32-17 37-37Z"
        fill="var(--yellow)"
        stroke="var(--ink)"
        strokeWidth="6"
        strokeLinejoin="round"
      />
      {/* ring */}
      <circle
        cx="268"
        cy="188"
        r="26"
        fill="none"
        stroke="var(--ink)"
        strokeWidth="8"
      />
      {/* triangle */}
      <path
        d="M46 158 86 106l40 52Z"
        fill="var(--pink)"
        stroke="var(--ink)"
        strokeWidth="6"
        strokeLinejoin="round"
      />
      {/* dots */}
      <circle cx="150" cy="286" r="9" fill="var(--ink)" />
      <circle cx="182" cy="276" r="6" fill="var(--red)" />
      <circle cx="208" cy="290" r="5" fill="var(--ink)" />
    </svg>
  );
}
