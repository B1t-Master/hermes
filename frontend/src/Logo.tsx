/* hermes plane mark: a brutalist paper-plane in the KQ palette.
 * Placeholder brand asset for the auth/boot screen and headers. */

export default function Logo({ size = 72 }: { size?: number }) {
  return (
    <svg
      className="logo-mark"
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role="img"
      aria-label="hermes plane logo"
    >
      <rect x="2" y="2" width="60" height="60" fill="#e31b23" stroke="#000" strokeWidth="3" />
      <path
        d="M10 38 L54 9 L39 55 L30 39 Z"
        fill="#fff"
        stroke="#000"
        strokeWidth="3"
        strokeLinejoin="round"
      />
      <path d="M30 39 L54 9" stroke="#000" strokeWidth="3" />
      <circle cx="14" cy="14" r="3" fill="#d4af37" stroke="#000" strokeWidth="2" />
    </svg>
  );
}
