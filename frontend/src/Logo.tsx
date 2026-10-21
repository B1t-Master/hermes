/* hermes brand mark: a rounded tile with the KQ-red paper plane.
 * Placeholder brand asset for the auth/boot screen. */

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
      <defs>
        <linearGradient id="hermes-red" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#f0424a" />
          <stop offset="1" stopColor="#c4161d" />
        </linearGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="16" fill="url(#hermes-red)" />
      <path d="M12 35.5 52 12 39.5 52 30.5 38.5 Z" fill="#fff" />
      <path
        d="M30.5 38.5 52 12"
        stroke="#f0424a"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
      <circle cx="15" cy="15" r="3" fill="#d4af37" />
    </svg>
  );
}
