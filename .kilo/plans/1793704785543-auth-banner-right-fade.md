# Plan: Auth page right-side banner photo with fade-in

## Goal
Add the banner photo (`kenya_airways_web_1200w.webp`) to the login/signup page so it sits vertically on the right of the auth section, with a fade-in at the leading edge of the banner (and an entrance animation on load), matching the Dribbble "Airplanes booking app" example.

## Current state
- Auth page = `AuthGate` component in `frontend/src/App.tsx:18-126`, rendered as `<main className="auth">` (flex column, centered).
- `frontend/src/styles.css:1402-1422` defines `.auth` (position: relative, isolate) and an unused `.auth-decor` class already positioned absolute on the right edge (`top: 50%; right: max(1vw, 8px); translateY(-50%); rotate(6deg); z-index: 0; opacity: 0.7`).
- `frontend/src/assets/kenya_airways_web_1200w.webp` — banner photo already added by user.
- Asset reference convention: `Logo.tsx:4` uses `src="/src/assets/..."` absolute path.
- `styles.css:1414` `.auth-decor` is defined but never referenced in JSX.

## Decisions
1. **Placement:** Reuse the existing absolute-positioned decor pattern — place the banner on the right edge, vertically centered, behind the form (z-index 0). Least invasive; keeps the centered form intact.
2. **"Fade into at the beginning":** Two-part fade:
   - Entrance: `@keyframes banner-fade-in` (opacity 0→1, slight translate) applied to the banner on mount.
   - Leading edge: `mask-image: linear-gradient(to right, transparent, black)` (and `-webkit-` prefix) on the banner's left edge so it blends into the canvas.
3. **Asset path:** `/src/assets/kenya_airways_web_1200w.webp` (matches `Logo.tsx` convention).
4. **Responsive:** On narrow screens, reduce banner width / lower opacity so it doesn't cover the form (extend existing media query at `styles.css:1632`).

## Affected files
- `frontend/src/App.tsx` — add `<img>` banner element inside `<main className="auth">`.
- `frontend/src/styles.css` — add `.auth-banner` rules, `@keyframes banner-fade-in`, mask gradient, responsive override.

## Tasks
1. `frontend/src/App.tsx`: Inside `<main className="auth">`, add:
   `<img src="/src/assets/kenya_airways_web_1200w.webp" className="auth-banner" alt="" />`
   placed after `.auth-brand`/form so it paints behind (z-index 0) — position absolute handles layering regardless of DOM order.
2. `frontend/src/styles.css`: Add `.auth-banner`:
   - `position: absolute; top: 50%; right: max(1vw, 8px); transform: translateY(-50%) rotate(6deg);`
   - `width: min(360px, 38vw); height: auto; object-fit: contain;` (tall/vertical)
   - `z-index: 0; pointer-events: none; user-select: none;`
   - `animation: banner-fade-in 1s ease-out both;`
   - `mask-image: linear-gradient(to right, transparent 0%, black 55%); -webkit-mask-image: same;`
3. `frontend/src/styles.css`: Add `@keyframes banner-fade-in { from { opacity: 0; transform: translateY(-50%) translateX(8px) rotate(6deg); } to { opacity: 1; transform: translateY(-50%) rotate(6deg); } }`
4. `frontend/src/styles.css`: In the existing media query (`styles.css:1632`), tighten `.auth-banner` (smaller width, lower opacity) for mobile.
5. Run `npm run typecheck` (and `npm run build`) in `frontend/` to verify no breakage.

## Validation
- `npm run typecheck` passes.
- `body::before`/auth background unaffected; form still centered and clickable (banner behind, pointer-events none).
- Banner visible on right edge of auth page on desktop; scaled down on mobile.

## Open question
None blocked. The mask-gradient + entrance animation combo is the recommended interpretation of "fade into at the beginning"; if the user wants only one, drop the other.
