# Plan: Paw-print plane-wing background overlay

## Goal
Replace the uniform repeating `plane-wing.svg` CSS background on `body::before` with a paw-print-style overlay: many instances of `plane-wing.svg` at varying sizes, rotations, positions, and opacities, tiled as a subtle background.

## Current state
- `frontend/src/assets/plane-wing.svg` — existing plane wing asset (800x800 viewBox).
- `frontend/src/styles.css:80-89` — `body::before` uses `background: url("./assets/plane-wing.svg") 60px 40px / 420px 420px repeat;` with an undefined `wing-fly` animation (no `@keyframes wing-fly` exists).
- `frontend/src/Wings.tsx` — renders `hermes.svg` brand mark beside wordmark (unrelated to background; leave untouched).
- I previously created `frontend/src/assets/paw-print.svg` (paw-shaped symbol) and `frontend/src/assets/paw-pattern.svg` (paw-shaped pattern). Per user clarification, the overlay must use `plane-wing.svg`, so those paw-shaped files are superseded and should be removed.

## Design decisions
1. **Element source:** Use `plane-wing.svg` path data as the repeating stamp, NOT paw shapes.
2. **Variation:** Each stamp gets random position, scale (0.35–1.7), rotation (0–360°), fill-opacity (0.03–0.12), deterministic via seeded PRNG so output is stable across regenerations.
3. **Layout:** Generate a single tileable SVG (e.g. 800×800 viewBox) containing ~40–60 transformed `<use>`/`<path>` instances of the wing, arranged in a jittered grid so edges tile cleanly. Use this as `body::before background: url("./assets/paw-plane-pattern.svg") repeat`.
4. **Animation:** Remove the broken `wing-fly` reference (undefined keyframe). Keep the layer static at low opacity; optionally add a very slow subtle drift later if desired.
5. **Cleanup:** Delete `paw-print.svg` and `paw-pattern.svg` (paw-shaped, not requested).

## Tasks
1. Write a Node generator script (in /tmp, not committed) that reads `plane-wing.svg` path data and emits `frontend/src/assets/paw-plane-pattern.svg` with varied transform/opacity instances.
2. Run the generator to produce `paw-plane-pattern.svg`.
3. Edit `frontend/src/styles.css:87-88`: point `body::before` background at `./assets/paw-plane-pattern.svg`, drop `animation: wing-fly ...`.
4. Delete `frontend/src/assets/paw-print.svg` and `frontend/src/assets/paw-pattern.svg`.
5. Run `npm run typecheck` (and `npm run build` if available) in `frontend/` to verify no breakage.
6. Visually confirm the background shows varied plane wings at low opacity.

## Risks / open questions
- Large inline SVG as CSS background increases CSS size modestly; acceptable for a decorative backdrop. If too large, fallback to a React component rendering positioned instances instead of CSS background.
- `wing-fly` animation is undefined; confirm user does not want a custom animation before adding one (recommended: leave static).

## Validation
- `npm run typecheck` passes.
- `paw-plane-pattern.svg` exists and references plane-wing geometry.
- `body::before` no longer references the undefined `wing-fly` animation.
