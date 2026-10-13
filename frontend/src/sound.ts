/* Tiny synthesized audio cues (no asset files).
 *
 * Browsers block audio until a user gesture has happened; callers treat this
 * as best-effort feedback, so every failure is swallowed. The AudioContext is
 * created lazily and resumed on each call (a gesture has usually occurred by
 * the time an escalation is triggered). */

interface WindowWithWebkit extends Window {
  webkitAudioContext?: typeof AudioContext;
}

let ctx: AudioContext | null = null;

function audioContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor = window.AudioContext ?? (window as WindowWithWebkit).webkitAudioContext;
  if (!Ctor) return null;
  ctx ??= new Ctor();
  return ctx;
}

function tone(freq: number, start: number, duration: number, gainValue: number): void {
  const audio = ctx;
  if (!audio) return;
  const osc = audio.createOscillator();
  const gain = audio.createGain();
  osc.type = "sine";
  osc.frequency.value = freq;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(gainValue, start + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  osc.connect(gain).connect(audio.destination);
  osc.start(start);
  osc.stop(start + duration + 0.02);
}

/** Two-note "handoff" chime for escalation to a human agent. */
export function playEscalationCue(): void {
  try {
    const audio = audioContext();
    if (!audio) return;
    void audio.resume();
    const now = audio.currentTime;
    tone(880, now, 0.18, 0.16);
    tone(1174.66, now + 0.14, 0.22, 0.14);
  } catch {
    /* autoplay blocked or audio unavailable: feedback is non-critical */
  }
}

/** Soft single note when a human agent joins an escalated chat. */
export function playAgentJoinedCue(): void {
  try {
    const audio = audioContext();
    if (!audio) return;
    void audio.resume();
    tone(659.25, audio.currentTime, 0.2, 0.12);
  } catch {
    /* non-critical */
  }
}
