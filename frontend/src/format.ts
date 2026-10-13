/* Small shared formatting helpers. */

const timeFmt = new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" });
const dateFmt = new Intl.DateTimeFormat([], { year: "numeric", month: "short", day: "numeric" });

/** Defensive HH:MM formatter: never crashes on odd/empty ISO strings. */
export function formatTime(iso: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : timeFmt.format(d);
  } catch {
    return iso;
  }
}

/** Defensive "12 Mar 2026" formatter for source dates. */
export function formatDate(iso: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : dateFmt.format(d);
  } catch {
    return iso;
  }
}