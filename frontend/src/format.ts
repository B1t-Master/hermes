/* Small shared formatting helpers. */

const timeFmt = new Intl.DateTimeFormat([], { hour: "2-digit", minute: "2-digit" });

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