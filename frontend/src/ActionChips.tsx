/* Quick-reply chips: used for soft-escalation options, an empty-state's
 * starter prompts, and any future slot prompts. Clicking sends `value` as a
 * normal passenger message. */

import type { MessageAction } from "./types";

export default function ActionChips({
  actions,
  onSelect,
  className,
}: {
  actions: MessageAction[];
  onSelect: (value: string) => void;
  className?: string;
}) {
  if (actions.length === 0) return null;
  return (
    <div className={className ?? "actions"}>
      {actions.map((a, i) => (
        <button
          key={`${a.label}-${i}`}
          type="button"
          className="action-chip"
          onClick={() => onSelect(a.value)}
        >
          {a.label}
        </button>
      ))}
    </div>
  );
}
