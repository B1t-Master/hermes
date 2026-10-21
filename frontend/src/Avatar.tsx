import Wings from "./Wings";

/** Monogram avatar for a message sender (bot shows the brand mark). */
export default function Avatar({
  name,
  kind,
}: {
  name: string;
  kind: "bot" | "passenger" | "agent";
}) {
  return (
    <span className={`avatar ${kind}`} aria-hidden="true">
      {kind === "bot" ? (
        <Wings size={18} />
      ) : (
        name.trim().charAt(0).toUpperCase() || "?"
      )}
    </span>
  );
}
