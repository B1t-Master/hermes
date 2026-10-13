import markUrl from "./assets/hermes.svg";

/** Hermes brand mark shown beside the "hermes" wordmark (headers + bot meta). */
export default function Wings({ size = 22 }: { size?: number }) {
  return (
    <img
      className="wings"
      src={markUrl}
      width={size}
      height={size}
      alt=""
      aria-hidden="true"
    />
  );
}
