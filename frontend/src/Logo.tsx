import logoUrl from "./assets/Kenya_Airways-Logo.svg";

const Logo = ({ size = 40 }: { size?: number }) => {
  return (
    <img
      src={logoUrl}
      alt="Kenya Airways"
      width={size}
      height={size}
    />
  );
};

export default Logo;
