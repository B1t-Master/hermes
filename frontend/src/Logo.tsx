const Logo = ({ size = 40 }: { size?: number }) => {
  return (
    <img
      src="/src/assets/Kenya_Airways-Logo.svg"
      alt="Kenya Airways"
      width={size}
      height={size}
    />
  );
};

export default Logo;
