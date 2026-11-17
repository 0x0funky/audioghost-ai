import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // start.bat / start.sh run `next dev`: hide the dev-tools badge, which
  // otherwise sits on top of the mix bar.
  devIndicators: false,
};

export default nextConfig;
