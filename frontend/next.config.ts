import type { NextConfig } from "next";

// `npm run build:static` (AUDIOGHOST_STATIC_EXPORT=1): a static export that the
// backend serves itself, for the desktop app. Images are served as they are.
const staticExport = process.env.AUDIOGHOST_STATIC_EXPORT === "1";

const nextConfig: NextConfig = {
  // start.bat / start.sh run `next dev`: hide the dev-tools badge, which
  // otherwise sits on top of the mix bar.
  devIndicators: false,
  ...(staticExport ? { output: "export" as const, images: { unoptimized: true } } : {}),
};

export default nextConfig;
