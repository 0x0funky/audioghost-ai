import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for Docker deployment
  output: "standalone",
  
  // Disable telemetry in production
  experimental: {
    // Optimize for Docker
  },
};

export default nextConfig;
