/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  },
  // Optimize for production
  swcMinify: true,
  // Handle large files
  experimental: {
    largePageDataBytes: 128 * 1000, // 128KB
  },
};

export default nextConfig;
