import type { Metadata } from "next";
import { Inter, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const spaceGrotesk = Space_Grotesk({ subsets: ["latin"], weight: ["500", "600"], variable: "--font-space-grotesk" });

export const metadata: Metadata = {
  title: "AudioGhost AI - Pull any sound out of audio or video",
  description:
    "Describe the sounds you want (vocals, a dog barking, traffic) and AudioGhost separates each onto its own track, on your own GPU. Mix them and export what you hear.",
  keywords: ["audio separation", "stem separation", "vocal removal", "SAM-Audio", "AI"],
};

// Applies the saved theme before the first paint (no flash of the wrong theme).
const themeScript = `try{var t=localStorage.getItem("audioghost.theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${spaceGrotesk.variable}`}>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
