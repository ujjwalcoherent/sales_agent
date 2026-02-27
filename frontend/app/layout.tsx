import type { Metadata } from "next";
import { DM_Serif_Display, DM_Sans } from "next/font/google";
import "./globals.css";

const dmSerifDisplay = DM_Serif_Display({
  variable: "--font-display",
  subsets: ["latin"],
  weight: "400",
});

const dmSans = DM_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Sales Intelligence",
  description: "AI-powered market signal to lead pipeline",
};

// Inline script that restores dark mode from localStorage BEFORE paint.
// Runs synchronously to prevent flash-of-wrong-theme.
const themeScript = `(function(){try{if(localStorage.getItem("harbinger_dark")==="true")document.documentElement.classList.add("dark")}catch(e){}})()`;

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body className={`${dmSerifDisplay.variable} ${dmSans.variable} antialiased`}>
        {children}
      </body>
    </html>
  );
}
