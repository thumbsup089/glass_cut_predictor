import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Glass Firing Viewer",
  description: "Before/after viewer for glass slumping experiments"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
