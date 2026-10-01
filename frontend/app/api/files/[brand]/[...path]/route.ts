import fs from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import { resolveDataFile } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TYPES: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".csv": "text/csv; charset=utf-8",
  ".stl": "model/stl"
};

export async function GET(
  _request: Request,
  context: { params: Promise<{ brand: string; path: string[] }> }
) {
  try {
    const params = await context.params;
    const filePath = resolveDataFile(
      decodeURIComponent(params.brand),
      params.path
    );
    const data = await fs.readFile(filePath);
    const type = TYPES[path.extname(filePath).toLowerCase()] ?? "application/octet-stream";
    return new NextResponse(data, {
      headers: {
        "Content-Type": type,
        "Cache-Control": "public, max-age=3600"
      }
    });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "File not found." }, { status: 404 });
  }
}
