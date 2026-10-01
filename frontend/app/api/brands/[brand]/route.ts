import fs from "fs/promises";
import path from "path";
import { NextResponse } from "next/server";
import { resolveBrandPath } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _request: Request,
  context: { params: Promise<{ brand: string }> }
) {
  try {
    const { brand } = await context.params;
    const metadataPath = path.join(
      resolveBrandPath(decodeURIComponent(brand)),
      "metadata.json"
    );
    const raw = await fs.readFile(metadataPath, "utf8");
    return NextResponse.json(JSON.parse(raw));
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Could not load metadata." }, { status: 404 });
  }
}
