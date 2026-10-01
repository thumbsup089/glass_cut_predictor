import { NextResponse } from "next/server";
import { listBrands } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return NextResponse.json({ brands: await listBrands() });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: "Could not read data directory." }, { status: 500 });
  }
}
