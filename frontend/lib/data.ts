import fs from "fs/promises";
import path from "path";

export const DATA_DIR = process.env.GLASS_DATA_DIR
  ? path.resolve(process.env.GLASS_DATA_DIR)
  : path.resolve(process.cwd(), "..", "data");

const BRAND_PATTERN = /^Brandt(\d+)_(\d{2})_(\d{2})_(\d{4})$/;

export function safeBrand(brand: string) {
  if (!BRAND_PATTERN.test(brand)) throw new Error("Invalid brand.");
  return brand;
}

export async function listBrands() {
  const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });

  return entries
    .filter((e) => e.isDirectory())
    .map((e) => {
      const m = e.name.match(BRAND_PATTERN);
      if (!m) return null;
      const [, n, d, mo, y] = m;
      return {
        folder: e.name,
        label: `Brand ${n} — ${d}.${mo}.${y}`,
        date: `${y}-${mo}-${d}`
      };
    })
    .filter((x): x is NonNullable<typeof x> => x !== null)
    .sort((a, b) => b.date.localeCompare(a.date));
}

export function resolveBrandPath(brand: string) {
  safeBrand(brand);
  const p = path.resolve(DATA_DIR, brand);
  const rel = path.relative(DATA_DIR, p);
  if (rel.startsWith("..") || path.isAbsolute(rel)) throw new Error("Invalid path.");
  return p;
}

export function resolveDataFile(brand: string, parts: string[]) {
  const base = resolveBrandPath(brand);
  const p = path.resolve(base, ...parts.map(decodeURIComponent));
  const rel = path.relative(base, p);
  if (rel.startsWith("..") || path.isAbsolute(rel)) throw new Error("Invalid file path.");
  return p;
}
