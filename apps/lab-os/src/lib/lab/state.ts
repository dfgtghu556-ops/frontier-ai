/**
 * The single data source of the dashboard: the repository snapshot bundled at build time.
 *
 * To refresh it, run `python scripts/export_lab_state.py` in the frontier-ai repository root
 * (a test there fails while the committed snapshot is out of date).
 */

import raw from "@/data/lab_state.json";
import type { LabState } from "./types";

export const lab = raw as unknown as LabState;

if (lab.schema !== "frontier-lab-state-v1") {
  throw new Error(`unexpected lab_state.json schema: ${String(lab.schema)}`);
}

export const LANGUAGE_NAMES: Record<string, string> = {
  as: "Assamese",
  bn: "Bengali",
  en: "English",
  gu: "Gujarati",
  hi: "Hindi",
  kn: "Kannada",
  ml: "Malayalam",
  mr: "Marathi",
  or: "Odia",
  pa: "Punjabi",
  ta: "Tamil",
  te: "Telugu",
  ur: "Urdu",
};

export const languageName = (code: string) => LANGUAGE_NAMES[code] ?? code;

export const fmtInt = (n: number | null | undefined) =>
  n === null || n === undefined ? "—" : n.toLocaleString("en-IN");

export const fmtBpb = (n: number | null | undefined, digits = 4) =>
  n === null || n === undefined ? "—" : n.toFixed(digits);

export function fmtBytes(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} KB`;
  return `${n} B`;
}

export function fmtCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

export const short = (hash: string, n = 12) => hash.slice(0, n);

export const repoFile = (path: string) =>
  `${lab.repository.url}/blob/${lab.repository.branch}/${path}`;

/** The record (experiment or decision) with this id, if the snapshot has it. */
export function recordUrl(id: string): string | null {
  const exp = lab.experiments.find((e) => e.id === id);
  if (exp) return exp.url;
  const dec = lab.decisions.find((d) => d.id === id);
  return dec ? dec.url : null;
}
