import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Frosted instrument panel — the base surface of every module. */
export function Panel({
  title,
  meta,
  action,
  children,
  className,
}: {
  title?: string;
  meta?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("panel p-4", className)}>
      {(title || meta || action) && (
        <header className="mb-3 flex items-center justify-between gap-3">
          {title ? <h2 className="label-xs">{title}</h2> : <span />}
          <div className="flex items-center gap-2">
            {meta ? <span className="font-mono text-[10px] text-faint">{meta}</span> : null}
            {action}
          </div>
        </header>
      )}
      {children}
    </section>
  );
}

/** Renders a value, or an explicit placeholder when there is no real data. */
export function Value({
  value,
  fallback = "Not available",
  mono = true,
}: {
  value: string | number | null | undefined;
  fallback?: string;
  mono?: boolean;
}) {
  const empty = value === null || value === undefined || value === "";
  return (
    <span
      className={cn(mono && "font-mono", "text-[13px]", empty ? "text-faint" : "text-foreground")}
    >
      {empty ? fallback : value}
    </span>
  );
}

export function Field({
  label,
  value,
  fallback,
  children,
}: {
  label: string;
  value?: string | number | null;
  fallback?: string;
  children?: ReactNode;
}) {
  return (
    <div className="min-w-0">
      <p className="text-[10px] uppercase tracking-wider text-faint">{label}</p>
      <div className="mt-1 break-words text-[13px] font-medium">
        {children ?? <Value value={value ?? null} fallback={fallback ?? "—"} />}
      </div>
    </div>
  );
}

type Tone = "accent" | "neutral" | "warn" | "fail" | "ok";

const toneClass: Record<Tone, string> = {
  accent: "bg-accent-soft text-accent",
  neutral: "bg-foreground/5 text-muted-foreground",
  warn: "bg-state-warn/12 text-state-warn",
  fail: "bg-state-fail/12 text-state-fail",
  ok: "bg-state-ok/12 text-state-ok",
};

export function Chip({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center rounded-full px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider",
        toneClass[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const statusTone: Record<string, Tone> = {
  // experiments / roadmap
  complete: "ok",
  in_progress: "warn",
  current: "accent",
  next: "accent",
  planned: "neutral",
  not_started: "neutral",
  abandoned: "fail",
  // decisions
  accepted: "ok",
  provisional: "warn",
  proposed: "warn",
  superseded: "neutral",
  // compute / artifacts
  in_use: "accent",
  frozen: "ok",
  protected: "ok",
  not_available_yet: "warn",
  CHECKED: "ok",
};

export function StatusChip({ status }: { status: string }) {
  return <Chip tone={statusTone[status] ?? "neutral"}>{status.replace(/_/g, " ")}</Chip>;
}

/** A metric tile that states plainly when no backend is connected. */
export function Metric({
  label,
  value,
  note,
  emphasis = false,
}: {
  label: string;
  value: string | number | null;
  note?: string;
  emphasis?: boolean;
}) {
  const empty = value === null || value === undefined;
  return (
    <div className="panel p-3">
      <p className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </p>
      <p
        className={cn(
          "mt-1 font-mono text-lg font-semibold tracking-tight",
          empty ? "text-faint" : emphasis ? "text-accent" : "text-foreground",
        )}
      >
        {empty ? "—" : value}
      </p>
      {note ? <p className="font-mono text-[10px] text-faint">{note}</p> : null}
    </div>
  );
}

export function PlaceholderNote({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-[10px] border border-dashed border-line/80 px-3 py-5 text-center">
      <p className="font-mono text-[11px] text-muted-foreground">{children}</p>
    </div>
  );
}

export function PageHeader({
  title,
  description,
  meta,
}: {
  title: string;
  description: string;
  meta?: ReactNode;
}) {
  return (
    <header className="mb-4">
      <div className="flex flex-wrap items-center gap-2">
        <h1 className="text-lg font-semibold tracking-tight">{title}</h1>
        {meta}
      </div>
      <p className="mt-1 max-w-3xl text-[13px] text-muted-foreground">{description}</p>
    </header>
  );
}

/** Hairline data table used across every registry module. */
export function DataTable({ columns, children }: { columns: string[]; children: ReactNode }) {
  return (
    <div className="-mx-1 overflow-x-auto px-1">
      <table className="w-full min-w-[640px] border-collapse text-left">
        <thead>
          <tr>
            {columns.map((c) => (
              <th
                key={c}
                className="hairline pb-2 pr-4 text-[10px] font-medium uppercase tracking-wider text-faint"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({ children }: { children: ReactNode }) {
  return <tr className="hairline transition-colors hover:bg-foreground/[0.03]">{children}</tr>;
}

export function Cell({
  children,
  mono = false,
  muted = false,
}: {
  children: ReactNode;
  mono?: boolean;
  muted?: boolean;
}) {
  return (
    <td
      className={cn(
        "py-2.5 pr-4 align-top text-[12px]",
        mono && "font-mono text-[11px]",
        muted && "text-muted-foreground",
      )}
    >
      {children}
    </td>
  );
}

export function Dash() {
  return <span className="text-faint">—</span>;
}

/** Link to a record or file in the frontier-ai repository (opens GitHub in a new tab). */
export function RepoLink({ href, children }: { href: string | null; children: ReactNode }) {
  if (!href) return <span className="font-mono text-[11px]">{children}</span>;
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="font-mono text-[11px] text-accent underline-offset-2 hover:underline"
    >
      {children}
    </a>
  );
}
