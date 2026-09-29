import { Link } from "@tanstack/react-router";
import { Menu, X } from "lucide-react";
import { useState, type ReactNode } from "react";
import { ThemeToggle } from "./theme";
import { cn } from "@/lib/utils";
import { lab, short } from "@/lib/lab/state";

type NavItem = { to: string; label: string };
type NavGroup = { group: string; items: NavItem[] };

export const navigation: NavGroup[] = [
  { group: "Control", items: [{ to: "/", label: "Overview" }] },
  {
    group: "Science",
    items: [
      { to: "/roadmap", label: "Roadmap" },
      { to: "/experiments", label: "Experiments" },
      { to: "/decisions", label: "Decisions" },
      { to: "/evaluation", label: "Evaluation" },
    ],
  },
  {
    group: "Artifacts",
    items: [
      { to: "/models", label: "Models" },
      { to: "/datasets", label: "Data & tokenizer" },
    ],
  },
  {
    group: "Operations",
    items: [
      { to: "/compute", label: "Compute" },
      { to: "/documentation", label: "Documentation" },
    ],
  },
];

function NavList({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-col gap-5">
      {navigation.map((group) => (
        <div key={group.group}>
          <p className="px-2 text-[9px] font-medium uppercase tracking-[0.2em] text-faint">
            {group.group}
          </p>
          <ul className="mt-1.5 space-y-0.5">
            {group.items.map((item) => (
              <li key={item.to}>
                <Link
                  to={item.to}
                  onClick={onNavigate}
                  activeOptions={{ exact: item.to === "/" }}
                  activeProps={{
                    className: "bg-accent-soft text-accent",
                  }}
                  inactiveProps={{ className: "text-muted-foreground hover:bg-foreground/5" }}
                  className="block rounded-md px-2 py-1.5 text-[13px] font-medium transition-colors"
                >
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

function Brand() {
  return (
    <div className="flex items-center gap-2">
      <span className="grid size-7 place-items-center rounded-md bg-foreground font-mono text-[10px] font-semibold tracking-wider text-background">
        FA
      </span>
      <span className="text-sm font-semibold tracking-tight">Frontier AI Lab</span>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="min-h-screen">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-60 flex-col border-r border-sidebar-border bg-sidebar/70 px-3 py-4 backdrop-blur-xl lg:flex">
        <Brand />
        <div className="mt-6 flex-1 overflow-y-auto">
          <NavList />
        </div>
        <p className="px-2 pt-3 font-mono text-[9px] uppercase tracking-[0.2em] text-faint">
          Reads the repo · not the model
        </p>
      </aside>

      {/* Mobile drawer */}
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button
            aria-label="Close navigation"
            className="absolute inset-0 bg-foreground/20 backdrop-blur-sm"
            onClick={() => setOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-64 border-r border-sidebar-border bg-sidebar px-3 py-4">
            <div className="flex items-center justify-between">
              <Brand />
              <button
                aria-label="Close navigation"
                onClick={() => setOpen(false)}
                className="grid size-7 place-items-center rounded-md text-muted-foreground hover:bg-foreground/5"
              >
                <X className="size-4" />
              </button>
            </div>
            <div className="mt-6 overflow-y-auto">
              <NavList onNavigate={() => setOpen(false)} />
            </div>
          </div>
        </div>
      )}

      <div className="lg:pl-60">
        <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b border-border/70 bg-background/70 px-3 backdrop-blur-xl sm:px-5">
          <button
            aria-label="Open navigation"
            onClick={() => setOpen(true)}
            className="grid size-8 place-items-center rounded-md text-muted-foreground hover:bg-foreground/5 lg:hidden"
          >
            <Menu className="size-4" />
          </button>
          <div className="lg:hidden">
            <Brand />
          </div>
          <span
            title="Every number on these pages is read from files in the frontier-ai repository by scripts/export_lab_state.py. This is the digest of those files."
            className="ml-auto rounded-full bg-accent-soft px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-accent"
          >
            Repo snapshot · {short(lab.source_digest, 8)}
          </span>
          <span className="hidden rounded-full bg-foreground/5 px-2 py-0.5 font-mono text-[9px] font-medium uppercase tracking-wider text-muted-foreground sm:inline">
            Read-only
          </span>
          <ThemeToggle />
        </header>

        <main className={cn("mx-auto max-w-[1400px] px-3 pb-16 pt-5 sm:px-5")}>{children}</main>
      </div>
    </div>
  );
}
