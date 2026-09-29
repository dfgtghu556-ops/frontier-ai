import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import {
  Cell,
  Chip,
  DataTable,
  PageHeader,
  Panel,
  RepoLink,
  Row,
  StatusChip,
} from "@/components/lab/primitives";
import { lab, recordUrl, repoFile } from "@/lib/lab/state";
import type { ExperimentCategory } from "@/lib/lab/types";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experiments — Frontier AI Lab" },
      {
        name: "description",
        content: "Every experiment recorded in EXPERIMENTS.md, with its status.",
      },
    ],
  }),
  component: Experiments,
});

const FILTERS: ("all" | ExperimentCategory)[] = [
  "all",
  "complete",
  "in_progress",
  "planned",
  "abandoned",
];

function Experiments() {
  const [filter, setFilter] = useState<(typeof FILTERS)[number]>("all");
  const rows = [...lab.experiments]
    .reverse()
    .filter((e) => filter === "all" || e.category === filter);
  const missing = lab.experiment_numbers_not_in_log;

  return (
    <div className="space-y-3">
      <PageHeader
        title="Experiments"
        description="Parsed from EXPERIMENTS.md. Each entry is append-only, so the status shown is the last status line recorded for that experiment. Click an ID to read the full record."
      />
      <div className="flex flex-wrap gap-1.5">
        {FILTERS.map((f) => {
          const n =
            f === "all"
              ? lab.experiments.length
              : lab.experiments.filter((e) => e.category === f).length;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={
                "rounded-md px-2 py-1 font-mono text-[11px] " +
                (filter === f
                  ? "bg-accent-soft text-accent"
                  : "text-muted-foreground hover:bg-foreground/5")
              }
            >
              {f.replace("_", " ")} ({n})
            </button>
          );
        })}
      </div>
      <Panel>
        <DataTable columns={["ID", "Date", "Step", "Experiment", "Status", "Decisions"]}>
          {rows.map((e) => (
            <Row key={e.id}>
              <Cell mono>
                <RepoLink href={e.url}>{e.id}</RepoLink>
              </Cell>
              <Cell mono muted>
                {e.date ?? "—"}
              </Cell>
              <Cell mono muted>
                {e.step ?? "—"}
              </Cell>
              <Cell>
                <p className="font-medium">{e.title}</p>
                {e.summary ? <p className="mt-0.5 text-muted-foreground">{e.summary}</p> : null}
                <p className="mt-1 font-mono text-[10px] text-faint">Recorded status: {e.status}</p>
              </Cell>
              <Cell>
                <StatusChip status={e.category} />
              </Cell>
              <Cell>
                <div className="flex flex-wrap gap-1.5">
                  {e.decisions.map((d) => (
                    <RepoLink key={d} href={recordUrl(d)}>
                      {d}
                    </RepoLink>
                  ))}
                </div>
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
      {missing.length > 0 ? (
        <Panel title="Numbers not in EXPERIMENTS.md">
          <p className="text-[12px] text-muted-foreground">
            EXP-{String(missing[0]).padStart(3, "0")} to EXP-
            {String(missing[missing.length - 1]).padStart(3, "0")} <Chip>{missing.length}</Chip>{" "}
            were the tokenizer-corpus acquisition steps. They are recorded in the tokenizer corpus
            documents, not in EXPERIMENTS.md, so they are not listed above:{" "}
            <RepoLink href={repoFile("docs/tokenizer_corpus_stage_a.md")}>stage A</RepoLink>,{" "}
            <RepoLink href={repoFile("docs/tokenizer_corpus_stage_b_acquisition.md")}>
              stage B
            </RepoLink>
            , <RepoLink href={repoFile("docs/tokenizer_corpus_handover.md")}>handover</RepoLink>.
          </p>
        </Panel>
      ) : null}
    </div>
  );
}
