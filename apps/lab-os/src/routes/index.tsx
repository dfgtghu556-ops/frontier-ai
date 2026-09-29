import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Cell,
  DataTable,
  Field,
  Metric,
  Panel,
  RepoLink,
  Row,
  StatusChip,
} from "@/components/lab/primitives";
import { fmtBpb, fmtInt, lab, recordUrl } from "@/lib/lab/state";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Overview — Frontier AI Lab" },
      {
        name: "description",
        content: "Where the frontier-ai project stands, read directly from the repository.",
      },
    ],
  }),
  component: Overview,
});

function Overview() {
  const done = lab.roadmap.filter((s) => s.status === "complete").length;
  const next = lab.roadmap.find((s) => s.status === "next" || s.status === "current");
  const expDone = lab.experiments.filter((e) => e.category === "complete").length;
  const accepted = lab.decisions.filter((d) => d.category === "accepted").length;
  const base = lab.baseline_model;
  const largest = Math.max(...lab.models.map((m) => m.n_params));
  const recent = [...lab.experiments].slice(-6).reverse();
  const c = lab.corpus_v1;

  return (
    <div className="space-y-3">
      <Panel>
        <p className="label-xs">Goal</p>
        <p className="mt-1 max-w-3xl text-[13px]">
          Build a language model from scratch that can compete with the best models and become the
          best model for India. This page shows where the project really is today. Every number
          comes from a file in the repository.
        </p>
      </Panel>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Roadmap steps done" value={`${done} / ${lab.roadmap.length}`} emphasis />
        <Metric
          label="Experiments complete"
          value={`${expDone} / ${lab.experiments.length}`}
          note="in EXPERIMENTS.md"
        />
        <Metric
          label="Decisions accepted"
          value={`${accepted} / ${lab.decisions.length}`}
          note="in DECISIONS.md"
        />
        <Metric
          label="Baseline bits/byte"
          value={base ? fmtBpb(base.record.bpb_mean) : null}
          note={
            base
              ? `${base.model} · ${base.record.seeds} seeds · ${base.evidence.join(", ")} · lower is better`
              : "no baseline recorded"
          }
        />
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <Panel title="Working on now">
          <ul className="space-y-3">
            {lab.current_work.map((w) => (
              <li key={w.title}>
                <p className="text-[13px] font-medium">{w.title}</p>
                <p className="mt-0.5 text-[12px] text-muted-foreground">{w.note}</p>
                <div className="mt-1 flex flex-wrap gap-2">
                  {w.evidence.map((id) => (
                    <RepoLink key={id} href={recordUrl(id)}>
                      {id}
                    </RepoLink>
                  ))}
                </div>
              </li>
            ))}
          </ul>
        </Panel>
        <Panel
          title="Next roadmap step"
          action={
            <Link to="/roadmap" className="text-[11px] text-accent">
              All 24 steps →
            </Link>
          }
        >
          {next ? (
            <div>
              <p className="text-[13px] font-medium">
                Step {next.step}: {next.title}
              </p>
              <p className="mt-0.5 text-[12px] text-muted-foreground">{next.note}</p>
            </div>
          ) : (
            <p className="text-[12px] text-muted-foreground">No step is marked next.</p>
          )}
        </Panel>
      </div>

      <Panel title="How far there is to go (honest view)">
        <div className="grid gap-4 sm:grid-cols-3">
          <Field label="Largest model trained so far" value={`${fmtInt(largest)} parameters`} />
          <Field
            label="Training text (FrontierCorpus v1 pilot)"
            value={`${fmtInt(c.sides.train.chars)} characters`}
          />
          <Field label="Hardware used so far" value="CPU only (no GPU yet)" />
        </div>
        <p className="mt-3 text-[12px] text-muted-foreground">
          These small models exist to make careful choices (tokenizer, architecture, settings)
          cheaply. They are not assistants and cannot chat. Competing with the best models needs GPU
          training (step 10 onward) and far more clean text, which is the data work now in progress.
        </p>
      </Panel>

      <Panel
        title="Latest experiments"
        action={
          <Link to="/experiments" className="text-[11px] text-accent">
            All →
          </Link>
        }
      >
        <DataTable columns={["ID", "Date", "Title", "Status"]}>
          {recent.map((e) => (
            <Row key={e.id}>
              <Cell mono>
                <RepoLink href={e.url}>{e.id}</RepoLink>
              </Cell>
              <Cell mono muted>
                {e.date ?? "—"}
              </Cell>
              <Cell>{e.title}</Cell>
              <Cell>
                <StatusChip status={e.category} />
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
    </div>
  );
}
