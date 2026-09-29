import { createFileRoute } from "@tanstack/react-router";
import {
  Cell,
  DataTable,
  PageHeader,
  Panel,
  RepoLink,
  Row,
  StatusChip,
} from "@/components/lab/primitives";
import { lab, recordUrl, repoFile } from "@/lib/lab/state";

export const Route = createFileRoute("/decisions")({
  head: () => ({
    meta: [
      { title: "Decisions — Frontier AI Lab" },
      {
        name: "description",
        content: "Every decision recorded in DECISIONS.md, and the open questions.",
      },
    ],
  }),
  component: Decisions,
});

function Decisions() {
  const rows = [...lab.decisions].reverse();
  return (
    <div className="space-y-3">
      <PageHeader
        title="Decisions"
        description="Parsed from DECISIONS.md. A decision fixes a choice (a tokenizer, an evaluation standard, a data policy) and cites the experiments behind it. Provisional decisions are expected to be revisited."
      />
      <Panel>
        <DataTable columns={["ID", "Date", "Decision", "Status", "Evidence"]}>
          {rows.map((d) => (
            <Row key={d.id}>
              <Cell mono>
                <RepoLink href={d.url}>{d.id}</RepoLink>
              </Cell>
              <Cell mono muted>
                {d.date ?? "—"}
              </Cell>
              <Cell>
                <p className="font-medium">{d.title}</p>
                {d.summary ? <p className="mt-0.5 text-muted-foreground">{d.summary}</p> : null}
              </Cell>
              <Cell>
                <StatusChip status={d.category} />
              </Cell>
              <Cell>
                <div className="flex flex-wrap gap-1.5">
                  {d.experiments.map((x) => (
                    <RepoLink key={x} href={recordUrl(x)}>
                      {x}
                    </RepoLink>
                  ))}
                </div>
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
      <Panel
        title="Open questions"
        meta={<RepoLink href={repoFile("DECISIONS.md")}>DECISIONS.md · Open items</RepoLink>}
      >
        <DataTable columns={["ID", "Question", "Deferred to"]}>
          {lab.open_questions.map((q) => (
            <Row key={q.id}>
              <Cell mono>{q.id}</Cell>
              <Cell>{q.question}</Cell>
              <Cell muted>{q.deferred_to}</Cell>
            </Row>
          ))}
        </DataTable>
        <p className="mt-2 text-[11px] text-faint">
          This table is copied from DECISIONS.md as written; some items may since have been answered
          by later decisions.
        </p>
      </Panel>
    </div>
  );
}
