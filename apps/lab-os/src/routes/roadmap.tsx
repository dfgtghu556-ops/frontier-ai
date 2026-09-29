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
import { lab, repoFile } from "@/lib/lab/state";

export const Route = createFileRoute("/roadmap")({
  head: () => ({
    meta: [
      { title: "Roadmap — Frontier AI Lab" },
      {
        name: "description",
        content: "The 24-step method (MASTER_CONTEXT §37) and each step's evidence.",
      },
    ],
  }),
  component: Roadmap,
});

function Roadmap() {
  return (
    <div className="space-y-3">
      <PageHeader
        title="Roadmap"
        description="The 24 steps from MASTER_CONTEXT §37, in order. Step status lives in lab/registry.json; the exporter refuses to build if a title differs from §37, if cited evidence does not exist, or if a step is marked complete while an experiment it cites is not complete or a decision it cites is not accepted."
      />
      <Panel meta={<RepoLink href={repoFile("MASTER_CONTEXT.md")}>MASTER_CONTEXT.md §37</RepoLink>}>
        <DataTable columns={["#", "Step", "Status", "Evidence", "Note"]}>
          {lab.roadmap.map((s) => (
            <Row key={s.step}>
              <Cell mono muted>
                {s.step}
              </Cell>
              <Cell>{s.title}</Cell>
              <Cell>
                <StatusChip status={s.status} />
              </Cell>
              <Cell>
                <div className="flex flex-wrap gap-x-2 gap-y-1">
                  {s.evidence.map((id, i) => (
                    <RepoLink key={id} href={s.evidence_links[i] ?? null}>
                      {id}
                    </RepoLink>
                  ))}
                  {s.evidence_files.map((f) => (
                    <RepoLink key={f} href={repoFile(f)}>
                      {f.split("/").pop()}
                    </RepoLink>
                  ))}
                </div>
              </Cell>
              <Cell muted>{s.note}</Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
    </div>
  );
}
