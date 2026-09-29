import { createFileRoute } from "@tanstack/react-router";
import { Cell, DataTable, PageHeader, Panel, RepoLink, Row } from "@/components/lab/primitives";
import { lab, repoFile, short } from "@/lib/lab/state";

export const Route = createFileRoute("/documentation")({
  head: () => ({
    meta: [
      { title: "Documentation — Frontier AI Lab" },
      {
        name: "description",
        content: "The project's key documents, and the files this snapshot was built from.",
      },
    ],
  }),
  component: Documentation,
});

function Documentation() {
  return (
    <div className="space-y-3">
      <PageHeader
        title="Documentation"
        description="The documents that define the project. They live in the frontier-ai repository; links open them on GitHub."
      />
      <Panel title="Key documents">
        <DataTable columns={["Document", "File"]}>
          {lab.documents.map((d) => (
            <Row key={d.path}>
              <Cell>{d.title}</Cell>
              <Cell mono>
                <RepoLink href={d.url}>{d.path}</RepoLink>
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
      <Panel title="How this snapshot was built" meta={`digest ${short(lab.source_digest, 16)}`}>
        <p className="mb-3 text-[12px] text-muted-foreground">
          <span className="font-mono">scripts/export_lab_state.py</span> read these files and wrote{" "}
          <span className="font-mono">apps/lab-os/src/data/lab_state.json</span>. The digest in the
          header changes whenever any of them changes, so two identical digests mean identical data.
        </p>
        <DataTable columns={["Input file", "sha256"]}>
          {Object.entries(lab.sources).map(([path, sha]) => (
            <Row key={path}>
              <Cell mono>
                <RepoLink href={repoFile(path)}>{path}</RepoLink>
              </Cell>
              <Cell mono muted>
                {short(sha, 16)}
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>
    </div>
  );
}
