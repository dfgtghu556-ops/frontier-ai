import { createFileRoute } from "@tanstack/react-router";
import { Field, PageHeader, Panel, RepoLink, StatusChip } from "@/components/lab/primitives";
import { lab, repoFile } from "@/lib/lab/state";

export const Route = createFileRoute("/compute")({
  head: () => ({
    meta: [
      { title: "Compute — Frontier AI Lab" },
      {
        name: "description",
        content: "The machines the project has, and what is verified about each.",
      },
    ],
  }),
  component: Compute,
});

function Compute() {
  return (
    <div className="space-y-3">
      <PageHeader
        title="Compute"
        description="What the project actually runs on. Hardware details are shown only where a file in the repository records them; anything else is marked NOT VERIFIED."
      />
      <div className="grid gap-3 lg:grid-cols-3">
        {lab.compute.map((c) => (
          <Panel key={c.id} title={c.label} action={<StatusChip status={c.status} />}>
            <p className="text-[13px]">{c.detail}</p>
            {c.measured_environment ? (
              <div className="mt-3 grid grid-cols-1 gap-3">
                {Object.entries(c.measured_environment).map(([k, v]) => (
                  <Field key={k} label={k} value={v} />
                ))}
                {c.measured_environment_source ? (
                  <p className="text-[11px] text-faint">
                    Measured by the evaluation harness:{" "}
                    <RepoLink href={repoFile(c.measured_environment_source)}>report.json</RepoLink>
                  </p>
                ) : null}
              </div>
            ) : null}
            <div className="mt-3 flex flex-wrap gap-2">
              {c.evidence_files.map((f) => (
                <RepoLink key={f} href={repoFile(f)}>
                  {f}
                </RepoLink>
              ))}
            </div>
          </Panel>
        ))}
      </div>
    </div>
  );
}
