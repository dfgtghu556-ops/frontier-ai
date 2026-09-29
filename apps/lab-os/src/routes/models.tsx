import { createFileRoute } from "@tanstack/react-router";
import { Cell, DataTable, PageHeader, Panel, RepoLink, Row } from "@/components/lab/primitives";
import { fmtBpb, fmtInt, lab, recordUrl } from "@/lib/lab/state";

export const Route = createFileRoute("/models")({
  head: () => ({
    meta: [
      { title: "Models — Frontier AI Lab" },
      { name: "description", content: "Every model configuration trained and measured so far." },
    ],
  }),
  component: Models,
});

function Models() {
  const experiments = [...new Set(lab.models.map((m) => m.experiment))];

  return (
    <div className="space-y-3">
      <PageHeader
        title="Models"
        description="Every configuration trained so far, grouped by experiment. All of them are small research models trained on a laptop CPU to compare choices fairly; none is a finished or usable model."
      />
      {experiments.map((x) => {
        const rows = lab.models.filter((m) => m.experiment === x);
        const ablation = lab.ablations.find((a) => a.experiment === x);
        return (
          <Panel
            key={x}
            title={x}
            meta={
              <RepoLink href={ablation ? ablation.summary_url : recordUrl(x)}>
                {ablation ? "summary.json" : "record"}
              </RepoLink>
            }
          >
            <DataTable
              columns={[
                "Configuration",
                "Seeds",
                "Parameters",
                "Body params",
                "Steps",
                "Bits/byte (mean ± sd)",
              ]}
            >
              {rows.map((m) => (
                <Row key={m.id}>
                  <Cell mono>{m.name}</Cell>
                  <Cell mono muted>
                    {m.seeds}
                  </Cell>
                  <Cell mono>{fmtInt(m.n_params)}</Cell>
                  <Cell mono muted>
                    {fmtInt(m.body_params)}
                  </Cell>
                  <Cell mono muted>
                    {fmtInt(m.steps)}
                  </Cell>
                  <Cell mono>
                    {fmtBpb(m.bpb_mean)}
                    {m.bpb_std !== null && m.seeds > 1 ? ` ± ${fmtBpb(m.bpb_std)}` : ""}
                  </Cell>
                </Row>
              ))}
            </DataTable>
            <p className="mt-2 text-[11px] text-faint">
              Read the decision in the <RepoLink href={recordUrl(x)}>{x} record</RepoLink>: a lower
              number here is not automatically adopted. Rows with 1 seed are side checks (for
              example a learning-rate probe), not comparisons.
            </p>
          </Panel>
        );
      })}
    </div>
  );
}
