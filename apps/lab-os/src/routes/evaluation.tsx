import { createFileRoute } from "@tanstack/react-router";
import {
  Cell,
  DataTable,
  Field,
  PageHeader,
  Panel,
  RepoLink,
  Row,
  StatusChip,
} from "@/components/lab/primitives";
import { fmtBpb, fmtInt, lab, languageName, recordUrl, short } from "@/lib/lab/state";

export const Route = createFileRoute("/evaluation")({
  head: () => ({
    meta: [
      { title: "Evaluation — Frontier AI Lab" },
      {
        name: "description",
        content: "The protected held-out suite and every published evaluation report.",
      },
    ],
  }),
  component: Evaluation,
});

function Evaluation() {
  const s = lab.suite;
  const reports = lab.evaluation_reports;
  const langs = Object.keys(s.per_language);

  return (
    <div className="space-y-3">
      <PageHeader
        title="Evaluation"
        description="Models are compared by bits per byte on a protected held-out suite: how many bits the model needs, on average, to predict each byte of text it never trained on. Lower is better, and it is fair across tokenizers and scripts."
      />

      <Panel title="Protected suite" meta={<RepoLink href={s.url}>SUITE.json</RepoLink>}>
        <div className="grid gap-4 sm:grid-cols-4">
          <Field label="Suite" value={s.id} />
          <Field label="Status">
            <StatusChip status={s.status} />
          </Field>
          <Field
            label="Size"
            value={`${fmtInt(s.totals.documents)} docs · ${fmtInt(s.totals.chars)} chars`}
          />
          <Field label="Fingerprint" value={short(s.fingerprint, 16)} />
        </div>
        <p className="mt-3 text-[11px] text-faint">
          Standard harness: <RepoLink href={recordUrl("D-042")}>D-042</RepoLink>. Training data is
          checked against this suite before any score is published.
        </p>
      </Panel>

      <Panel
        title="Published reports"
        meta={`${reports.length} reports · evals/results/*/*/report.json`}
      >
        <DataTable
          columns={[
            "Experiment",
            "Model",
            "Params",
            "Steps",
            "Bits/byte",
            "95% CI",
            "Contamination",
          ]}
        >
          {reports.map((r) => (
            <Row key={r.url}>
              <Cell mono>
                <RepoLink href={recordUrl(r.experiment)}>{r.experiment}</RepoLink>
              </Cell>
              <Cell mono>
                <RepoLink href={r.url}>{r.label}</RepoLink>
              </Cell>
              <Cell mono muted>
                {fmtInt(r.n_params)}
              </Cell>
              <Cell mono muted>
                {fmtInt(r.step)}
              </Cell>
              <Cell mono>{fmtBpb(r.bits_per_byte)}</Cell>
              <Cell mono muted>
                {fmtBpb(r.ci95[0])}–{fmtBpb(r.ci95[1])}
              </Cell>
              <Cell>
                <StatusChip status={r.contamination} />
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>

      <Panel title="Bits/byte per language" meta="lower is better">
        <DataTable columns={["Model", ...langs.map(languageName)]}>
          {reports.map((r) => (
            <Row key={r.url}>
              <Cell mono>{r.label}</Cell>
              {langs.map((l) => (
                <Cell key={l} mono muted>
                  {fmtBpb(r.per_language[l], 2)}
                </Cell>
              ))}
            </Row>
          ))}
        </DataTable>
        <p className="mt-2 text-[11px] text-faint">
          Compare models within a column, not languages across a row. UTF-8 stores an English letter
          in 1 byte, an Urdu letter in 2 and a letter of the Indic scripts in 3, so the same amount
          of information is spread over a different number of bytes in each language.
        </p>
      </Panel>
    </div>
  );
}
