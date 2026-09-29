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
import {
  fmtBytes,
  fmtCount,
  fmtInt,
  fmtPct,
  lab,
  languageName,
  recordUrl,
  short,
} from "@/lib/lab/state";

export const Route = createFileRoute("/datasets")({
  head: () => ({
    meta: [
      { title: "Data & tokenizer — Frontier AI Lab" },
      {
        name: "description",
        content: "FrontierCorpus v1, the Sangraha slice for v2, and the frozen tokenizer.",
      },
    ],
  }),
  component: Datasets,
});

function Datasets() {
  const t = lab.tokenizer;
  const c = lab.corpus_v1;
  const s = lab.sangraha_slice1;
  const langs = Object.entries(c.per_language);

  return (
    <div className="space-y-3">
      <PageHeader
        title="Data & tokenizer"
        description="Read from the tokenizer freeze record, the FrontierCorpus v1 manifest, the pinned Sangraha slice file and its EXP-034 inspection. Sizes are measured. The only estimate on this page is the Sangraha token column, and it is labelled as one."
      />

      <Panel title="Frozen tokenizer" meta={<RepoLink href={t.url}>FREEZE.json</RepoLink>}>
        <div className="grid gap-4 sm:grid-cols-4">
          <Field label="Name" value={t.id} />
          <Field label="Status">
            <StatusChip status={t.status} />
          </Field>
          <Field label="Vocabulary" value={fmtInt(t.vocab_size)} />
          <Field label="Pre-tokenizer" value={t.pretoken} />
          <Field label="Chosen by">
            <RepoLink href={recordUrl(t.decision)}>{t.decision}</RepoLink>{" "}
            <span className="text-[11px] text-muted-foreground">({t.selected_by})</span>
          </Field>
          <Field label="Frozen at" value={t.frozen_at} />
          <Field
            label="Trained on"
            value={`${t.training_corpus} · ${fmtInt(t.trained_chars)} chars`}
          />
          <Field label="Artifact sha256" value={short(t.dir_sha256, 16)} />
        </div>
      </Panel>

      <Panel
        title={`FrontierCorpus v1 (${c.version})`}
        meta={<RepoLink href={c.url}>manifest.json</RepoLink>}
      >
        <div className="grid gap-4 sm:grid-cols-4">
          <Field
            label="Train"
            value={`${fmtInt(c.sides.train.documents)} docs · ${fmtInt(c.sides.train.chars)} chars`}
          />
          <Field
            label="Held-out"
            value={`${fmtInt(c.sides.held_out.documents)} docs · ${fmtInt(c.sides.held_out.chars)} chars`}
          />
          <Field
            label="Sources / licences"
            value={`${c.sources} sources · ${c.licenses.join(", ")}`}
          />
          <Field label="Domains" value={c.domains.join(", ")} />
        </div>
        <p className="mb-2 mt-4 label-xs">Cleaning pipeline</p>
        <DataTable columns={["Stage", "In", "Kept", "Removed"]}>
          {c.stages.map((st) => (
            <Row key={st.stage}>
              <Cell mono>{st.stage}</Cell>
              <Cell mono muted>
                {fmtInt(st.documents_in)}
              </Cell>
              <Cell mono>{fmtInt(st.kept)}</Cell>
              <Cell mono muted>
                {fmtInt(st.removed)}
              </Cell>
            </Row>
          ))}
        </DataTable>
        <p className="mb-2 mt-4 label-xs">Per language</p>
        <DataTable
          columns={[
            "Language",
            "Sources",
            "Train docs",
            "Train chars",
            "Held-out docs",
            "Held-out chars",
          ]}
        >
          {langs.map(([code, l]) => (
            <Row key={code}>
              <Cell>
                {languageName(code)}{" "}
                <span className="font-mono text-[10px] text-faint">{code}</span>
              </Cell>
              <Cell mono muted>
                {fmtInt(l.sources)}
              </Cell>
              <Cell mono>{fmtInt(l.train_documents)}</Cell>
              <Cell mono>{fmtInt(l.train_chars)}</Cell>
              <Cell mono muted>
                {fmtInt(l.held_out_documents)}
              </Cell>
              <Cell mono muted>
                {fmtInt(l.held_out_chars)}
              </Cell>
            </Row>
          ))}
        </DataTable>
      </Panel>

      <Panel
        title="Next data: Sangraha Verified, slice 1 (not filtered yet)"
        meta={<RepoLink href={s.url}>sangraha_slice1.json</RepoLink>}
      >
        <p className="mb-3 text-[12px] text-muted-foreground">{s.status}</p>
        <div className="grid gap-4 sm:grid-cols-4">
          <Field label="Dataset" value={`${s.dataset} · ${s.subset}`} />
          <Field label="Licence" value={s.license_id} />
          <Field label="Pinned revision" value={short(s.revision, 12)} />
          <Field
            label="Download size"
            value={`${fmtBytes(s.total_bytes)} · ${s.files.length} files`}
          />
        </div>
        <p className="mt-3 text-[11px] text-faint">{s.attribution}</p>
        <div className="mt-3">
          <DataTable
            columns={[
              "Language",
              "File size",
              "sha256",
              "Documents",
              "Characters",
              "Script check pass",
              "Exact dups",
              "Suite hits",
              "PDF (OCR) share",
              "Tokens (estimate)",
            ]}
          >
            {s.files.map((f) => {
              const m = s.inspection?.per_language[f.language];
              return (
                <Row key={f.language}>
                  <Cell>
                    {languageName(f.language)}{" "}
                    <span className="font-mono text-[10px] text-faint">{f.language}</span>
                  </Cell>
                  <Cell mono>{fmtBytes(f.size)}</Cell>
                  <Cell mono muted>
                    {short(f.sha256, 12)}
                    {m?.sha256_verified ? " ✓" : ""}
                  </Cell>
                  <Cell mono>{m ? fmtInt(m.documents) : "—"}</Cell>
                  <Cell mono>{m ? fmtCount(m.chars) : "—"}</Cell>
                  <Cell mono>{m ? fmtPct(m.script_pass_share) : "—"}</Cell>
                  <Cell mono muted>
                    {m ? fmtInt(m.exact_duplicates) : "—"}
                  </Cell>
                  <Cell mono>{m ? fmtInt(m.suite_hits) : "—"}</Cell>
                  <Cell mono muted>
                    {m ? fmtPct(m.pdf_char_share) : "—"}
                  </Cell>
                  <Cell mono muted>
                    {m ? `≈ ${fmtCount(m.estimated_tokens)}` : "—"}
                  </Cell>
                </Row>
              );
            })}
          </DataTable>
        </div>
        {s.inspection ? (
          <p className="mt-2 text-[11px] text-faint">
            Measured before any filtering by{" "}
            <RepoLink href={recordUrl(s.inspection.experiment)}>{s.inspection.experiment}</RepoLink>{" "}
            (<RepoLink href={s.inspection.url}>SUMMARY.txt</RepoLink>). Suite check:{" "}
            {s.inspection.suite_status}. Tokens are an estimate from a sample of each file (the
            frozen tokenizer on every 50th document), not a corpus size. Suite hits are documents
            that share text with the protected test set; the build must remove them.
          </p>
        ) : null}
      </Panel>
    </div>
  );
}
