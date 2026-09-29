/**
 * Types of the repository snapshot (`src/data/lab_state.json`, schema frontier-lab-state-v1).
 *
 * The snapshot is written by `scripts/export_lab_state.py` in the frontier-ai repository from
 * the repository's own records (EXPERIMENTS.md, DECISIONS.md, the tokenizer freeze record, the
 * corpus manifest, the protected suite and the evaluation results). The UI only renders it:
 * nothing shown here is typed in by hand or invented.
 */

export type StepStatus = "complete" | "current" | "next" | "not_started";
export type ExperimentCategory = "complete" | "in_progress" | "planned" | "abandoned" | "other";
export type DecisionCategory = "accepted" | "provisional" | "superseded" | "proposed" | "other";

export interface RoadmapStep {
  step: number;
  title: string;
  status: StepStatus;
  evidence: string[];
  evidence_files: string[];
  evidence_links: (string | null)[];
  note: string;
}

export interface CurrentWork {
  title: string;
  evidence: string[];
  evidence_files: string[];
  note: string;
}

export interface ExperimentRecord {
  id: string;
  title: string;
  date: string | null;
  status: string;
  category: ExperimentCategory;
  step: number | null;
  summary: string;
  decisions: string[];
  line: number;
  url: string;
}

export interface DecisionRecord {
  id: string;
  title: string;
  date: string | null;
  status: string;
  category: DecisionCategory;
  summary: string;
  experiments: string[];
  line: number;
  url: string;
}

export interface OpenQuestion {
  id: string;
  question: string;
  deferred_to: string;
}

export interface TokenizerRecord {
  id: string;
  status: string;
  frozen_at: string;
  decision: string;
  frozen_by: string;
  vocab_size: number;
  merges: number;
  pretoken: string;
  impl: string;
  dir_sha256: string;
  trained_chars: number;
  training_corpus: string;
  selected_by: string;
  url: string;
}

export interface CorpusSide {
  documents: number;
  chars: number;
  bytes: number;
}

export interface CorpusLanguage {
  train_documents?: number;
  train_chars?: number;
  held_out_documents?: number;
  held_out_chars?: number;
  sources?: number;
}

export interface CorpusStage {
  stage: string;
  documents_in: number | null;
  kept: number;
  removed: number;
}

export interface CorpusV1 {
  id: string;
  version: string;
  created_at: string;
  content_sha256: string;
  domains: string[];
  licenses: string[];
  sources: number;
  normalization: { policy: string; version: string };
  sides: { train: CorpusSide; held_out: CorpusSide };
  per_language: Record<string, CorpusLanguage>;
  stages: CorpusStage[];
  url: string;
}

export interface SliceInspectionLanguage {
  documents: number;
  chars: number;
  script_pass_share: number;
  exact_duplicates: number;
  suite_hits: number;
  pdf_char_share: number;
  estimated_tokens: number;
  token_sample_documents: number;
  sha256_verified: boolean;
}

export interface SliceInspection {
  experiment: string;
  complete: boolean;
  suite_status: string;
  finished_at: string;
  per_language: Record<string, SliceInspectionLanguage>;
  url: string;
}

export interface SangrahaSlice {
  slice_id: string;
  dataset: string;
  subset: string;
  revision: string;
  license_id: string;
  trust_level: string;
  attribution: string;
  total_bytes: number;
  files: { language: string; size: number; sha256: string }[];
  status: string;
  inspection: SliceInspection | null;
  url: string;
}

export interface SuiteRecord {
  id: string;
  status: string;
  totals: { documents: number; bytes: number; chars: number };
  fingerprint: string;
  per_language: Record<string, { documents: number; bytes: number; chars: number }>;
  url: string;
}

export interface EvaluationReport {
  experiment: string;
  label: string;
  created_at: string;
  n_params: number;
  vocab_size: number;
  step: number;
  tokenizer: string;
  suite: string;
  bits_per_byte: number;
  ci95: [number, number];
  documents: number;
  per_language: Record<string, number>;
  contamination: string;
  environment: Record<string, string>;
  url: string;
}

export interface AblationRow {
  group: string;
  arm: string;
  lr: number | null;
  seeds: number[];
  bpb: number[];
  mean: number;
  std: number;
  n_params: number;
  body_params: number;
  kv_cache_values_per_token: number | null;
}

export interface Ablation {
  experiment: string;
  max_steps: number;
  rows: AblationRow[];
  summary_url: string;
}

export interface ModelRecord {
  id: string;
  experiment: string;
  name: string;
  seeds: number;
  n_params: number;
  body_params: number | null;
  vocab_size: number | null;
  steps: number;
  bpb_mean: number;
  bpb_std: number | null;
  hardware: string;
}

export interface BaselineModel {
  model: string;
  evidence: string[];
  note: string;
  record: ModelRecord;
}

export interface ComputeRecord {
  id: string;
  label: string;
  kind: "cpu" | "gpu";
  status: string;
  detail: string;
  evidence_files: string[];
  measured_environment: Record<string, string> | null;
  measured_environment_source: string | null;
}

export interface DocumentLink {
  path: string;
  title: string;
  url: string;
}

export interface LabState {
  schema: "frontier-lab-state-v1";
  repository: { url: string; branch: string };
  source_digest: string;
  sources: Record<string, string>;
  roadmap: RoadmapStep[];
  current_work: CurrentWork[];
  experiments: ExperimentRecord[];
  experiment_numbers_not_in_log: number[];
  decisions: DecisionRecord[];
  open_questions: OpenQuestion[];
  tokenizer: TokenizerRecord;
  corpus_v1: CorpusV1;
  sangraha_slice1: SangrahaSlice;
  suite: SuiteRecord;
  evaluation_reports: EvaluationReport[];
  ablations: Ablation[];
  models: ModelRecord[];
  baseline_model: BaselineModel | null;
  compute: ComputeRecord[];
  documents: DocumentLink[];
}
