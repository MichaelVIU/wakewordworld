# Hosting a community wake-word benchmark on Hugging Face (and alternatives) — Sept 2026

## 1. Open ASR Leaderboard (the reference model)
Sources: https://github.com/huggingface/open_asr_leaderboard ; https://huggingface.co/spaces/hf-audio/open_asr_leaderboard ; https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results ; https://arxiv.org/abs/2510.06961
- Submitters run the evaluation themselves via HF Jobs on a 1x H200; Docker image per model family; full English short-form run costs ~$3–6.
- Submission = GitHub pull request with template (per-split WER, RTFx, run via HF Jobs, fixed decoding params, licence, training-data disclosure). Space is display-only.
- Results = CSV/Parquet in dataset repos, pinned by git SHA. Eval code Apache-2.0.
- Accepts commercial APIs via `api/providers/` (15 providers, keys as env vars).
- Contamination stance: multiple datasets per track, at least one NC-licensed dataset per track, plus vendor-private sets.

## 2. HF evaluation products: alive or dead

| Name | Status | Auto-runs evals on submission? |
|---|---|---|
| Evaluation on the Hub / autoevaluate / Model Evaluator | Dead (runtime errors, last update 2023) | No |
| `evaluate` library | Maintenance mode, points to LightEval | n/a |
| Open LLM Leaderboard | Archived | No |
| Community Evals / Eval Results (beta, Feb 2026) https://huggingface.co/docs/hub/eval-results ; https://huggingface.co/blog/community-evals | Live. Dataset repo + `eval.yaml` → Benchmark with leaderboard on the dataset card; scores in `.eval_results/*.yaml` on model repos; badges verified/community/leaderboard/source. `verified` only for inspect-ai runs on Jobs. Registration needs HF allow-listing. Only LLM benchmarks registered; nothing audio. | Only inspect-ai verification; no auto-run |
| Every Eval Ever (EEE) https://huggingface.co/blog/eee-community-evals | Live; unified JSON datastore | No compute |
| OpenEvals org / Community leaderboards https://huggingface.co/docs/leaderboards/index | Live; Spaces category `model-benchmarking` | No |
| lighteval | Active, LLM only | No audio |
| "Open Reports" | Nothing by that name on HF | — |

Bottom line: no HF product runs an arbitrary audio evaluation automatically on submission. Jobs + webhooks/scheduling must be wired up by the benchmark maintainers.

## 3. HF Jobs and Spaces
Sources: https://huggingface.co/docs/hub/jobs-overview ; https://huggingface.co/docs/hub/en/jobs-pricing ; https://huggingface.co/docs/hub/jobs-schedule ; https://huggingface.co/docs/hub/en/jobs-webhooks ; https://huggingface.co/docs/hub/spaces-overview ; https://huggingface.co/docs/hub/storage-buckets
- Jobs: any user/org with positive credit balance; billed per minute. `hf jobs run <image> <cmd>`, `hf jobs uv run script.py`, Python `run_job()`, HTTP API.
- Hardware/prices per hour: cpu-basic 2 vCPU/16 GB $0.01; cpu-upgrade 8 vCPU/32 GB $0.03; cpu-xl 16 vCPU $1.00; t4-small $0.40; a10g-small $1.00; a100-large $2.50; h200 $5.00.
- Default timeout 30 min, `--timeout 1d` possible. Volumes: mount dataset repos read-only or Storage Buckets read-write (`-v hf://datasets/org/ds:/data`). Secrets via `-s`.
- Scheduled jobs: `hf jobs scheduled uv run "@weekly" ...`.
- Webhook-triggered jobs: `create_webhook(job_id=..., watched=[...], domains=["repo","discussion"])` re-runs a template Job on repo content updates, new PR refs, discussion events. A PR to a dataset repo can automatically launch an eval Job.
- Spaces: Gradio and Docker Spaces now require a paid plan (PRO $9/month for personal, Team/Enterprise for orgs); static Spaces free; CPU Basic no hourly cost; ZeroGPU quotas unsuitable for long eval loops. Persistent storage via Storage Buckets ($12/TB/month public).

## 4. Hosting the audio test set on HF
Sources: https://huggingface.co/docs/hub/storage-limits ; https://huggingface.co/docs/hub/datasets-gated ; https://huggingface.co/docs/hub/datasets-audio ; https://huggingface.co/docs/hub/repositories-licenses
- No per-repo size limit; free public = best-effort generous; PRO 10 TB public. ≤10k entries per folder (hard), files <200 GB, ~50–100 files per commit. Xet default backend. Storage grants for high-impact open work.
- Gated datasets: automatic or manual approval, click-through `extra_gated_prompt`, custom fields, acceptance log per user, REST API. Public gated datasets keep the viewer.
- Audio layouts: AudioFolder with `metadata.csv/jsonl/parquet`; Parquet with `audio` column recommended for <1 MB clips; WebDataset tars for large sets. Viewer converts first 5 GB.
- Mixed licences in one repo: allowed; People's Speech uses `license: [cc-by-4.0, cc-by-sa-4.0, ...]` with configs per licence subset and a `license` column per row. No per-file licence field in the spec; document in card.
- Leaderboard UI: `gradio-templates/leaderboard` (Gradio 6, Apache-2.0, maintained); `gradio_leaderboard` component unmaintained.

## 5. Credibility patterns from other benchmarks
- Open ASR: organiser-reviewed, pinned hardware, Docker per model, results by git SHA, mixed public/NC/private sets.
- TTS Arena V2 (https://docs.ttsarena.org): Bradley–Terry, ranked by lower CI bound, 100 votes minimum, anti-fraud.
- AudioBench: email submission, weak. Dynamic-SUPERB: self-reported JSON PRs, no re-run. MMAU: withheld test labels, prediction upload, "unverified" tag for community scores.
- MLPerf Tiny (https://github.com/mlcommons/tiny): Keyword Spotting + Streaming Wake Word (v1.3, 2025); submitters provide code + binaries + logs, all made public; "results that cannot be replicated are not valid"; ~4-week peer review; membership-based, per-round.
- No audio leaderboard continuously runs third-party on-device engines as pinned containers; Open ASR is closest.
- Sealed vs open test sets: private held-out splits (Open ASR, Scale SEAL first-look rule), withheld labels (MMAU), periodic refresh (LiveBench monthly), multiple correlated sets, NC-licensed decoy, canary strings (BIG-bench GUID), lower-CI ranking with minimum sample counts. Survey: https://arxiv.org/abs/2406.04244

## 6. Alternatives outside HF
- GitHub Actions self-hosted runners: ARM64/ARM32 supported (Raspberry Pi as runner; ESP32 as flashed DUT). Never for fork PRs on public repos; use JIT runners, maintainer-merged commits only, no secrets.
- GitLab CI, Codeberg Woodpecker (attach own agents).
- Zenodo: 50 GB per record, concept DOI + version DOI — ideal for citable frozen releases mirrored from HF.
- Papers with Code: shut down July 2025. OpenML: tabular only.

## 7. Recommended architecture
1. Data: HF dataset repo `wakeword-bench-public` (gated, automatic approval, click-through no-training terms, canary GUID in metadata), configs split by licence, `license` column per row, Parquet audio for short clips / WebDataset for long files; private `wakeword-bench-sealed` split read only by eval Jobs, rotated yearly with old sealed split released; each release tagged and mirrored to Zenodo.
2. Eval code: GitHub, Apache-2.0. Harness defines streaming protocol, detection semantics, metrics; submission contract = Docker image digest + pinned engine/model hash + config; per-family Dockerfiles.
3. Who runs it: benchmark org runs every scored result on HF Jobs `cpu-upgrade` ($0.03/h; a 10-h test set costs cents); optional GPU lane. On-device latency/energy lane: self-hosted runner on Raspberry Pi 5 + ESP32-S3 DUT in maintainer's lab, maintainer-approved commits only.
4. Submissions: PR to harness repo or `submissions/<engine>.yaml` in results repo; HF webhook triggers template Job; weekly scheduled re-run for drift. Commercial/API engines allowed via provider adapters, flagged "API, not on-device".
5. Publishing: results Parquet in `wakeword-bench-results` (git-versioned); leaderboard on Gradio 6 Space (needs PRO) or static HTML (free); `.eval_results` YAML PRs to submitters' model repos; register `eval.yaml` for Community Evals once non-inspect frameworks are allowed; push to Every Eval Ever.
6. Credibility: organiser-run only; sealed split with first-look rule; public-vs-sealed gap as contamination signal; annual refresh; canary strings + gated access log; bootstrap CIs, rank by lower bound with minimum clip threshold; per-clip predictions stored; objection window via GitHub issues.

Key URLs: https://github.com/huggingface/open_asr_leaderboard · https://huggingface.co/docs/hub/eval-results · https://github.com/huggingface/community-evals · https://huggingface.co/docs/hub/jobs-overview · https://huggingface.co/docs/hub/en/jobs-webhooks · https://huggingface.co/docs/hub/datasets-gated · https://huggingface.co/spaces/gradio-templates/leaderboard · https://github.com/mlcommons/tiny · https://about.zenodo.org/policies/
