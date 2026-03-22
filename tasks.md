# prompt-drift — Implementation Tasks

This file tracks all implementation tasks derived from `SPEC.md`. Each task maps to a specific requirement, feature, or edge case described in the specification.

---

## Phase 1: Project Scaffolding and Core Types

- [x] **Set up dev dependencies** — Install `typescript`, `vitest`, `eslint`, and `@types/node` as dev dependencies. Verify `npm run build`, `npm run test`, and `npm run lint` all execute (even if there is no real code yet). | Status: done
- [ ] **Add CLI bin entry to package.json** — Add `"bin": { "prompt-drift": "dist/cli.js" }` to `package.json` so the CLI is available after global install or via npx. | Status: not_done
- [x] **Create `src/types.ts`** — Define all TypeScript types and interfaces: `EmbedFn`, `LlmFn`, `MonitorOptions`, `SnapshotOptions`, `ProbeOptions`, `CheckOptions`, `Snapshot`, `DriftReport`, `ProbeResult`, `MetricResult`, `MetricThresholds`, `MetricWeights`, `DriftSeverity`, plus the `lengthStats` and `vocabularyStats` sub-interfaces. Match the spec exactly. | Status: done
- [ ] **Create `src/errors.ts`** — Implement `DriftError` class extending `Error` with a typed `code` property. Supported codes: `INSUFFICIENT_OUTPUTS`, `INCONSISTENT_DIMENSIONS`, `NO_BASELINE`, `INCOMPATIBLE_SNAPSHOTS`, `INVALID_SNAPSHOT`, `EMBED_FAILED`. | Status: not_done
- [x] **Create `src/index.ts` public API exports** — Export `createMonitor`, all public types, `DriftError`, `saveSnapshot`, `loadSnapshot`. This is the barrel file. | Status: done

---

## Phase 2: Math Utilities

- [x] **Implement `src/math.ts` — cosine similarity** — Compute cosine similarity between two equal-length number arrays using dot product / (L2 norm * L2 norm). Handle zero-vector edge case (return 0). | Status: done
- [ ] **Implement `src/math.ts` — cosine distance** — `1 - cosineSimilarity(a, b)`. | Status: not_done
- [ ] **Implement `src/math.ts` — dot product** — Element-wise multiply and sum two equal-length number arrays. | Status: not_done
- [ ] **Implement `src/math.ts` — L2 norm** — Square root of sum of squares. | Status: not_done
- [x] **Implement `src/math.ts` — element-wise mean** — Given an array of equal-length number arrays, compute the mean of each dimension. Returns a single number array (the centroid). | Status: done
- [ ] **Implement `src/math.ts` — element-wise variance** — Given an array of equal-length number arrays and their centroid, compute the variance per dimension. | Status: not_done
- [x] **Implement `src/math.ts` — percentile computation** — Given a sorted array of numbers and a target percentile (e.g., 5, 50, 95), return the interpolated percentile value. Used for output length stats (p5, median, p95). | Status: done
- [x] **Implement `src/math.ts` — basic statistics** — `mean`, `stddev`, `median` helper functions over number arrays. | Status: done
- [ ] **Implement `src/math.ts` — reservoir sampling** — Given an array and a sample size k, return a random sample of k elements. Used for sampling embeddings and output strings. | Status: not_done
- [ ] **Write `src/__tests__/math.test.ts`** — Unit tests for all math utilities: cosine similarity (identical vectors = 1, orthogonal = 0, known values), cosine distance, dot product, L2 norm, element-wise mean (verified against hand-computed 3x4 matrix), element-wise variance, percentile (p5, p50, p95 of known arrays), mean, stddev, median, reservoir sampling (correct length, subset of input). | Status: not_done

---

## Phase 3: Tokenizer

- [ ] **Implement `src/tokenizer.ts`** — Simple whitespace/punctuation tokenizer: split on whitespace, lowercase all tokens, strip leading/trailing punctuation characters. Return an array of cleaned tokens. No external NLP dependency. | Status: not_done
- [ ] **Write `src/__tests__/tokenizer.test.ts`** — Unit tests: mixed case input lowercased, punctuation stripped, empty strings produce empty array, multi-space/tab/newline splitting, unicode handling. | Status: not_done

---

## Phase 4: Snapshot Creation

- [ ] **Implement `src/snapshot.ts` — `createSnapshot` function** — Accepts outputs (string[]), embedFn, promptId, embeddingModel, and SnapshotOptions. Orchestrates the full snapshot creation pipeline: validate inputs, embed, compute all statistics, sample, assign UUID and timestamp. Returns a `Snapshot`. | Status: not_done
- [ ] **Snapshot validation — minimum output count** — Throw `DriftError` with code `INSUFFICIENT_OUTPUTS` if fewer than 20 outputs are provided. | Status: not_done
- [ ] **Snapshot validation — embedding dimensionality consistency** — After calling embedFn, verify all returned vectors have the same length. Throw `DriftError` with code `INCONSISTENT_DIMENSIONS` if not. | Status: not_done
- [ ] **Snapshot validation — embedFn error handling** — Wrap embedFn call in try/catch. If embedFn throws, re-throw as `DriftError` with code `EMBED_FAILED` and the original error as `cause`. | Status: not_done
- [x] **Snapshot — centroid computation** — Compute element-wise mean of all embedding vectors using `math.ts` utilities. Store as `snapshot.centroid`. | Status: done
- [ ] **Snapshot — per-dimension variance** — Compute variance per dimension using `math.ts`. Store as `snapshot.variance`. | Status: not_done
- [x] **Snapshot — spread computation** — Compute mean cosine distance from each embedding to the centroid. Store as `snapshot.spread`. | Status: done
- [x] **Snapshot — pairwise similarity statistics** — Sample M pairs (M = min(500 or configured pairwiseSamplePairs, n*(n-1)/2)), compute cosine similarity for each pair, record mean and standard deviation. Store as `snapshot.meanPairwiseSimilarity` and `snapshot.stdPairwiseSimilarity`. | Status: done
- [x] **Snapshot — output length statistics** — Compute mean, stddev, median, p5, p95 of `output.length` (character count) for each output string. Store in `snapshot.lengthStats`. | Status: done
- [x] **Snapshot — vocabulary statistics** — Tokenize all outputs using `tokenizer.ts`, compute term frequencies, record top-K terms (default K=100 or configured topK), vocabulary size, and total token count. Store in `snapshot.vocabularyStats`. | Status: done
- [ ] **Snapshot — sample embeddings (reservoir sampling)** — Randomly sample `sampleSize` (default 50) embedding vectors using reservoir sampling. Store as `snapshot.sampleEmbeddings`. If n < sampleSize, store all. | Status: not_done
- [ ] **Snapshot — sample outputs (reservoir sampling)** — Randomly sample `outputSampleSize` (default 10) raw output strings using reservoir sampling. Store as `snapshot.sampleOutputs`. If n < outputSampleSize, store all. | Status: not_done
- [x] **Snapshot — UUID and timestamp** — Assign a UUID v4 (using `node:crypto.randomUUID()`) and ISO 8601 timestamp (`new Date().toISOString()`). | Status: done
- [ ] **Snapshot — dimensionality and metadata** — Record `dimensionality` from the embedding vector length, `embeddingModel` from config, and optional caller-provided `metadata`. | Status: not_done
- [ ] **Write `src/__tests__/snapshot.test.ts`** — Unit tests: calls embedFn with full output array; correct promptId and dimensionality; centroid is arithmetic mean (verified for 3 outputs, 4 dims); per-dimension variance verified manually; spread verified for small input set; output length stats for known lengths [100,200,300]; vocabulary stats tokenize/lowercase/count correctly; sampleEmbeddings length = min(sampleSize, n); sampleOutputs length = min(outputSampleSize, n); throws INSUFFICIENT_OUTPUTS for <20 outputs; throws INCONSISTENT_DIMENSIONS for mixed-length embeddings; throws EMBED_FAILED when embedFn throws. | Status: not_done

---

## Phase 5: Drift Metrics

### Metric 1: Centroid Distance

- [x] **Implement `src/metrics/centroid.ts`** — Compute cosine distance between two centroid vectors. Normalize by dividing by the normalization constant (default 0.2). Clamp to [0, 1] with `Math.min(..., 1.0)`. Return a `MetricResult` with score, computed=true, interpretation string, and details (raw cosine distance). | Status: done
- [ ] **Write `src/__tests__/metrics/centroid.test.ts`** — Tests: identical centroids produce score=0; orthogonal centroids produce score=1.0; slightly shifted centroids produce proportional score; verify normalization constant behavior. | Status: not_done

### Metric 2: Spread Change

- [x] **Implement `src/metrics/spread.ts`** — Compute spread ratio: `max(spreadA, spreadB) / (min(spreadA, spreadB) + epsilon)`. Normalize: `(ratio - 1.0) / normalization_constant` (default 1.0). Clamp to [0, 1]. Return `MetricResult`. | Status: done
- [ ] **Write `src/__tests__/metrics/spread.test.ts`** — Tests: identical spread produces score=0; spread 0.1 vs 0.2 (ratio 2.0) produces elevated score; symmetry (A vs B same as B vs A). | Status: not_done

### Metric 3: Jensen-Shannon Divergence (JSD)

- [x] **Implement `src/metrics/jsd.ts`** — Compute cosine distances from all sample embeddings (both baseline and current) to the baseline centroid. Bin into K bins (default 20). Construct discrete distributions P and Q with smoothing (1e-10). Compute JSD = 0.5*KL(P||M) + 0.5*KL(Q||M) where M = 0.5*(P+Q). Normalize by dividing by log(2). Return `MetricResult`. | Status: done
- [ ] **Write `src/__tests__/metrics/jsd.test.ts`** — Tests: identical distributions produce score=0; non-overlapping distributions produce score=1.0; slightly shifted distribution produces proportional score; verify no NaN/Infinity from smoothing; verify bin boundary handling. | Status: not_done

### Metric 4: Population Stability Index (PSI)

- [x] **Implement `src/metrics/psi.ts`** — Use same binning approach as JSD. Compute PSI = sum((Q[i] - P[i]) * ln(Q[i] / P[i])). Normalize: min(PSI / normalization_constant, 1.0) where default normalization_constant=0.5. Return `MetricResult`. | Status: done
- [ ] **Write `src/__tests__/metrics/psi.test.ts`** — Tests: identical distributions produce score=0; significantly different distributions produce score>0.3; PSI always non-negative; smoothing prevents log(0). | Status: not_done

### Metric 5: Output Length Drift

- [x] **Implement `src/metrics/output-length.ts`** — Compute Cohen's d using pooled standard deviation. Compute variance ratio. Combine: `0.7 * cohens_d + 0.3 * (var_ratio - 1.0)`. Normalize by dividing by 2.0. Clamp to [0, 1]. Return `MetricResult`. | Status: done
- [ ] **Write `src/__tests__/metrics/output-length.test.ts`** — Tests: identical length distributions produce score=0; Cohen's d=1.0 produces score ~0.35; different variances but similar means produce elevated score from variance ratio component. | Status: not_done

### Metric 6: Vocabulary Drift

- [x] **Implement `src/metrics/vocabulary.ts`** — Compute cosine distance between term frequency vectors (union vocabulary, zero-fill missing terms). Compute new term ratio (fraction of current top terms absent from baseline top terms). Combine: `0.6 * cosine_distance + 0.4 * new_term_ratio`. Normalize by dividing by 0.5. Clamp to [0, 1]. Return `MetricResult`. | Status: done
- [ ] **Write `src/__tests__/metrics/vocabulary.test.ts`** — Tests: identical term frequencies produce score=0; completely different vocabularies produce score=1.0; partially overlapping vocabularies produce proportional score; new term ratio correctly computed from top-K terms. | Status: not_done

---

## Phase 6: Composite Score and Severity Classification

- [x] **Implement `src/composite.ts` — composite score computation** — Compute weighted average of per-metric scores. Default weights: centroid=0.25, jsd=0.25, psi=0.15, vocabulary=0.15, spread=0.10, outputLength=0.10. Normalize weights to sum to 1. If custom weights provided, use those. | Status: done
- [ ] **Implement `src/composite.ts` — weight redistribution for disabled metrics** — When a metric is disabled (enabledMetrics.X = false), set its score to 0 and redistribute its weight proportionally to remaining enabled metrics. | Status: not_done
- [x] **Implement `src/composite.ts` — severity classification** — Map composite score to severity: <0.05 = none, <0.15 = low, <0.35 = medium, <0.60 = high, >=0.60 = critical. | Status: done
- [x] **Implement `src/composite.ts` — explanation generation** — Generate human-readable explanation based on dominant contributing metric(s). If centroid dominant: semantic content shifted. If spread dominant: diversity changed. If output length dominant: verbosity changed. If vocabulary dominant: different words. If multiple contribute roughly equally: broad behavioral shift. | Status: done
- [ ] **Write `src/__tests__/composite.test.ts`** — Tests: verify weighted average formula with default weights against manually computed value; verify disabling a metric redistributes weight proportionally; verify severity classification: 0.03->none, 0.10->low, 0.25->medium, 0.45->high, 0.70->critical; verify explanation generation for each dominant metric scenario. | Status: not_done

---

## Phase 7: Alert System

- [x] **Implement `src/alert.ts` — alert threshold evaluation** — Alert fires when severity >= alertSeverity (default 'high') OR any per-metric score exceeds its configured threshold in `thresholds`. Set `report.alert = true` when triggered. | Status: done
- [x] **Implement `src/alert.ts` — onDrift callback dispatch** — Invoke the `onDrift` callback with the full DriftReport when alert fires. Do not await async callbacks (fire-and-forget). Swallow errors from the callback to prevent disruption. | Status: done
- [x] **Implement `src/alert.ts` — alert cooldown** — Track last alert timestamp per monitor instance. Suppress alerts within `alertCooldownMs` of the last fired alert. | Status: done
- [ ] **Write `src/__tests__/alert.test.ts`** — Tests: report.alert is false when severity < alertSeverity; report.alert is true when severity >= alertSeverity; report.alert is true when per-metric score exceeds its threshold even if severity is below alertSeverity; onDrift callback called when alert fires; onDrift not called when no alert; async onDrift errors are swallowed; cooldown suppresses repeated alerts within window; cooldown expires and allows new alerts after window passes. | Status: not_done

---

## Phase 8: DriftMonitor Class

- [x] **Implement `src/monitor.ts` — `createMonitor(options)` factory** — Validate required options (promptId, embedFn, embeddingModel). Merge defaults for optional options (alertSeverity='high', default weights, all metrics enabled, cooldownMs=0, histogramBins=20, pairwiseSamplePairs=500). Return a `DriftMonitor` instance. | Status: done
- [x] **Implement `monitor.snapshot(outputs, options?)`** — Delegate to `snapshot.ts` createSnapshot, passing embedFn, promptId, embeddingModel, and merged options. Return the Snapshot. | Status: done
- [x] **Implement `monitor.compare(baseline, current)`** — Validate snapshot compatibility (same dimensionality, same embeddingModel; throw INCOMPATIBLE_SNAPSHOTS if not). Run all enabled drift metrics. Compute composite score and severity. Generate explanation. Evaluate alert thresholds. Fire onDrift if alert. Return DriftReport with UUID, timestamp, promptId, snapshotIds, per-metric results, composite score, severity, alert flag, explanation, and comparison metadata. | Status: done
- [x] **Implement `monitor.setBaseline(snapshot)` and `monitor.getBaseline()`** — Store/retrieve the baseline snapshot on the monitor instance. getBaseline returns null if none set. | Status: done
- [x] **Implement `monitor.check(outputs, options?)`** — Throw `DriftError` with code `NO_BASELINE` if no baseline set. Create a snapshot from outputs. Compare against baseline using `compare()`. Return the DriftReport. | Status: done
- [x] **Write `src/__tests__/monitor.test.ts`** — Integration tests: createMonitor with valid options returns monitor; snapshot creates valid Snapshot; compare with identical snapshots returns low drift; compare with different snapshots returns elevated drift; setBaseline/getBaseline roundtrip; check throws NO_BASELINE when none set; check returns DriftReport when baseline set; compare throws INCOMPATIBLE_SNAPSHOTS for mismatched dimensionality or embeddingModel. | Status: done

---

## Phase 9: Active Probing

- [ ] **Implement `src/probe.ts` — concurrency-controlled LLM execution** — Run llmFn(prompt) `runs` times (default 30) with configurable concurrency (default 5). Use a simple semaphore/chunking pattern: split runs into batches of `concurrency` size, await each batch with `Promise.all`. Collect all output strings. | Status: not_done
- [x] **Implement `src/probe.ts` — `probe()` method integration** — After collecting outputs, create snapshot via monitor.snapshot(). If baseline exists, compare. If `setAsBaseline` is true and no baseline exists, call monitor.setBaseline(). Record timing: totalMs, llmCallsMs, embeddingMs, analysisMs. Return `ProbeResult`. | Status: done
- [ ] **Write `src/__tests__/probe.test.ts`** — Tests: probe calls llmFn the correct number of times; probe respects concurrency limit (verify with a mock that tracks concurrent invocations); probe creates snapshot from collected outputs; probe compares to baseline when one exists; probe auto-sets baseline when setAsBaseline=true and no baseline; probe returns null report when no baseline and setAsBaseline=false; timing fields are populated. | Status: not_done

---

## Phase 10: Serialization

- [ ] **Implement `src/serialization.ts` — `saveSnapshot(snapshot, filePath)`** — Write pretty-printed JSON (2-space indent) to the specified file path using `node:fs.writeFileSync`. | Status: not_done
- [ ] **Implement `src/serialization.ts` — `loadSnapshot(filePath)`** — Read file using `node:fs.readFileSync`, parse JSON, validate required fields (id, createdAt, promptId, sampleCount, embeddingModel, dimensionality, centroid, variance, spread, meanPairwiseSimilarity, stdPairwiseSimilarity, lengthStats, vocabularyStats, sampleEmbeddings, sampleOutputs). Throw `DriftError` with code `INVALID_SNAPSHOT` if validation fails. Return the parsed Snapshot. | Status: not_done
- [ ] **Write `src/__tests__/serialization.test.ts`** — Tests: saveSnapshot writes valid JSON parseable back to Snapshot; loadSnapshot reads back the same snapshot (deep equality); loadSnapshot throws INVALID_SNAPSHOT for malformed JSON; loadSnapshot throws INVALID_SNAPSHOT for JSON missing required fields (test each required field individually); round-trip: save then load produces identical snapshot. | Status: not_done

---

## Phase 11: CLI

- [ ] **Implement `src/cli.ts` — argument parser** — Parse `process.argv` manually (no external dependency). Support subcommands: `baseline`, `check`, `probe`, `report`. Parse flags: `--outputs`, `--output`, `--prompt`, `--model`, `--embed-provider`, `--api-key`, `--sample-size`, `--format`, `--baseline`, `--alert-severity`, `--prompt-file`, `--runs`, `--concurrency`, `--provider`, `--llm-model`, `--embed-model`, `--save-baseline`, `--input`. | Status: not_done
- [ ] **Implement CLI — built-in embed provider support** — Support `--embed-provider openai` and `--embed-provider cohere`. Use native `fetch` to call the embedding API (OpenAI: `https://api.openai.com/v1/embeddings`, Cohere: `https://api.cohere.com/v1/embed`). Read API key from `--api-key` flag or `OPENAI_API_KEY`/`COHERE_API_KEY` env vars. | Status: not_done
- [ ] **Implement CLI — `baseline` command** — Read outputs from `--outputs` JSON file. Create a snapshot using the built-in embed provider. Write snapshot JSON to `--output` path or stdout. Support `--format summary` and `--format json`. Print human-readable summary when format=summary. | Status: not_done
- [ ] **Implement CLI — `check` command** — Load baseline snapshot from `--baseline` path. Read new outputs from `--outputs` JSON file. Embed outputs. Compare. Write DriftReport to `--output` or stdout. Default format: summary. Support `--alert-severity` for exit code control. Exit code 1 when severity >= alert-severity. | Status: not_done
- [ ] **Implement CLI — `probe` command** — Read prompt from `--prompt` text or `--prompt-file` path. Support `--provider openai|anthropic` for LLM calls. Run probe with `--runs` and `--concurrency`. Load baseline from `--baseline` if provided. Support `--save-baseline` to save new baseline. Output format: summary or json. Exit code 1 when severity >= alert-severity. | Status: not_done
- [ ] **Implement CLI — `report` command** — Read a DriftReport JSON from `--input` path. Format and print as summary, json, or markdown. | Status: not_done
- [ ] **Implement CLI — exit codes** — Exit 0 for no significant drift. Exit 1 for drift at or above alert-severity. Exit 2 for configuration/usage errors (missing required options, invalid flags, unreadable files). | Status: not_done
- [ ] **Implement CLI — human-readable summary formatting** — Format the summary output matching the spec examples: prompt info, baseline info, per-metric drift scores with checkmark or ELEVATED marker, composite score, severity, interpretation, sample outputs when elevated, action recommendation. | Status: not_done
- [ ] **Implement CLI — error handling and usage messages** — Print helpful error messages for missing required options. Print usage help when no subcommand provided or `--help` flag used. | Status: not_done
- [ ] **Add hashbang to cli.ts** — Add `#!/usr/bin/env node` at the top of `src/cli.ts` so it can be executed directly. | Status: not_done

---

## Phase 12: Test Fixtures

- [ ] **Create `src/__fixtures__/stable-outputs.json`** — Pre-computed array of 30+ output strings simulating a stable LLM (consistent topic, vocabulary, length). | Status: not_done
- [ ] **Create `src/__fixtures__/drifted-outputs.json`** — Pre-computed array of 30+ output strings simulating a drifted LLM (different vocabulary, different length, different semantic content). | Status: not_done
- [ ] **Create `src/__fixtures__/mock-embeddings.json`** — Pre-computed embedding vectors corresponding to the fixture outputs. Use small dimensionality (e.g., 8 or 16 dims) for fast testing. | Status: not_done

---

## Phase 13: Integration Tests

- [ ] **Write `src/__tests__/integration/drift-detection.test.ts` — no-drift baseline test** — Generate two sets of outputs from the same mock LLM. Create snapshots. Compare. Verify composite score < 0.10 and severity is `none` or `low`. | Status: not_done
- [ ] **Write `src/__tests__/integration/drift-detection.test.ts` — full behavioral shift test** — Generate outputs from two mock LLMs with different behaviors (different vocabulary, length, semantic content). Create snapshots. Compare. Verify composite score > 0.40 and severity is `high` or `critical`. | Status: not_done
- [ ] **Write `src/__tests__/integration/drift-detection.test.ts` — probe round-trip test** — Call monitor.probe with setAsBaseline=true. Probe again with same mock LLM: verify drift < 0.10. Probe with different mock LLM: verify drift > 0.30. | Status: not_done
- [ ] **Write `src/__tests__/integration/drift-detection.test.ts` — end-to-end check() test** — Set baseline from stable outputs. Check with stable outputs (low drift). Check with drifted outputs (high drift). Verify alert fires for drifted case. | Status: not_done
- [ ] **Write `src/__tests__/integration/cli.test.ts` — CLI baseline command test** — Invoke via `child_process.execSync`. Verify JSON output is a valid snapshot. Verify exit code 0. | Status: not_done
- [ ] **Write `src/__tests__/integration/cli.test.ts` — CLI check command test (no drift)** — Create a baseline file. Run check with same outputs. Verify exit code 0. Verify summary output contains severity "none" or "low". | Status: not_done
- [ ] **Write `src/__tests__/integration/cli.test.ts` — CLI check command test (drift detected)** — Create a baseline file. Run check with drifted outputs. Verify exit code 1 (when alert-severity=high and drift is high). | Status: not_done
- [ ] **Write `src/__tests__/integration/cli.test.ts` — CLI error handling test** — Invoke with missing required flags. Verify exit code 2. | Status: not_done

---

## Phase 14: Time Window Support

- [ ] **Implement time-series buffer in `monitor.ts`** — Add an internal buffer that stores outputs with their timestamps. Configurable maximum duration (default: 90 days). Outputs older than the max are discarded on each insertion. | Status: not_done
- [ ] **Implement window duration parsing** — Parse window duration strings: `'1h'`, `'6h'`, `'12h'`, `'24h'`, `'7d'`, `'14d'`, `'30d'`, or raw millisecond numbers. Convert to milliseconds. | Status: not_done
- [ ] **Implement rolling window mode in `monitor.check()`** — When `options.window` is provided, add outputs to the time-series buffer. Construct baseline window snapshot from [T-2W, T-W] and current window snapshot from [T-W, T]. Compare the two window snapshots. | Status: not_done
- [ ] **Handle insufficient data in time windows** — If a window has fewer than 20 outputs, throw `DriftError` with code `INSUFFICIENT_OUTPUTS` and a message indicating the window period. | Status: not_done
- [ ] **Write tests for time window comparison** — Tests: rolling window mode correctly partitions outputs by time; window duration parsing handles all supported formats; insufficient data in a window throws; two identical windows produce low drift; windows with different data produce elevated drift. | Status: not_done

---

## Phase 15: Edge Cases and Error Handling

- [ ] **Handle embedFn returning wrong number of vectors** — If embedFn returns a different number of vectors than the number of inputs, throw a descriptive DriftError. | Status: not_done
- [ ] **Handle empty string outputs** — Outputs that are empty strings should be accepted (they are valid LLM outputs). Ensure tokenizer, length stats, and vocabulary stats handle empty strings gracefully. | Status: not_done
- [ ] **Handle very long outputs** — Ensure no issues with outputs that are extremely long (e.g., 100K characters). Vocabulary stats should not OOM on large token counts. | Status: not_done
- [ ] **Handle all-identical outputs** — When all N outputs are identical: spread=0, pairwise similarity=1, stdPairwiseSimilarity=0. Ensure no division by zero in spread change metric when both baselines have spread=0. Use epsilon in denominator. | Status: not_done
- [ ] **Handle single-dimension embeddings** — Ensure metrics work correctly with 1-dimensional embeddings (degenerate case). | Status: not_done
- [ ] **Handle outputs with no alphabetic content** — Outputs containing only numbers, symbols, or whitespace. Tokenizer should produce empty or minimal token lists. Vocabulary stats should handle zero vocabulary size gracefully. | Status: not_done
- [ ] **Handle NaN/Infinity in metric computations** — Add guards against NaN and Infinity in all metric computations (cosine similarity of zero vectors, log of zero, division by zero). Replace with 0 or 1 as appropriate. | Status: not_done

---

## Phase 16: Documentation

- [ ] **Create README.md** — Write comprehensive README with: package description, installation instructions, quick-start example, API reference for createMonitor/snapshot/compare/probe/check/setBaseline/getBaseline/saveSnapshot/loadSnapshot, CLI reference with all commands and flags, configuration reference table, integration examples (embed-cache, embed-drift, llm-regression, prompt-version, prompt-flags), and common use-case examples. | Status: not_done
- [ ] **Add JSDoc comments to all public functions** — Document all exported functions and types with JSDoc comments including param descriptions, return types, throws clauses, and usage examples. | Status: not_done

---

## Phase 17: Build and Publish Preparation

- [ ] **Verify TypeScript compilation** — Run `npm run build` and verify `dist/` output contains all .js, .d.ts, and .d.ts.map files. Verify no TypeScript errors. | Status: not_done
- [ ] **Verify lint passes** — Run `npm run lint` and fix any linting errors. Set up ESLint config if not already present. | Status: not_done
- [ ] **Verify full test suite passes** — Run `npm run test` (vitest) and confirm all tests pass. | Status: not_done
- [ ] **Verify package.json metadata** — Ensure `name`, `version`, `description`, `main`, `types`, `files`, `bin`, `engines`, `license`, `keywords`, and `publishConfig` are all correct. Add relevant keywords (e.g., "llm", "drift", "monitoring", "prompt", "embeddings", "semantic-drift"). | Status: not_done
- [ ] **Version bump** — Bump version in package.json per semver (0.1.0 -> appropriate version for initial release). | Status: not_done
- [ ] **Verify npm pack contents** — Run `npm pack --dry-run` and verify only `dist/` files are included (no src/, no tests, no fixtures). | Status: not_done
