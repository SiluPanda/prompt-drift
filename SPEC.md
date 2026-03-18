# prompt-drift -- Specification

## 1. Overview

`prompt-drift` detects when LLM outputs silently change over time by monitoring the statistical distribution of outputs for a given prompt and alerting when that distribution shifts beyond configurable thresholds. It does not require curated evaluation datasets, expected outputs, or ground truth labels. It needs only the prompt, access to the LLM, and time.

The gap this package fills is specific, well-defined, and currently unfilled. LLM providers update their models without notice. OpenAI's GPT-4 produced measurably different outputs between its March 2023 and June 2023 versions -- same API endpoint, same model name, same prompt, different behavior. Anthropic, Google, and every other provider do the same: model weights are updated, safety filters are adjusted, decoding strategies change, and the developer sees none of it. The API returns 200. The model name is unchanged. The outputs are different. Production applications that depend on consistent LLM behavior -- classification pipelines, extraction workflows, content generation systems, customer-facing chatbots with specific tone requirements -- silently degrade. The developer discovers the problem from user complaints, quality audits, or revenue drops, days or weeks after the change occurred.

This is the hardest category of production bug: the silent behavioral regression. There is no error, no version change, no deprecation notice. The system appears to work. The outputs are subtly wrong -- or substantially wrong, depending on what the provider changed. And today, in the npm ecosystem, there is no tool that monitors LLM output behavior over time.

The existing tools address adjacent but different problems. `embed-drift` (this monorepo) monitors embedding model changes -- it detects when the embedding function has changed by comparing embedding distributions. That is a different problem: same text, different vectors. `prompt-drift` monitors output behavior: same prompt, different text outputs. `llm-regression` (this monorepo) compares outputs between two explicit prompt versions -- baseline prompt A vs. candidate prompt B -- using curated test cases with expected outputs. That requires the developer to know that a change is happening and to have a test set. `prompt-drift` detects changes that the developer does not know about, without a test set. `prompt-snap` (this monorepo) compares a single output against a stored snapshot -- a point comparison, not a distributional one. `prompt-drift` monitors the distribution of outputs over time, detecting shifts that individual output comparisons miss.

In the broader ecosystem, Python has data drift detection tools -- Evidently AI, NannyML, whylogs -- that operate on tabular feature distributions for traditional ML models. None of them monitor LLM output distributions. They do not understand what a prompt is, what an LLM output is, or how to embed text outputs and track their semantic distribution. The npm ecosystem has nothing at all for this use case.

`prompt-drift` works through a distribution monitoring approach inspired by MLOps drift detection but designed specifically for LLM output text:

1. **Collect output samples**: Run a prompt against the LLM multiple times (active probing) or collect outputs from production logs (passive collection).
2. **Embed all outputs**: Generate embedding vectors for each output using a caller-provided embedding function.
3. **Compute a distribution snapshot**: Capture the statistical properties of the output embedding distribution -- centroid, spread, variance, output length statistics, vocabulary statistics.
4. **Compare against a baseline**: Apply statistical tests to quantify the distribution shift between the current snapshot and the baseline.
5. **Score and alert**: Compute a composite drift score, classify severity, and fire alerts when thresholds are exceeded.

No evaluation dataset is required. No expected outputs. No human labels. The system monitors its own behavior over time and alerts when that behavior changes. The baseline is "what the prompt used to do." The current state is "what the prompt does now." The drift score measures how different those two things are.

`prompt-drift` provides a programmatic TypeScript API and a CLI. The API returns structured `DriftReport` objects with per-metric drift scores, severity classifications, composite drift scores, explanations, and alert flags. The CLI creates baselines, probes prompts, checks drift, and prints reports as human-readable summaries or JSON. Both interfaces support configurable thresholds, alert callbacks, time window comparison, and integration with other packages in the monorepo.

---

## 2. Goals and Non-Goals

### Goals

- Provide a `createMonitor(options)` factory that returns a `DriftMonitor` instance encapsulating all drift detection state, baseline management, and configuration.
- Provide a `monitor.snapshot(outputs, options?)` method that embeds an array of LLM output strings and captures their distribution as a `Snapshot` object -- including embedding centroid, spread, variance, output length statistics, vocabulary statistics, and sample outputs.
- Provide a `monitor.compare(baseline, current)` method that computes a `DriftReport` quantifying the distribution shift between two snapshots, with per-metric drift scores, a composite score, severity classification, and a human-readable explanation.
- Provide a `monitor.probe(prompt, llmFn, options?)` method that actively probes a prompt by running it N times against the LLM, creates a snapshot of the outputs, compares to the stored baseline, and returns a `ProbeResult` containing the snapshot, the drift report, and the raw outputs.
- Provide a `monitor.check(newOutputs)` method that creates a snapshot from new outputs and compares it against the stored baseline, returning a `DriftReport`. This is the passive collection entry point: the caller provides outputs gathered from production logs.
- Provide `monitor.setBaseline(snapshot)` and `monitor.getBaseline()` methods for managing the reference distribution.
- Implement six drift metrics: centroid distance, spread change, Jensen-Shannon divergence, Population Stability Index (PSI), output length drift, and vocabulary drift. Combine them into a composite drift score in [0, 1].
- Classify drift severity into `none`, `low`, `medium`, `high`, `critical` bands with configurable thresholds.
- Support alerting via `onDrift(report)` callback hooks, with configurable severity thresholds and cooldown periods.
- Support time window comparison: compare outputs from the last N hours/days against a baseline or against a previous window of equal duration.
- Persist snapshots and baselines as portable JSON files. Provide convenience methods for filesystem serialization.
- Provide a CLI (`prompt-drift`) for baseline creation, active probing, drift checking, and report generation.
- Integrate with `embed-cache` (cache output embeddings), `embed-drift` (complementary monitoring), `llm-regression` (explicit version comparison when drift is detected), `prompt-version` (tag baselines with prompt versions), and `prompt-flags` (feature-flag prompt changes and monitor drift per variant).
- Zero mandatory runtime dependencies beyond a caller-provided embedding function and LLM function. All statistical computations are self-contained TypeScript. No native modules, no WASM, no Python bridge.
- Target Node.js 18 and above.

### Non-Goals

- **Not an LLM provider.** `prompt-drift` does not call any LLM API directly. It accepts a user-provided `llmFn` for active probing and operates on output strings that the caller has already obtained. Bring your own OpenAI client, Anthropic client, or local model.
- **Not an embedding provider.** `prompt-drift` does not ship an embedding model or call any embedding API. It accepts a user-provided `embedFn` for embedding output strings. The package is embedding-agnostic.
- **Not an evaluation framework.** `prompt-drift` detects that output behavior has changed. It does not evaluate whether the change is good or bad, whether outputs are factually correct, or whether they meet quality criteria. It detects drift -- the direction of that drift (improvement vs. degradation) requires human judgment or a separate evaluation tool like `output-grade` or `llm-regression`.
- **Not a prompt management tool.** `prompt-drift` does not store, version, or manage prompt templates. It monitors the outputs of a prompt over time. For prompt versioning, use `prompt-version`.
- **Not a continuous monitoring daemon.** `prompt-drift` performs point-in-time checks and returns. For continuous monitoring, wrap it in a cron job, scheduled worker, or background task that calls `monitor.probe()` or `monitor.check()` on a timer.
- **Not a general-purpose data drift framework.** `prompt-drift` is designed specifically for LLM output text distributions. It is not a tabular data drift tool and does not implement drift tests for scalar features, categorical variables, or non-text distributions. Use Evidently AI or NannyML for tabular drift.
- **Not a log aggregation system.** `prompt-drift` can consume outputs from production logs, but it does not instrument applications, collect logs, or manage log storage. The caller is responsible for gathering output strings and passing them to `monitor.check()`.
- **Not a replacement for explicit regression testing.** `prompt-drift` detects silent, unexpected changes. `llm-regression` tests explicit, intentional changes. Both are needed in a production LLM pipeline. They are complementary.

---

## 3. Target Users and Use Cases

### Production LLM Pipeline Operators

Teams running LLM-powered features in production -- classification, extraction, summarization, content generation, chatbots -- where consistent output behavior is a business requirement. The pipeline has been deployed, the prompt is stable, and the team expects stable outputs. When the LLM provider silently updates the model, outputs change without warning. `prompt-drift` runs a nightly or hourly probe: execute the production prompt N times, compare the output distribution against the baseline established at deployment, and alert if behavior has shifted. The operator learns about the model change within hours, not weeks.

### Prompt Engineers Maintaining Stable Prompts

Engineers who have invested significant effort tuning a prompt to produce outputs with specific characteristics -- a certain tone, a specific format, a consistent level of detail. They need assurance that their prompt continues to produce outputs in the same distribution, even as the underlying model evolves. `prompt-drift` provides this assurance: set a baseline when the prompt is finalized, monitor periodically, and get alerted when the distribution shifts.

### SLA Compliance Teams

Organizations that have contractual SLAs on LLM output behavior -- response format consistency, language compliance, content policy adherence. A silent model change that alters output behavior can cause SLA violations. `prompt-drift` provides the early warning system: detect the behavioral change before it causes a compliance breach.

### Model Update Detection Teams

Teams that need to know when their LLM provider has updated the model behind a stable API endpoint. Unlike `embed-drift` which detects embedding model changes via vector comparison, `prompt-drift` detects behavioral model changes by monitoring the outputs themselves. The approach is complementary: `embed-drift` catches changes in the embedding space, `prompt-drift` catches changes in the generation space.

### AI Quality Assurance Teams

QA teams responsible for validating that AI features continue to work as expected across deployments, model updates, and infrastructure changes. `prompt-drift` provides an automated, continuous quality signal: "has the behavior of this prompt changed?" The QA team sets baselines for each critical prompt, monitors drift scores over time, and investigates when scores exceed thresholds.

### CI/CD Pipeline Integrators

Teams that want to gate deployments on prompt stability. Before deploying a new version of their application (which may include infrastructure changes, dependency updates, or provider SDK upgrades that indirectly affect model behavior), a CI step probes critical prompts, checks for drift against the stored baseline, and fails the pipeline if significant drift is detected. The exit code (0 = no drift, 1 = drift detected) integrates with any CI system.

---

## 4. Core Concepts

### Silent LLM Behavior Changes

LLM providers routinely update their models without explicit version changes visible to the developer. These updates can include weight adjustments, safety filter changes, decoding strategy modifications, system prompt injections, and full model replacements behind the same API endpoint. The developer calls the same API, passes the same prompt, and receives different outputs. There is no error code, no deprecation warning, no changelog entry. The change is invisible at the API level and only observable in the statistical properties of the outputs.

The impact ranges from subtle (slightly different phrasing, minor tone shifts) to severe (different classification labels, different extracted entities, different format compliance). The severity depends on what the provider changed and how sensitive the application is to output variation.

### Output Distribution

An output distribution is the statistical population from which a set of LLM outputs is drawn. When a prompt is run N times against an LLM, each run produces an output string. These outputs vary (because LLMs are non-deterministic), but they cluster around characteristic patterns: similar topics, similar structure, similar length, similar vocabulary, similar semantic content. The collection of outputs forms a distribution in semantic space.

Two sets of outputs have the same distribution if they are drawn from the same behavioral regime -- same model, same prompt, same inference configuration. They have different distributions if the model, the prompt, or the inference configuration has changed.

### Snapshot

A snapshot is a compact statistical summary of an output distribution, captured at a specific point in time. It is not a copy of all outputs -- it is a set of statistics sufficient to detect distribution shifts when compared to a future snapshot.

A snapshot contains:
- **Prompt ID**: The identifier of the prompt being monitored. Used for organizational purposes and to ensure baselines are compared to the correct prompt.
- **Timestamp**: ISO 8601 creation time.
- **Sample count**: The number of outputs this snapshot was computed from.
- **Embedding centroid**: The mean embedding vector across all output embeddings. Represents the "center of gravity" of the output distribution in semantic space.
- **Embedding variance**: Per-dimension variance of the output embeddings. Captures how spread out the distribution is in each semantic direction.
- **Spread**: The average cosine distance from each output embedding to the centroid. A scalar summarizing the overall dispersion of outputs.
- **Pairwise similarity statistics**: Mean and standard deviation of cosine similarity between randomly sampled pairs of output embeddings. Captures the internal coherence of the output distribution.
- **Output length statistics**: Mean, standard deviation, median, p5, p95 of output lengths (in characters). Captures whether the model is producing longer or shorter responses.
- **Vocabulary statistics**: Term frequency distribution over the tokenized outputs. Top-K most frequent terms. Vocabulary size. These capture lexical shifts that may not be reflected in embedding space.
- **Sample embeddings**: A random sample of output embedding vectors, stored for statistical tests (JSD, PSI) that require raw samples.
- **Sample outputs**: A small random sample of raw output strings, stored for human review when drift is detected.
- **Embedding model**: The embedding model used to produce the vectors. Required for ensuring snapshots are comparable.
- **Metadata**: Caller-provided key-value pairs (e.g., model version, deployment ID, environment).

Snapshots are serialized to JSON and are portable across processes, machines, and time.

### Baseline

The baseline is the "known good" output distribution -- the distribution of outputs at the time the prompt was deployed, finalized, or last validated. All drift is measured relative to the baseline. The baseline answers the question: "what did this prompt used to do?"

A baseline is simply a snapshot that has been designated as the reference. It can be:
- **Established by probing**: Run the prompt N times at deployment, create a snapshot, set it as baseline.
- **Established from production logs**: Collect outputs from the first week of production, create a snapshot, set it as baseline.
- **Loaded from storage**: Read a previously saved snapshot from a JSON file.

Baselines should be refreshed when a prompt is intentionally changed. When `prompt-drift` detects drift and the team determines the new behavior is acceptable (or intentional), the current snapshot should replace the baseline.

### Drift Score

A drift score is a normalized measure of distribution shift in [0, 1]:
- **0.0**: No detectable drift. The current output distribution is statistically indistinguishable from the baseline by the methods tested.
- **1.0**: Complete drift. The distributions are as different as two unrelated distributions would be.

Drift scores are computed per-metric (centroid distance, spread change, JSD, PSI, output length drift, vocabulary drift) and combined into a composite drift score using configurable weights. Per-metric scores allow operators to understand which aspect of the distribution has shifted most.

### Drift Severity

Drift severity classifies the composite drift score into actionable bands:

| Score | Severity | Recommended Action |
|---|---|---|
| 0.00 -- 0.05 | `none` | No action needed. Distribution is stable. Normal LLM output variation. |
| 0.05 -- 0.15 | `low` | Monitor. Minor variation within expected bounds. Log for trend tracking. |
| 0.15 -- 0.35 | `medium` | Investigate. Output behavior has shifted. Review sample outputs. May indicate a minor model update. |
| 0.35 -- 0.60 | `high` | Action recommended. Significant behavioral change detected. Review outputs, consider re-establishing baseline or adjusting prompt. |
| 0.60 -- 1.00 | `critical` | Immediate action required. Major behavioral shift. Likely a model update or infrastructure change. Audit all outputs since baseline. |

### Time Window

A time window is a period of time over which outputs are collected and summarized into a snapshot. Time windows enable rolling comparisons: "how do this week's outputs compare to last week's?" or "how do the last 24 hours compare to the previous 24 hours?" This is the primary mechanism for continuous monitoring without maintaining a static baseline.

### Active Probing vs. Passive Collection

`prompt-drift` supports two modes of output collection:

**Active probing**: The monitor runs the prompt against the LLM N times, collects the responses, and analyzes them. This is controlled, repeatable, and does not depend on production traffic. The cost is N LLM API calls per probe. Use for scheduled health checks and CI gates.

**Passive collection**: The caller instruments their production code to collect outputs, then passes those outputs to `monitor.check()`. No additional LLM calls are made. The outputs reflect actual production behavior with real user inputs. Use for continuous production monitoring.

---

## 5. How Drift Detection Works

### The Full Pipeline

Drift detection follows a six-step pipeline from output collection to alert. Each step transforms data into a progressively more compact and actionable form.

### Step 1: Collect Output Samples

Gather a representative sample of LLM outputs for the prompt being monitored.

**Active probing** (`monitor.probe`): The monitor executes the prompt N times (default: 30) against the user-provided LLM function. Each execution uses the same prompt text. The LLM's natural non-determinism produces varied outputs. The N outputs form the sample.

**Passive collection** (`monitor.check`): The caller provides an array of output strings gathered from production. These may have been produced from different user inputs (which is expected -- the distribution includes the effect of input variation). The caller is responsible for gathering a representative sample.

**Replay** (manual workflow): The caller stores historical inputs, re-runs them against the current model, and passes the new outputs to `monitor.check()` alongside a baseline snapshot from the original run. This isolates model changes from input distribution changes.

**Sample size**: The statistical power of drift detection increases with sample size. A minimum of 20 outputs is required. 30 is the default for active probing. 50-100 provides good sensitivity for subtle drift. 200+ provides high sensitivity. The diminishing returns threshold is approximately 200 for most metrics.

### Step 2: Embed All Outputs

Each output string is embedded using the caller-provided `embedFn`, producing an embedding vector for each output.

```
outputs = ["The capital of France is Paris.", "Paris is the capital.", ...]
embeddings = await embedFn(outputs)
// embeddings: [[0.12, -0.34, ...], [0.11, -0.33, ...], ...]
```

The embedding model must be the same across all snapshots being compared. If the embedding model changes, the snapshots become incompatible (use `embed-drift` to detect this). The embedding model is recorded in the snapshot metadata.

### Step 3: Compute Distribution Snapshot

From the N embedding vectors and the raw output strings, compute a statistical snapshot of the distribution:

1. **Centroid**: Element-wise mean of all embedding vectors. The center of the semantic cloud.
2. **Variance**: Per-dimension variance of the embedding vectors.
3. **Spread**: Mean cosine distance from each embedding to the centroid.
4. **Pairwise similarity**: Sample M pairs, compute cosine similarity for each, record mean and standard deviation.
5. **Output length statistics**: Compute mean, standard deviation, median, p5, p95 of output string lengths.
6. **Vocabulary statistics**: Tokenize all outputs, compute term frequencies, record top-K terms and vocabulary size.
7. **Sample embeddings**: Randomly select `sampleSize` (default: 50) embedding vectors for storage.
8. **Sample outputs**: Randomly select `outputSampleSize` (default: 10) raw output strings for storage.

### Step 4: Compare Against Baseline

Apply each configured drift metric to quantify the shift between the baseline snapshot and the current snapshot. Each metric examines a different aspect of the distribution and produces an independent score in [0, 1].

### Step 5: Compute Composite Drift Score

Combine per-metric scores into a single composite score using configurable weights. Classify the composite score into a severity band. Generate a human-readable explanation of the drift.

### Step 6: Alert If Threshold Exceeded

If the composite drift score exceeds the configured alert threshold (or severity meets or exceeds the configured alert severity), invoke the `onDrift` callback with the full drift report. If cooldown is configured, suppress repeated alerts within the cooldown window.

### Pipeline Diagram

```
┌──────────────────────┐
│  Output Collection   │
│                      │
│  Active probing:     │     ┌─────────────────┐
│  prompt × N → LLM ──────→│  N output strings │
│                      │     └────────┬────────┘
│  Passive collection: │              │
│  production logs ────────→          │
└──────────────────────┘              │
                                      ▼
                            ┌─────────────────┐
                            │  Embed Outputs   │
                            │  embedFn(outputs)│
                            └────────┬────────┘
                                     │
                                     ▼
                            ┌─────────────────────┐
                            │  Compute Snapshot    │
                            │  centroid, spread,   │
                            │  variance, lengths,  │
                            │  vocabulary, samples │
                            └────────┬────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                        ▼                         ▼
               ┌─────────────┐          ┌──────────────┐
               │  Baseline   │          │   Current    │
               │  Snapshot   │          │   Snapshot   │
               └──────┬──────┘          └──────┬───────┘
                      │                        │
                      └───────────┬────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  Compare:       │
                        │  6 drift metrics│
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Composite Score│
                        │  + Severity     │
                        └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  Alert?         │
                        │  → onDrift()    │
                        └─────────────────┘
```

---

## 6. Snapshot Statistics

### What a Snapshot Captures

A snapshot is a compact, portable record of an LLM output distribution's statistical state at a point in time. It stores enough information to:
1. Detect semantic drift (centroid, variance, spread, sample embeddings).
2. Detect structural drift (output length statistics).
3. Detect lexical drift (vocabulary statistics).
4. Provide human-reviewable evidence (sample outputs).
5. Ensure comparability (embedding model ID, dimensionality).

### Snapshot Schema

```typescript
interface Snapshot {
  /** Unique identifier for this snapshot, UUID v4. */
  id: string;

  /** ISO 8601 timestamp of when this snapshot was created. */
  createdAt: string;

  /** Identifier for the prompt being monitored. */
  promptId: string;

  /** The number of outputs this snapshot was computed from. */
  sampleCount: number;

  /** The embedding model used to produce the vectors. */
  embeddingModel: string;

  /** The number of dimensions in each embedding vector. */
  dimensionality: number;

  /** Element-wise mean of all output embedding vectors. Length: dimensionality. */
  centroid: number[];

  /** Per-dimension variance of the output embeddings. Length: dimensionality. */
  variance: number[];

  /** Mean cosine distance from each embedding to the centroid. */
  spread: number;

  /** Mean pairwise cosine similarity across randomly sampled pairs. */
  meanPairwiseSimilarity: number;

  /** Standard deviation of pairwise cosine similarities. */
  stdPairwiseSimilarity: number;

  /** Output length statistics (in characters). */
  lengthStats: {
    mean: number;
    stddev: number;
    median: number;
    p5: number;
    p95: number;
  };

  /** Vocabulary statistics from tokenized outputs. */
  vocabularyStats: {
    /** Total unique terms across all outputs. */
    vocabularySize: number;
    /** Top K most frequent terms with their frequencies. */
    topTerms: Array<{ term: string; frequency: number }>;
    /** Total token count across all outputs. */
    totalTokens: number;
  };

  /**
   * A random sample of output embedding vectors.
   * Used for JSD and PSI computation.
   * Length: min(sampleSize, sampleCount). Each entry: number[] of length dimensionality.
   */
  sampleEmbeddings: number[][];

  /**
   * A random sample of raw output strings for human review.
   * Length: min(outputSampleSize, sampleCount).
   */
  sampleOutputs: string[];

  /** Caller-provided metadata. Passed through without modification. */
  metadata?: Record<string, unknown>;
}
```

### Creating a Snapshot

`monitor.snapshot(outputs, options?)` accepts an array of output strings, embeds them, and returns a `Snapshot`. The method:

1. Validates input: at least 20 output strings are required for meaningful statistics.
2. Embeds all outputs using the configured `embedFn`: `const embeddings = await embedFn(outputs)`.
3. Validates embeddings: all vectors must have the same dimensionality.
4. Computes centroid: element-wise mean of all embedding vectors.
5. Computes per-dimension variance.
6. Computes spread: mean cosine distance from each embedding to the centroid.
7. Samples M pairs (M = min(500, n*(n-1)/2)) and computes pairwise cosine similarity statistics.
8. Computes output length statistics: mean, standard deviation, median, p5, p95 of `output.length` for each output string.
9. Computes vocabulary statistics: tokenize all outputs (split on whitespace, lowercase, remove punctuation), count term frequencies, record top-K terms (default K=100) and vocabulary size.
10. Randomly samples `sampleSize` (default: 50) embedding vectors using reservoir sampling.
11. Randomly samples `outputSampleSize` (default: 10) raw output strings using reservoir sampling.
12. Assigns a UUID, timestamps, and attaches the prompt ID, embedding model, and caller-provided metadata.

Snapshot creation is asynchronous (due to the embedding step) and completes in O(n * d) computation time plus the embedding API latency for n outputs and d dimensions.

### Snapshot Size

A snapshot with 1536-dimensional embeddings, 50 sample vectors, and 10 sample outputs is approximately:
- Centroid: 1536 floats * 8 bytes = 12 KB
- Variance: 1536 floats * 8 bytes = 12 KB
- Sample embeddings (50 * 1536): ~600 KB
- Vocabulary stats: ~5-20 KB (depending on vocabulary size)
- Sample outputs: variable (~2-10 KB for typical LLM outputs)
- Total (uncompressed JSON): approximately 650-700 KB

For 3072-dimensional embeddings, approximately double the embedding-related sizes. Compressing with gzip reduces size by approximately 60-70%.

---

## 7. Drift Metrics

`prompt-drift` implements six drift metrics. Each produces a normalized score in [0, 1]. They are complementary: some capture semantic shifts, others capture structural or lexical changes. Together they provide a comprehensive view of how the output distribution has changed.

### Metric 1: Centroid Distance

**What it measures**: Whether the average semantic content of outputs has shifted. The centroid of the output embedding distribution represents the "typical" output in semantic space. If the centroid moves, the average meaning of outputs has changed.

**Algorithm**:
```
centroid_A = baseline_snapshot.centroid
centroid_B = current_snapshot.centroid
centroid_distance = 1 - cosine_similarity(centroid_A, centroid_B)
centroid_drift_score = min(centroid_distance / normalization_constant, 1.0)
```

The normalization constant is calibrated so that a cosine distance of 0.1 (which represents a meaningful semantic shift for LLM outputs) maps to approximately 0.5 in drift score. Default normalization constant: 0.2.

**Interpretation**: A centroid drift score of 0 means the average output is semantically identical. A score above 0.3 indicates the average output has shifted meaningfully. A score above 0.6 indicates a substantial change in the typical output.

**Sensitivity**: Good for detecting broad topical or semantic shifts. If the model starts producing outputs about a different aspect of the topic, or uses a fundamentally different framing, centroid drift catches it. Less sensitive to changes that preserve the mean but alter the variance or tails of the distribution (e.g., the model becomes more or less creative while staying on topic).

**Computational cost**: O(d) where d is the embedding dimensionality. Effectively instantaneous -- one cosine similarity computation on the pre-computed centroids.

**Weight in composite**: 0.25 (default).

### Metric 2: Spread Change

**What it measures**: Whether the dispersion of outputs has changed. The spread measures how varied the outputs are -- a tight distribution (low spread) means the model consistently produces similar outputs; a wide distribution (high spread) means outputs vary significantly from run to run. A change in spread indicates the model has become more or less consistent.

**Algorithm**:
```
spread_A = baseline_snapshot.spread
spread_B = current_snapshot.spread
spread_ratio = max(spread_A, spread_B) / (min(spread_A, spread_B) + epsilon)
spread_drift_score = min((spread_ratio - 1.0) / normalization_constant, 1.0)
```

The normalization constant is calibrated so that a 50% change in spread (ratio of 1.5) maps to approximately 0.5 in drift score. Default normalization constant: 1.0.

**Interpretation**: A score of 0 means the spread is identical. A score above 0.3 means the model has become meaningfully more or less varied in its outputs. This is distinct from centroid drift: the average output can stay the same while the variation around that average changes.

**Sensitivity**: Sensitive to changes in model temperature, decoding strategy, or safety filters that affect output diversity. Less sensitive to shifts that move the distribution without changing its shape.

**Computational cost**: O(1). Comparison of two pre-computed scalars.

**Weight in composite**: 0.10 (default).

### Metric 3: Jensen-Shannon Divergence (JSD)

**What it measures**: The overall distributional difference between two sets of output embeddings. JSD is a symmetric, bounded measure of divergence between two probability distributions. It captures all moments of the distribution difference -- mean, variance, skewness, multimodal structure -- making it a comprehensive single metric.

**Algorithm**:

Because embedding vectors are continuous and high-dimensional, JSD cannot be computed on the raw vectors directly. `prompt-drift` discretizes the embedding space using a binning approach over the sample embeddings:

1. Compute cosine distances from each sample embedding (in both baseline and current snapshots) to the baseline centroid.
2. Bin these distances into K bins (default: 20) spanning the observed range.
3. Construct two discrete probability distributions P (baseline) and Q (current) from the bin counts, normalized to sum to 1. Add a smoothing constant (1e-10) to avoid log(0).
4. Compute JSD:

```
M = 0.5 * (P + Q)
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
where KL(P || M) = sum(P[i] * log(P[i] / M[i]))
```

5. Normalize: JSD is bounded in [0, log(2)] for natural logarithm. Divide by log(2) to normalize to [0, 1]:

```
jsd_drift_score = JSD(P || Q) / log(2)
```

**Interpretation**: A JSD score of 0 means the distributions are identical. A score above 0.2 indicates meaningful distributional difference. A score above 0.5 indicates the distributions are substantially different. JSD saturates at 1.0 for completely non-overlapping distributions.

**Sensitivity**: High sensitivity to distributional changes of all kinds -- shifts, spreads, shape changes, multimodal emergence. Because it operates on the full distribution shape (via binning), it catches changes that centroid distance and spread change miss individually.

**Computational cost**: O(s * d) where s is the sample size and d is the dimensionality (for computing distances to centroid), plus O(K) for the JSD computation itself. For s=50 and d=1536, approximately 1-2ms.

**Weight in composite**: 0.25 (default).

### Metric 4: Population Stability Index (PSI)

**What it measures**: The magnitude of distributional shift, using the standard MLOps metric for monitoring data drift. PSI is a standard metric in credit scoring and model monitoring that quantifies how much a population has shifted. Applied to LLM output embeddings, it detects the same kind of distributional shift that MLOps teams track for feature distributions.

**Algorithm**:

Like JSD, PSI requires discretization. `prompt-drift` uses the same binning approach:

1. Compute cosine distances from sample embeddings to the baseline centroid.
2. Bin into K bins (default: 20).
3. Construct distributions P (baseline) and Q (current), smoothed with epsilon.
4. Compute PSI:

```
PSI = sum((Q[i] - P[i]) * ln(Q[i] / P[i]))
```

5. Normalize: PSI is unbounded (it can exceed 1.0 for extreme shifts). Normalize using a sigmoid-like mapping:

```
psi_drift_score = min(PSI / normalization_constant, 1.0)
```

Default normalization constant: 0.5 (a PSI of 0.25 is considered "significant shift" in MLOps practice and maps to approximately 0.5 in drift score).

**Interpretation**: In standard MLOps practice:
- PSI < 0.10: No significant shift.
- PSI 0.10 -- 0.25: Moderate shift, investigation recommended.
- PSI > 0.25: Significant shift, action required.

These thresholds are embedded in the normalization: a raw PSI of 0.10 maps to approximately 0.2 drift score, and 0.25 maps to approximately 0.5.

**Sensitivity**: Similar to JSD but weights tail differences more heavily due to the logarithmic ratio term. PSI is particularly good at detecting cases where one distribution has probability mass in regions where the other has near-zero mass -- which corresponds to the model producing outputs in entirely new semantic regions.

**Computational cost**: Same as JSD -- O(s * d + K). Approximately 1-2ms.

**Weight in composite**: 0.15 (default).

### Metric 5: Output Length Drift

**What it measures**: Whether the model is producing longer or shorter outputs. Output length is a simple but surprisingly effective proxy for behavioral change. When a model is updated, one of the most common observable effects is a change in output verbosity -- the model may become more concise, more verbose, or start including additional disclaimers, qualifications, or formatting.

**Algorithm**:
```
mean_A = baseline_snapshot.lengthStats.mean
mean_B = current_snapshot.lengthStats.mean
std_A  = baseline_snapshot.lengthStats.stddev
std_B  = current_snapshot.lengthStats.stddev

// Normalized mean difference (Cohen's d)
pooled_std = sqrt((std_A^2 + std_B^2) / 2)
cohens_d = |mean_A - mean_B| / (pooled_std + epsilon)

// Variance ratio
var_ratio = max(std_A, std_B) / (min(std_A, std_B) + epsilon)

length_drift_score = min((0.7 * cohens_d + 0.3 * (var_ratio - 1.0)) / normalization_constant, 1.0)
```

Default normalization constant: 2.0 (a Cohen's d of 1.0 -- a large effect size -- maps to approximately 0.35 drift score).

**Interpretation**: A score of 0 means output lengths are statistically identical. A score above 0.2 means the model is producing meaningfully different length outputs. This metric is deliberately given lower weight because length changes often accompany but do not fully characterize behavioral shifts.

**Sensitivity**: High sensitivity to verbosity changes, formatting changes (adding/removing markdown, bullet points), and disclaimer injection. Low sensitivity to semantic changes that preserve output length.

**Computational cost**: O(1). Comparison of pre-computed scalars.

**Weight in composite**: 0.10 (default).

### Metric 6: Vocabulary Drift

**What it measures**: Whether the model is using different words. A change in vocabulary indicates the model has shifted its language patterns -- using new terminology, dropping old terminology, or changing the frequency of common terms. This captures lexical-level changes that embedding-based metrics may miss (because embeddings can map different words to similar regions of semantic space).

**Algorithm**:
```
// Term frequency vectors (normalized to sum to 1)
tf_baseline = baseline_snapshot.vocabularyStats.topTerms (as frequency distribution)
tf_current  = current_snapshot.vocabularyStats.topTerms (as frequency distribution)

// Union vocabulary across both snapshots
union_vocab = union(keys(tf_baseline), keys(tf_current))

// Compute cosine distance between term frequency vectors
P = [tf_baseline[term] or 0 for term in union_vocab]
Q = [tf_current[term] or 0 for term in union_vocab]
vocab_cosine_distance = 1 - cosine_similarity(P, Q)

// New term ratio: fraction of current top terms not in baseline top terms
new_terms = count(term in tf_current.topTerms where term not in tf_baseline.topTerms)
new_term_ratio = new_terms / len(tf_current.topTerms)

vocabulary_drift_score = min(
  (0.6 * vocab_cosine_distance + 0.4 * new_term_ratio) / normalization_constant,
  1.0
)
```

Default normalization constant: 0.5.

**Interpretation**: A score of 0 means the model is using the same words at the same frequencies. A score above 0.2 indicates new vocabulary is appearing or old vocabulary is disappearing. A score above 0.5 indicates a substantial lexical shift. This metric is useful for detecting changes in domain language, jargon adoption, or stylistic shifts that may not register in embedding-based metrics.

**Sensitivity**: High sensitivity to lexical changes -- new terminology, dropped terminology, frequency shifts. Complementary to embedding-based metrics: embeddings capture semantic similarity even with different words, while vocabulary drift captures word-level changes even with similar semantics.

**Computational cost**: O(V) where V is the union vocabulary size. Typically 100-1000 terms. Under 1ms.

**Weight in composite**: 0.15 (default).

### Metric Summary

| Metric | Captures | Speed | Default Weight | Best For |
|---|---|---|---|---|
| Centroid distance | Semantic shift of average output | < 1ms | 0.25 | Topical changes, framing shifts |
| Spread change | Variation in output diversity | < 1ms | 0.10 | Temperature/decoding changes, consistency shifts |
| Jensen-Shannon divergence | Full distributional shape change | ~2ms | 0.25 | Comprehensive distributional change |
| Population Stability Index | Distributional shift magnitude | ~2ms | 0.15 | Tail changes, new output modes |
| Output length drift | Verbosity changes | < 1ms | 0.10 | Formatting changes, disclaimer injection |
| Vocabulary drift | Lexical changes | < 1ms | 0.15 | Terminology shifts, style changes |

---

## 8. Composite Drift Score and Severity

### Per-Metric Scores

Each drift metric produces a score in [0, 1]. The scores are independent and normalized to the same scale despite measuring fundamentally different aspects of the distribution.

| Metric | Score = 0 | Score = 1 | Normalization basis |
|---|---|---|---|
| Centroid distance | Identical centroids | Orthogonal centroids | Cosine distance / 0.2 |
| Spread change | Identical spread | Spread ratio >=2x | (ratio - 1) / 1.0 |
| JSD | Identical distributions | Non-overlapping distributions | JSD / log(2) |
| PSI | Identical distributions | Extreme shift | PSI / 0.5 |
| Output length | Identical length distributions | Cohen's d >= 2.0 | Combined d and var ratio / 2.0 |
| Vocabulary | Identical vocabulary | Completely different vocabulary | Combined cosine + new ratio / 0.5 |

### Composite Drift Score

The composite drift score is a weighted average of the per-metric scores:

```
composite_score = sum(weight[metric] * score[metric]) / sum(weight[metric])
```

Default weights:
- Centroid distance: 0.25 (highest weight alongside JSD -- the most informative semantic signals)
- JSD: 0.25
- PSI: 0.15
- Vocabulary drift: 0.15
- Spread change: 0.10
- Output length drift: 0.10

If the caller disables specific metrics, their weights are redistributed proportionally to the remaining enabled metrics.

### Severity Classification

```typescript
type DriftSeverity = 'none' | 'low' | 'medium' | 'high' | 'critical';

function classifySeverity(score: number): DriftSeverity {
  if (score < 0.05) return 'none';
  if (score < 0.15) return 'low';
  if (score < 0.35) return 'medium';
  if (score < 0.60) return 'high';
  return 'critical';
}
```

### Drift Report Explanation

Every `DriftReport` includes a `explanation` field -- a human-readable string describing what changed and how significant it is. The explanation is generated programmatically from the per-metric scores:

- If centroid distance is the dominant contributor: "The average semantic content of outputs has shifted. The model may be producing outputs about a different aspect of the topic or using a fundamentally different framing."
- If spread change is the dominant contributor: "The diversity of outputs has changed. The model is producing more (or less) varied responses than baseline."
- If output length drift is the dominant contributor: "The model is producing significantly longer (or shorter) outputs. This may indicate a formatting change, added disclaimers, or a different level of detail."
- If vocabulary drift is the dominant contributor: "The model is using different vocabulary. New terminology is appearing or established terms are being dropped."
- If multiple metrics contribute roughly equally: "Broad behavioral shift detected across multiple dimensions (semantic content, output structure, and vocabulary)."

---

## 9. Time Window Comparison

### How Time Windows Work

Instead of comparing against a single static baseline, time window comparison divides outputs into temporal buckets and compares adjacent windows:

```
Window A (baseline):   outputs from [T - 2*W, T - W]
Window B (current):    outputs from [T - W, T]
```

Where T is the current time and W is the window duration.

### Configurable Windows

```typescript
monitor.check(outputs, {
  window: {
    duration: '7d',       // 7-day windows
    baseline: 'previous', // compare to the previous window of equal duration
  },
});
```

Supported window durations: `'1h'`, `'6h'`, `'12h'`, `'24h'`, `'7d'`, `'14d'`, `'30d'`, or a number of milliseconds.

### Rolling Window Mode

In rolling window mode, each `monitor.check()` call adds the provided outputs to an internal time-series buffer. The monitor automatically constructs snapshots for the baseline and current windows and compares them. This enables continuous monitoring with a single API: call `monitor.check(todaysOutputs)` daily, and the monitor internally compares this week to last week.

The time-series buffer has a configurable maximum duration (default: 90 days). Outputs older than this are discarded.

### Static Baseline Mode

In static baseline mode (the default), `monitor.check()` always compares against the explicitly set baseline snapshot. The baseline does not change unless `monitor.setBaseline()` is called. This is simpler and appropriate for detecting drift relative to a known-good deployment state.

---

## 10. Output Collection

### Active Probing

`monitor.probe(prompt, llmFn, options?)` executes the prompt against the LLM multiple times and returns a `ProbeResult`:

```typescript
const result = await monitor.probe(
  'Summarize the concept of machine learning in 2-3 sentences.',
  llmFn,
  { runs: 50, temperature: 0.7 },
);
```

**How probing works**:
1. Execute `llmFn(prompt)` a total of `runs` times (default: 30). Calls are made with configurable concurrency (default: 5 concurrent calls) to respect rate limits.
2. Collect all N output strings.
3. Create a snapshot from the outputs.
4. If a baseline exists, compare the snapshot to the baseline and produce a `DriftReport`.
5. Return a `ProbeResult` containing the outputs, the snapshot, and the drift report (if baseline exists).

**Probe options**:
- `runs` (default: 30): Number of LLM calls.
- `concurrency` (default: 5): Maximum concurrent LLM calls.
- `temperature` (optional): Suggested temperature for the LLM call. The caller's `llmFn` is responsible for applying this.
- `setAsBaseline` (default: false): If true and no baseline exists, automatically set the resulting snapshot as the baseline.

### Passive Collection

`monitor.check(outputs)` accepts an array of output strings collected from production:

```typescript
// Collect outputs from production logs
const todaysOutputs = await getOutputsFromLogs({ prompt: 'classify-intent', since: '24h' });
const report = await monitor.check(todaysOutputs);
```

The caller is responsible for:
- Filtering outputs to the correct prompt.
- Ensuring a representative sample (not biased toward a particular time of day, user segment, or input type).
- Providing enough outputs for meaningful statistics (minimum 20, recommended 50+).

### Replay

Replay is not a built-in API but a supported workflow:

1. Store historical inputs and their outputs in a database.
2. When testing for drift, re-run the stored inputs against the current model.
3. Compare the new outputs to the stored outputs using `monitor.compare()`.

This isolates model changes from input distribution changes -- the same inputs produce different outputs only if the model has changed. It is the most controlled form of drift detection but requires storing historical inputs.

---

## 11. API Surface

### Installation

```bash
npm install prompt-drift
```

### Factory: `createMonitor`

```typescript
import { createMonitor } from 'prompt-drift';

const monitor = createMonitor({
  promptId: 'classify-intent',
  embedFn: async (texts) => {
    const resp = await openai.embeddings.create({
      model: 'text-embedding-3-small',
      input: texts,
    });
    return resp.data.map(d => d.embedding);
  },
  embeddingModel: 'text-embedding-3-small',
});
```

**Signature:**
```typescript
function createMonitor(options: MonitorOptions): DriftMonitor;
```

### `monitor.snapshot`

Creates a distribution snapshot from an array of LLM output strings.

```typescript
const outputs = ['Output 1...', 'Output 2...', /* ... */];
const snap = await monitor.snapshot(outputs);
// snap.centroid, snap.spread, snap.lengthStats, snap.vocabularyStats
```

**Signature:**
```typescript
snapshot(
  outputs: string[],
  options?: SnapshotOptions,
): Promise<Snapshot>;
```

**Options:**
```typescript
interface SnapshotOptions {
  /** Number of embedding vectors to store as sample. Default: 50. */
  sampleSize?: number;

  /** Number of raw output strings to store as sample. Default: 10. */
  outputSampleSize?: number;

  /** Number of top vocabulary terms to record. Default: 100. */
  topK?: number;

  /** Caller-provided metadata to attach to the snapshot. Default: {}. */
  metadata?: Record<string, unknown>;
}
```

**Throws** `DriftError` if:
- Input array contains fewer than 20 outputs.
- `embedFn` returns vectors with inconsistent dimensionality.
- `embedFn` throws an error.

### `monitor.compare`

Computes a `DriftReport` quantifying the distribution shift between two snapshots.

```typescript
const report = monitor.compare(baselineSnapshot, currentSnapshot);
console.log(report.driftScore);   // 0.28
console.log(report.severity);     // 'medium'
console.log(report.explanation);  // 'The average semantic content of outputs has shifted...'
console.log(report.alert);        // false (severity < alertSeverity)
```

**Signature:**
```typescript
compare(baseline: Snapshot, current: Snapshot): DriftReport;
```

**Throws** `DriftError` with code `INCOMPATIBLE_SNAPSHOTS` if:
- The two snapshots have different embedding dimensionalities.
- The two snapshots use different embedding models.

### `monitor.probe`

Actively probes a prompt by running it N times and comparing to baseline.

```typescript
const result = await monitor.probe(
  'Classify the following text as positive, negative, or neutral.',
  async (prompt) => {
    const res = await openai.chat.completions.create({
      model: 'gpt-4o',
      messages: [{ role: 'user', content: prompt }],
    });
    return res.choices[0].message.content ?? '';
  },
  { runs: 50 },
);

console.log(result.outputs.length);       // 50
console.log(result.snapshot.spread);       // 0.12
console.log(result.report?.driftScore);    // 0.31 (if baseline exists)
console.log(result.report?.severity);      // 'medium'
```

**Signature:**
```typescript
probe(
  prompt: string,
  llmFn: LlmFn,
  options?: ProbeOptions,
): Promise<ProbeResult>;
```

**Options:**
```typescript
interface ProbeOptions {
  /** Number of times to run the prompt. Default: 30. */
  runs?: number;

  /** Maximum concurrent LLM calls. Default: 5. */
  concurrency?: number;

  /** If true and no baseline exists, set this probe's snapshot as baseline. Default: false. */
  setAsBaseline?: boolean;

  /** Snapshot options for the resulting snapshot. */
  snapshotOptions?: SnapshotOptions;

  /** Caller-provided metadata. */
  metadata?: Record<string, unknown>;
}
```

### `monitor.check`

Compares new outputs against the stored baseline. This is the passive collection entry point.

```typescript
const outputs = await getOutputsFromProductionLogs();
const report = await monitor.check(outputs);
if (report.alert) {
  console.log('Drift detected:', report.explanation);
}
```

**Signature:**
```typescript
check(
  outputs: string[],
  options?: CheckOptions,
): Promise<DriftReport>;
```

**Options:**
```typescript
interface CheckOptions {
  /** Snapshot options for the new snapshot. */
  snapshotOptions?: SnapshotOptions;
}
```

**Throws** `DriftError` with code `NO_BASELINE` if `monitor.setBaseline()` has not been called.

### `monitor.setBaseline`

Stores a snapshot as the baseline for subsequent `check()` and `probe()` comparisons.

```typescript
monitor.setBaseline(snapshot);
```

**Signature:**
```typescript
setBaseline(snapshot: Snapshot): void;
```

### `monitor.getBaseline`

Returns the current baseline snapshot, or `null` if no baseline has been set.

```typescript
const baseline = monitor.getBaseline();
```

**Signature:**
```typescript
getBaseline(): Snapshot | null;
```

### `monitor.saveSnapshot` / `monitor.loadSnapshot`

Convenience methods for filesystem serialization.

```typescript
monitor.saveSnapshot(snapshot, './baselines/classify-intent-2026-03-18.json');
const loaded = monitor.loadSnapshot('./baselines/classify-intent-2026-03-18.json');
monitor.setBaseline(loaded);
```

**Signatures:**
```typescript
saveSnapshot(snapshot: Snapshot, filePath: string): void;
loadSnapshot(filePath: string): Snapshot;
```

Both use Node.js synchronous `fs` APIs. `saveSnapshot` writes pretty-printed JSON. `loadSnapshot` reads and parses JSON, validates the snapshot schema, and throws `DriftError` with code `INVALID_SNAPSHOT` if the file does not conform.

### Type Definitions

```typescript
// ── Function Types ───────────────────────────────────────────────────

/**
 * A function that embeds an array of texts and returns an array of vectors.
 * The returned array must have the same length and order as the input array.
 */
type EmbedFn = (texts: string[]) => Promise<number[][]>;

/**
 * A function that sends a prompt to an LLM and returns the output string.
 */
type LlmFn = (prompt: string) => Promise<string>;

// ── Monitor Options ──────────────────────────────────────────────────

interface MonitorOptions {
  /**
   * Identifier for the prompt being monitored.
   * Used for organizational purposes and baseline management.
   * Required.
   */
  promptId: string;

  /**
   * Embedding function for converting output strings to vectors.
   * Required.
   */
  embedFn: EmbedFn;

  /**
   * The embedding model identifier. Stored in every snapshot for compatibility checking.
   * Required.
   */
  embeddingModel: string;

  /**
   * The drift severity level at which the onDrift callback is invoked.
   * Default: 'high'.
   */
  alertSeverity?: DriftSeverity;

  /**
   * Per-metric score thresholds that trigger an alert independently of severity.
   * An alert fires if severity >= alertSeverity OR any per-metric score exceeds its threshold.
   * Default: {} (no per-metric overrides; severity alone controls alerts).
   */
  thresholds?: Partial<MetricThresholds>;

  /**
   * Callback invoked when an alert fires.
   * Default: undefined (no callback).
   */
  onDrift?: (report: DriftReport) => void | Promise<void>;

  /**
   * Metric weights for composite drift score computation.
   * Weights are normalized to sum to 1 after excluding disabled metrics.
   * Default: { centroid: 0.25, jsd: 0.25, psi: 0.15, vocabulary: 0.15, spread: 0.10, outputLength: 0.10 }
   */
  metricWeights?: Partial<MetricWeights>;

  /**
   * Which drift metrics to run in compare().
   * Disabling metrics reduces computation.
   * Default: all metrics enabled.
   */
  enabledMetrics?: {
    centroid?: boolean;       // default: true
    spread?: boolean;         // default: true
    jsd?: boolean;            // default: true
    psi?: boolean;            // default: true
    outputLength?: boolean;   // default: true
    vocabulary?: boolean;     // default: true
  };

  /**
   * Cooldown period in milliseconds between consecutive alert firings.
   * After an alert fires, subsequent alerts are suppressed for this duration.
   * Default: 0 (no cooldown).
   */
  alertCooldownMs?: number;

  /**
   * Number of bins for JSD and PSI histogram computation.
   * Default: 20.
   */
  histogramBins?: number;

  /**
   * Number of random pairs to sample for pairwise similarity estimation.
   * Default: 500.
   */
  pairwiseSamplePairs?: number;
}

// ── Thresholds ───────────────────────────────────────────────────────

interface MetricThresholds {
  composite: number;
  centroid: number;
  spread: number;
  jsd: number;
  psi: number;
  outputLength: number;
  vocabulary: number;
}

interface MetricWeights {
  centroid: number;
  spread: number;
  jsd: number;
  psi: number;
  outputLength: number;
  vocabulary: number;
}

// ── Drift Severity ───────────────────────────────────────────────────

type DriftSeverity = 'none' | 'low' | 'medium' | 'high' | 'critical';

// ── Metric Result ────────────────────────────────────────────────────

interface MetricResult {
  /** Drift score for this metric, in [0, 1]. */
  score: number;

  /** Whether this metric was computed. False when metric is disabled. */
  computed: boolean;

  /** Human-readable interpretation of this metric's result. */
  interpretation: string;

  /** Metric-specific details. */
  details?: Record<string, unknown>;
}

// ── Drift Report ─────────────────────────────────────────────────────

interface DriftReport {
  /** Unique identifier for this report, UUID v4. */
  id: string;

  /** ISO 8601 timestamp of when this report was generated. */
  createdAt: string;

  /** The prompt ID being monitored. */
  promptId: string;

  /** The two snapshot IDs being compared. */
  snapshotIds: [string, string];

  /** Per-metric drift results. */
  metrics: {
    centroid: MetricResult;
    spread: MetricResult;
    jsd: MetricResult;
    psi: MetricResult;
    outputLength: MetricResult;
    vocabulary: MetricResult;
  };

  /** Composite drift score in [0, 1]. Weighted average of per-metric scores. */
  driftScore: number;

  /** Severity classification. */
  severity: DriftSeverity;

  /** Whether this report triggered an alert. */
  alert: boolean;

  /** Human-readable explanation of the drift. */
  explanation: string;

  /** Comparison metadata. */
  comparison: {
    baselineCreatedAt: string;
    currentCreatedAt: string;
    baselineSampleCount: number;
    currentSampleCount: number;
  };
}

// ── Probe Result ─────────────────────────────────────────────────────

interface ProbeResult {
  /** The raw output strings collected from the LLM. */
  outputs: string[];

  /** The snapshot computed from the outputs. */
  snapshot: Snapshot;

  /** The drift report, if a baseline was available for comparison. Null otherwise. */
  report: DriftReport | null;

  /** Timing information. */
  timing: {
    totalMs: number;
    llmCallsMs: number;
    embeddingMs: number;
    analysisMs: number;
  };
}

// ── Drift Error ──────────────────────────────────────────────────────

interface DriftError extends Error {
  code: 'INSUFFICIENT_OUTPUTS' | 'INCONSISTENT_DIMENSIONS' | 'NO_BASELINE'
      | 'INCOMPATIBLE_SNAPSHOTS' | 'INVALID_SNAPSHOT' | 'EMBED_FAILED';
}
```

---

## 12. Alerting

### Alert Conditions

An alert fires when either of two conditions is met:

1. The overall drift severity meets or exceeds `alertSeverity` (default: `'high'`), OR
2. Any per-metric score exceeds its configured threshold in `thresholds`.

Both conditions trigger the same `onDrift` callback. The `report.alert` field records whether the report crossed an alert threshold.

### The `onDrift` Callback

```typescript
const monitor = createMonitor({
  promptId: 'classify-intent',
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
  alertSeverity: 'medium',
  onDrift: (report) => {
    sendSlackNotification({
      channel: '#llm-health',
      text: `Prompt drift alert: ${report.promptId} — score=${report.driftScore.toFixed(3)}, severity=${report.severity}`,
    });
  },
});
```

The callback may be synchronous or async. If async, `prompt-drift` does not await it -- fire-and-forget. If the callback throws, the error is swallowed (to prevent alert failures from disrupting the main pipeline). Callers that need guaranteed delivery should handle errors inside the callback.

### Alert Cooldown

When `alertCooldownMs` is configured, after an alert fires, subsequent alerts for the same monitor are suppressed for the specified duration. This prevents alert fatigue when running frequent probes that all detect the same drift.

```typescript
const monitor = createMonitor({
  // ... other options
  alertSeverity: 'medium',
  alertCooldownMs: 3600_000, // 1 hour cooldown
  onDrift: (report) => { /* ... */ },
});
```

### Severity Levels and Response

| Severity | Alert? (default `alertSeverity: 'high'`) | Recommended Response |
|---|---|---|
| `none` | No | No action. Log for trend tracking. |
| `low` | No | Monitor. Check again in the next window. |
| `medium` | No | Investigate. Review sample outputs. Compare to baseline. |
| `high` | Yes | Action recommended. Review outputs, re-establish baseline, or adjust prompt. |
| `critical` | Yes | Immediate action. Audit all recent outputs. Likely a model update. |

### Integration with Monitoring Systems

**Webhook**:
```typescript
onDrift: async (report) => {
  await fetch('https://alerts.internal/prompt-drift', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(report),
  });
},
```

**Structured logging**:
```typescript
onDrift: (report) => {
  logger.warn({ event: 'prompt_drift', ...report });
},
```

**Metrics emission**:
```typescript
onDrift: (report) => {
  metrics.gauge('prompt_drift_score', report.driftScore, { promptId: report.promptId });
  metrics.increment('prompt_drift_alerts', { severity: report.severity });
},
```

---

## 13. Baseline Management

### Setting a Baseline

A baseline is established in one of three ways:

**1. From a probe**: Run the prompt N times and use the resulting snapshot as baseline:
```typescript
const result = await monitor.probe(prompt, llmFn, { setAsBaseline: true, runs: 100 });
// result.snapshot is now the baseline
```

**2. From production outputs**: Collect outputs from the first week of production:
```typescript
const initialOutputs = await getOutputsFromLogs({ since: '7d' });
const snapshot = await monitor.snapshot(initialOutputs);
monitor.setBaseline(snapshot);
```

**3. From storage**: Load a previously saved baseline:
```typescript
const baseline = monitor.loadSnapshot('./baselines/classify-intent-baseline.json');
monitor.setBaseline(baseline);
```

### Baseline Persistence

Baselines should be persisted to survive process restarts:

```typescript
// Save
const snapshot = await monitor.snapshot(outputs);
monitor.saveSnapshot(snapshot, './baselines/classify-intent.json');

// Load on next startup
const baseline = monitor.loadSnapshot('./baselines/classify-intent.json');
monitor.setBaseline(baseline);
```

### Baseline Versioning

Baselines should be tagged with contextual information for auditability:

```typescript
const snapshot = await monitor.snapshot(outputs, {
  metadata: {
    version: '2026-03-18',
    modelVersion: 'gpt-4o-2025-11-20',
    promptVersion: 'v3.2',
    environment: 'production',
    establishedBy: 'nightly-baseline-job',
  },
});
```

When drift is detected and the new behavior is acceptable, refresh the baseline:

```typescript
if (report.severity === 'medium' && teamApproves) {
  // The new behavior is intentional or acceptable; update baseline
  monitor.setBaseline(currentSnapshot);
  monitor.saveSnapshot(currentSnapshot, './baselines/classify-intent.json');
}
```

### Baseline Staleness

A baseline becomes stale when it no longer represents the intended behavior of the prompt. This happens when:
- The prompt is intentionally updated (the old baseline reflects old prompt behavior).
- The team decides to accept the new model behavior (the old baseline is unnecessarily conservative).
- The production input distribution has changed fundamentally (the baseline reflects old input patterns).

`prompt-drift` does not automatically refresh baselines. Baseline management is an explicit human decision. The `metadata` field on snapshots provides the audit trail for when and why baselines were established or refreshed.

---

## 14. Configuration Reference

### `MonitorOptions` Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `promptId` | `string` | required | Identifier for the prompt being monitored |
| `embedFn` | `EmbedFn` | required | Function to embed output strings into vectors |
| `embeddingModel` | `string` | required | Embedding model ID recorded in snapshots |
| `alertSeverity` | `DriftSeverity` | `'high'` | Minimum severity to trigger `onDrift` callback |
| `thresholds.composite` | `number` | `undefined` | Composite score threshold for alert |
| `thresholds.centroid` | `number` | `undefined` | Per-metric threshold for centroid distance |
| `thresholds.spread` | `number` | `undefined` | Per-metric threshold for spread change |
| `thresholds.jsd` | `number` | `undefined` | Per-metric threshold for JSD |
| `thresholds.psi` | `number` | `undefined` | Per-metric threshold for PSI |
| `thresholds.outputLength` | `number` | `undefined` | Per-metric threshold for output length drift |
| `thresholds.vocabulary` | `number` | `undefined` | Per-metric threshold for vocabulary drift |
| `onDrift` | `(report) => void \| Promise<void>` | `undefined` | Callback invoked when an alert fires |
| `metricWeights.centroid` | `number` | `0.25` | Weight for centroid distance in composite |
| `metricWeights.jsd` | `number` | `0.25` | Weight for JSD in composite |
| `metricWeights.psi` | `number` | `0.15` | Weight for PSI in composite |
| `metricWeights.vocabulary` | `number` | `0.15` | Weight for vocabulary drift in composite |
| `metricWeights.spread` | `number` | `0.10` | Weight for spread change in composite |
| `metricWeights.outputLength` | `number` | `0.10` | Weight for output length drift in composite |
| `enabledMetrics.centroid` | `boolean` | `true` | Enable centroid distance metric |
| `enabledMetrics.spread` | `boolean` | `true` | Enable spread change metric |
| `enabledMetrics.jsd` | `boolean` | `true` | Enable JSD metric |
| `enabledMetrics.psi` | `boolean` | `true` | Enable PSI metric |
| `enabledMetrics.outputLength` | `boolean` | `true` | Enable output length drift metric |
| `enabledMetrics.vocabulary` | `boolean` | `true` | Enable vocabulary drift metric |
| `alertCooldownMs` | `number` | `0` | Cooldown between consecutive alerts (ms) |
| `histogramBins` | `number` | `20` | Bins for JSD and PSI histogram computation |
| `pairwiseSamplePairs` | `number` | `500` | Random pairs for pairwise similarity estimation |

### `SnapshotOptions` Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `sampleSize` | `number` | `50` | Number of embedding vectors to store as sample |
| `outputSampleSize` | `number` | `10` | Number of raw output strings to store |
| `topK` | `number` | `100` | Number of top vocabulary terms to record |
| `metadata` | `Record<string, unknown>` | `{}` | Caller-provided metadata |

### `ProbeOptions` Reference

| Option | Type | Default | Description |
|---|---|---|---|
| `runs` | `number` | `30` | Number of LLM calls to make |
| `concurrency` | `number` | `5` | Maximum concurrent LLM calls |
| `setAsBaseline` | `boolean` | `false` | Automatically set as baseline if no baseline exists |
| `snapshotOptions` | `SnapshotOptions` | `{}` | Options for the resulting snapshot |
| `metadata` | `Record<string, unknown>` | `{}` | Caller-provided metadata |

---

## 15. CLI

### Installation and Invocation

```bash
# Global install
npm install -g prompt-drift
prompt-drift baseline --outputs outputs.json --model text-embedding-3-small --prompt classify-intent

# npx (no install)
npx prompt-drift check --baseline baseline.json --outputs new-outputs.json

# Package script
# package.json: { "scripts": { "drift-check": "prompt-drift check --baseline baseline.json --outputs today.json" } }
npm run drift-check
```

### CLI Binary Name

`prompt-drift`

### Commands

#### `prompt-drift baseline`

Creates a baseline snapshot from a JSON file of output strings.

```
prompt-drift baseline [options]

Options:
  --outputs <path>      Path to JSON file: string[] (array of LLM output strings). Required.
  --output <path>       Path to write the baseline snapshot JSON. Default: stdout.
  --prompt <id>         Prompt identifier to record in the snapshot. Required.
  --model <model>       Embedding model ID. Required.
  --embed-provider <p>  Provider for embedding: openai (default) | cohere.
  --api-key <key>       API key (or set OPENAI_API_KEY / COHERE_API_KEY env var).
  --sample-size <n>     Number of sample embeddings to store. Default: 50.
  --format <fmt>        Output format: json (default) | summary.
```

**Input format**: A JSON file containing an array of strings:
```json
["The capital of France is Paris.", "Paris serves as France's capital.", ...]
```

**Human output** (`--format summary`):
```
prompt-drift baseline

  Prompt:      classify-intent
  Model:       text-embedding-3-small
  Outputs:     200
  Dimensions:  1536

  Centroid spread:  0.142
  Mean pairwise similarity: 0.823 (std=0.094)
  Output length: mean=156, median=142, p5=67, p95=312
  Vocabulary size: 847 unique terms

  Baseline saved to: ./baselines/classify-intent.json
```

#### `prompt-drift check`

Compares new outputs against a baseline and prints a drift report.

```
prompt-drift check [options]

Options:
  --baseline <path>     Path to baseline snapshot JSON. Required.
  --outputs <path>      Path to JSON file of new output strings (string[]). Required.
  --model <model>       Embedding model ID. Required.
  --embed-provider <p>  Provider for embedding: openai (default) | cohere.
  --api-key <key>       API key (or set env var).
  --output <path>       Path to write the DriftReport JSON. Default: stdout for json format.
  --format <fmt>        Output format: summary (default) | json.
  --alert-severity <s>  Severity threshold for non-zero exit code: none|low|medium|high|critical.
                         Default: high (exit code 1 when severity >= high).
```

**Human output** (`--format summary`, default):
```
prompt-drift check

  Prompt:    classify-intent
  Baseline:  classify-intent-baseline.json  (2026-03-01, 200 outputs)
  Current:   today-outputs.json             (150 outputs)

  Drift scores:
    Centroid distance:  0.042  ✓
    Spread change:      0.018  ✓
    JSD:                0.067  ✓
    PSI:                0.054  ✓
    Output length:      0.031  ✓
    Vocabulary:         0.089  ✓

  Composite score:   0.053
  Severity:          low

  Interpretation: Minor output variation within expected bounds. No action needed.
```

**Alert output** (significant drift detected):
```
prompt-drift check

  Prompt:    classify-intent
  Baseline:  classify-intent-baseline.json  (2026-03-01, 200 outputs)
  Current:   today-outputs.json             (150 outputs)

  Drift scores:
    Centroid distance:  0.312  ← ELEVATED
    Spread change:      0.087  ✓
    JSD:                0.421  ← ELEVATED
    PSI:                0.378  ← ELEVATED
    Output length:      0.234  ← ELEVATED
    Vocabulary:         0.289  ← ELEVATED

  Composite score:   0.324
  Severity:          MEDIUM

  Interpretation: Output behavior has shifted across multiple dimensions.
  The model may have been updated. Review sample outputs below.

  Sample baseline outputs:
    1. "This is a billing inquiry about a recent charge."
    2. "The customer is asking about shipping status."

  Sample current outputs:
    1. "I'd be happy to help! This appears to be a billing-related inquiry..."
    2. "Thank you for reaching out! This looks like a shipping status question..."

  Action: Review outputs. Re-establish baseline if new behavior is acceptable.

Exit code: 1
```

#### `prompt-drift probe`

Actively probes a prompt against an LLM and checks for drift.

```
prompt-drift probe [options]

Options:
  --prompt <text>       The prompt text to probe. Required (or --prompt-file).
  --prompt-file <path>  Path to a file containing the prompt text.
  --baseline <path>     Path to baseline snapshot JSON. If omitted, creates a new baseline.
  --runs <n>            Number of LLM calls. Default: 30.
  --concurrency <n>     Maximum concurrent LLM calls. Default: 5.
  --provider <p>        LLM provider: openai (default) | anthropic.
  --llm-model <m>      LLM model name. Default: gpt-4o-mini.
  --embed-provider <p>  Embedding provider: openai (default) | cohere.
  --embed-model <m>    Embedding model name. Default: text-embedding-3-small.
  --api-key <key>       API key (or set env var).
  --output <path>       Path to write the report/snapshot JSON.
  --format <fmt>        Output format: summary (default) | json.
  --alert-severity <s>  Severity threshold for non-zero exit code. Default: high.
  --save-baseline       Save the probe snapshot as a new baseline file.
```

#### `prompt-drift report`

Generates a formatted report from a previously saved DriftReport JSON.

```
prompt-drift report [options]

Options:
  --input <path>        Path to a DriftReport JSON file. Required.
  --format <fmt>        Output format: summary (default) | json | markdown.
```

### Exit Codes

| Code | Meaning |
|---|---|
| `0` | No significant drift detected. Composite severity is below `--alert-severity`. |
| `1` | Drift detected at or above `--alert-severity`. Investigation or action recommended. |
| `2` | Configuration or usage error (missing required options, invalid flags, unreadable files). |

```bash
prompt-drift check --baseline baseline.json --outputs today.json --model text-embedding-3-small
if [ $? -eq 1 ]; then
  echo "Drift detected! Notifying team..."
  ./scripts/notify-drift.sh
fi
```

---

## 16. Integration

### Integration with `embed-cache`

`embed-cache` stores embedding vectors in a content-addressable cache. When `prompt-drift` embeds output strings for snapshot creation, routing through `embed-cache` avoids redundant embedding API calls for outputs that have been seen before (common in replay workflows).

```typescript
import { createCache } from 'embed-cache';
import { createMonitor } from 'prompt-drift';

const cache = createCache({ model: 'text-embedding-3-small', embedder: openaiEmbedder });
const monitor = createMonitor({
  promptId: 'classify-intent',
  embedFn: (texts) => cache.embedBatch(texts),
  embeddingModel: 'text-embedding-3-small',
});
```

### Integration with `embed-drift`

`embed-drift` monitors embedding model changes. `prompt-drift` monitors LLM output behavior changes. Together they provide complete coverage:

- **`embed-drift`** catches: embedding model updates that silently degrade retrieval quality.
- **`prompt-drift`** catches: LLM model updates that silently change output behavior.

In a production pipeline with both a retrieval layer (embeddings) and a generation layer (LLM outputs), both monitors should run. They detect different failure modes and require different remediation (re-embedding vs. prompt adjustment).

```typescript
import { createMonitor as createEmbedMonitor } from 'embed-drift';
import { createMonitor as createPromptMonitor } from 'prompt-drift';

// Monitor the embedding model
const embedMonitor = createEmbedMonitor({ modelId: 'text-embedding-3-small' });

// Monitor the LLM output behavior
const promptMonitor = createPromptMonitor({
  promptId: 'rag-answer',
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
});
```

### Integration with `llm-regression`

When `prompt-drift` detects drift, the next step is often to understand what changed. `llm-regression` provides the comparison machinery: take baseline outputs and current outputs, compare them side by side with similarity metrics, and produce a detailed regression report.

```typescript
import { createMonitor } from 'prompt-drift';
import { compareBatch } from 'llm-regression';

const monitor = createMonitor({
  promptId: 'classify-intent',
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
  onDrift: async (report) => {
    // Drift detected -- run detailed regression analysis
    const baseline = monitor.getBaseline();
    const regressionReport = await compareBatch(
      baseline.sampleOutputs.map((b, i) => ({
        id: `sample-${i}`,
        baseline: b,
        candidate: report.comparison.currentSampleOutputs?.[i] ?? '',
      })),
      { metrics: ['semantic', 'jaccard'] },
    );
    console.log('Regression analysis:', regressionReport.summary);
  },
});
```

### Integration with `prompt-version`

`prompt-version` tracks prompt template versions. When establishing a baseline, tag it with the prompt version:

```typescript
import { getVersion } from 'prompt-version';

const version = getVersion('classify-intent');
const snapshot = await monitor.snapshot(outputs, {
  metadata: { promptVersion: version.id, promptHash: version.hash },
});
```

When drift is detected, the prompt version metadata helps determine whether the drift is caused by a prompt change (check if the prompt version has changed) or a model change (prompt version unchanged but behavior changed).

### Integration with `prompt-flags`

`prompt-flags` enables feature-flagged prompt variants. Monitor each variant independently:

```typescript
import { getFlag } from 'prompt-flags';

const variant = getFlag('classify-intent-v2');
const monitor = createMonitor({
  promptId: `classify-intent-${variant}`,
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
});
```

---

## 17. Testing Strategy

### Unit Tests

Unit tests cover each drift metric and utility in isolation, using small deterministic inputs.

**Snapshot creation tests:**
- `monitor.snapshot(outputs)` calls `embedFn` with the full output array and returns a snapshot with correct prompt ID, dimensionality, and sample counts.
- Centroid is the arithmetic mean of the embedding vectors (verified numerically for a 3-output, 4-dimension input).
- Per-dimension variance is computed correctly (verified against manual calculation).
- Spread is the mean cosine distance from each embedding to the centroid (verified for a small input set).
- Output length statistics are correct: for outputs of length [100, 200, 300], mean=200, median=200, stddev=81.65, p5=110, p95=290.
- Vocabulary statistics correctly tokenize, lowercase, and count term frequencies.
- `sampleEmbeddings` has length `min(sampleSize, n)`.
- `sampleOutputs` has length `min(outputSampleSize, n)`.
- `snapshot` throws `DriftError` with code `INSUFFICIENT_OUTPUTS` for fewer than 20 outputs.
- `snapshot` throws `DriftError` with code `INCONSISTENT_DIMENSIONS` when embeddings have different lengths.
- `snapshot` throws `DriftError` with code `EMBED_FAILED` when `embedFn` throws.

**Centroid distance tests:**
- Two identical snapshots produce centroid drift score = 0.
- Snapshots with orthogonal centroids produce score = 1.0.
- Snapshots with slightly shifted centroids produce a proportional score.

**Spread change tests:**
- Two snapshots with identical spread produce score = 0.
- Snapshot with spread 0.1 vs. 0.2 (ratio 2.0) produces elevated score.
- Snapshot with spread 0.1 vs. 0.1 produces score = 0.

**JSD tests:**
- Two identical distributions produce JSD score = 0.
- Two non-overlapping distributions produce JSD score = 1.0.
- A slightly shifted distribution produces a proportional score between 0 and 1.
- Verify bin boundaries and smoothing do not produce NaN or Infinity.

**PSI tests:**
- Two identical distributions produce PSI score = 0.
- Two significantly different distributions produce PSI score > 0.3.
- Verify PSI is always non-negative.
- Verify smoothing prevents log(0).

**Output length drift tests:**
- Identical length distributions produce score = 0.
- Distributions with Cohen's d = 1.0 produce score approximately 0.35.
- Distributions with very different variances but similar means produce elevated score from the variance ratio component.

**Vocabulary drift tests:**
- Identical term frequency distributions produce score = 0.
- Completely different vocabularies produce score = 1.0.
- Partially overlapping vocabularies produce proportional score.
- New term ratio is correctly computed from top-K terms.

**Composite score tests:**
- Verify weighted average formula with default weights against a manually computed expected value.
- Verify that disabling a metric redistributes its weight proportionally.
- Verify severity classification: score 0.03 maps to `none`, 0.10 to `low`, 0.25 to `medium`, 0.45 to `high`, 0.70 to `critical`.

**Alert tests:**
- `report.alert` is `false` when `severity < alertSeverity`.
- `report.alert` is `true` when `severity >= alertSeverity`.
- `report.alert` is `true` when a per-metric score exceeds its threshold even if severity is below `alertSeverity`.
- `onDrift` callback is called when alert fires and not called when alert does not fire.
- `onDrift` async errors are swallowed without crashing.
- Alert cooldown suppresses repeated alerts within the cooldown window.

**Serialization tests:**
- `saveSnapshot` writes valid JSON parseable back to a `Snapshot`.
- `loadSnapshot` reads back the same snapshot (deep equality).
- `loadSnapshot` throws `DriftError('INVALID_SNAPSHOT')` for malformed JSON.
- `loadSnapshot` throws `DriftError('INVALID_SNAPSHOT')` for JSON missing required fields.

### Integration Tests

- **End-to-end drift detection**: Generate two sets of outputs from a deterministic mock LLM (one set with consistent responses, one with systematically different responses). Verify `compare()` produces a composite score that correctly reflects the behavioral difference.
- **No-drift baseline**: Generate two samples from the same mock LLM. Verify composite score < 0.10 and severity is `none` or `low`.
- **Full behavioral shift simulation**: Generate outputs from two mock LLMs with different behaviors (different vocabulary, different length, different semantic content). Verify composite score > 0.40 and severity is `high` or `critical`.
- **Probe round-trip**: Call `monitor.probe(prompt, mockLlm, { setAsBaseline: true })`. Then call `probe` again with the same mock LLM. Verify drift score < 0.10. Then call with a different mock LLM. Verify drift score > 0.30.
- **CLI integration**: Invoke the CLI via `child_process.execSync` in tests. Verify exit codes for no-drift and high-drift scenarios.

### Test Data

Tests use three categories of data:

1. **Deterministic synthetic embeddings**: Small (4-dimensional) embedding vectors with analytically known properties. Used for testing individual metric computations where the correct answer can be verified by hand.

2. **Mock LLM functions**: Deterministic functions that return predefined outputs, simulating stable and drifted LLM behavior. The "stable" mock returns outputs from a fixed pool. The "drifted" mock returns outputs from a different pool with systematically different vocabulary, length, and semantic content.

3. **Mock embed functions**: Deterministic functions that return predefined embedding vectors for known input strings. Used to isolate snapshot creation and comparison logic from real embedding API behavior.

---

## 18. Performance

### Snapshot Creation

Snapshot creation is dominated by the embedding API call. The statistical computation is fast:

| Component | Complexity | Time (n=100 outputs, d=1536) |
|---|---|---|
| Embed outputs (API call) | Network-dependent | 500-2000ms |
| Centroid computation | O(n * d) | ~1ms |
| Per-dimension variance | O(n * d) | ~1ms |
| Spread computation | O(n * d) | ~1ms |
| Pairwise similarity sampling | O(M * d), M=500 | ~2ms |
| Output length statistics | O(n) | < 1ms |
| Vocabulary statistics | O(n * L), L=avg output length | ~5ms |
| Reservoir sampling | O(n) | < 1ms |
| **Total (excluding embedding)** | | **~10ms** |
| **Total (including embedding)** | | **500-2000ms** |

The embedding API call dominates. For 100 outputs, most providers batch-embed in a single API call (50-200ms for OpenAI). For 1000 outputs, the API call may take 1-5 seconds.

### Snapshot Comparison

Comparison operates on pre-computed snapshot statistics. No embedding API calls are needed:

| Component | Complexity | Time (sampleSize=50, d=1536) |
|---|---|---|
| Centroid distance | O(d) | < 1ms |
| Spread change | O(1) | < 1ms |
| JSD (binning + computation) | O(s * d + K) | ~2ms |
| PSI (binning + computation) | O(s * d + K) | ~2ms |
| Output length drift | O(1) | < 1ms |
| Vocabulary drift | O(V), V=union vocab size | < 1ms |
| Composite + severity | O(1) | < 1ms |
| **Total** | | **~5-10ms** |

Snapshot comparison is designed to be fast: under 20ms for default settings. It does not make any network calls.

### Active Probing

A probe involves N LLM calls plus one embedding call plus snapshot creation plus comparison:

| Component | Time (N=30, concurrency=5) |
|---|---|
| LLM calls (30 calls, 5 concurrent) | 5-30s (API latency dependent) |
| Embedding (30 outputs, 1 batch call) | 200-500ms |
| Snapshot creation (statistical computation) | ~10ms |
| Comparison (if baseline exists) | ~10ms |
| **Total** | **5-30s** (LLM latency dominates) |

### Memory Footprint

| Component | Memory |
|---|---|
| Centroid (1536 dims) | ~12 KB |
| Variance (1536 dims) | ~12 KB |
| Sample embeddings (50 * 1536 dims) | ~600 KB |
| Vocabulary stats | ~5-20 KB |
| Sample outputs (~10 outputs, ~150 chars each) | ~2 KB |
| **Per snapshot total** | **~650 KB** |

Two snapshots in memory for comparison: ~1.3 MB. The time-series buffer for rolling window mode stores only the outputs and their timestamps (not embedding vectors), which is approximately 10-50 KB per day of outputs at typical collection rates.

---

## 19. Dependencies

### Runtime Dependencies

**Zero mandatory runtime dependencies.** All drift metrics (centroid distance, spread change, JSD, PSI, output length drift, vocabulary drift) are implemented in pure TypeScript. No native modules, no WASM.

Node.js built-ins used:

- `node:crypto` -- UUID v4 generation for snapshot and report IDs.
- `node:fs` -- filesystem snapshot serialization in `saveSnapshot` / `loadSnapshot`.
- `node:path` -- path utilities in CLI.

External dependencies required by the caller (not bundled):
- An embedding function (`embedFn`) -- the caller provides this. Typically backed by OpenAI, Cohere, or a local model.
- An LLM function (`llmFn`) -- the caller provides this for active probing. Typically backed by OpenAI, Anthropic, or a local model.

### Dev Dependencies

| Package | Purpose |
|---|---|
| `typescript` | TypeScript compiler |
| `vitest` | Test runner |
| `eslint` | Linting |
| `@types/node` | Node.js type definitions |

---

## 20. File Structure

```
prompt-drift/
├── src/
│   ├── index.ts                      # Public API exports: createMonitor, types, errors
│   ├── monitor.ts                    # DriftMonitor class: snapshot, compare, probe, check
│   ├── snapshot.ts                   # Snapshot creation: centroid, variance, spread, stats
│   ├── metrics/
│   │   ├── centroid.ts               # Centroid distance: cosine distance between centroids
│   │   ├── spread.ts                 # Spread change: ratio of spread values
│   │   ├── jsd.ts                    # Jensen-Shannon divergence: binned distribution comparison
│   │   ├── psi.ts                    # Population Stability Index: standard MLOps drift metric
│   │   ├── output-length.ts          # Output length drift: Cohen's d + variance ratio
│   │   └── vocabulary.ts             # Vocabulary drift: term frequency cosine distance + new terms
│   ├── composite.ts                  # Composite score computation and severity classification
│   ├── alert.ts                      # Alert threshold evaluation, cooldown, onDrift dispatch
│   ├── probe.ts                      # Active probing: run prompt N times with concurrency control
│   ├── tokenizer.ts                  # Simple whitespace/punctuation tokenizer for vocabulary stats
│   ├── serialization.ts              # saveSnapshot, loadSnapshot, schema validation
│   ├── math.ts                       # Shared math: cosine similarity, dot product, L2 norm, stats
│   ├── cli.ts                        # CLI entry point (prompt-drift command)
│   ├── types.ts                      # All TypeScript type definitions
│   └── errors.ts                     # DriftError class with typed error codes
├── src/__tests__/
│   ├── snapshot.test.ts              # Snapshot creation unit tests
│   ├── monitor.test.ts               # DriftMonitor integration: compare, check, probe, alert
│   ├── metrics/
│   │   ├── centroid.test.ts          # Centroid distance unit tests
│   │   ├── spread.test.ts            # Spread change unit tests
│   │   ├── jsd.test.ts               # JSD unit tests
│   │   ├── psi.test.ts               # PSI unit tests
│   │   ├── output-length.test.ts     # Output length drift unit tests
│   │   └── vocabulary.test.ts        # Vocabulary drift unit tests
│   ├── composite.test.ts             # Composite score and severity classification tests
│   ├── alert.test.ts                 # Alert threshold, cooldown, and callback dispatch tests
│   ├── probe.test.ts                 # Active probing unit tests
│   ├── serialization.test.ts         # saveSnapshot, loadSnapshot, validation tests
│   ├── math.test.ts                  # Math utility unit tests
│   └── integration/
│       ├── drift-detection.test.ts   # End-to-end: no-drift baseline, behavioral shift simulation
│       └── cli.test.ts               # CLI invocation via child_process, exit code verification
├── src/__fixtures__/
│   ├── stable-outputs.json           # Pre-computed outputs from a stable mock LLM
│   ├── drifted-outputs.json          # Pre-computed outputs from a drifted mock LLM
│   └── mock-embeddings.json          # Pre-computed embeddings for fixture outputs
├── package.json
├── tsconfig.json
├── README.md
└── SPEC.md
```

---

## 21. Implementation Roadmap

### Phase 1: Core Statistical Engine (Week 1)

The foundation of `prompt-drift` is the statistical computation and snapshot layer. This phase implements:

1. **`math.ts`**: Shared math utilities -- cosine similarity, cosine distance, L2 norm, dot product, element-wise mean, element-wise variance, percentile computation. All operations use standard JavaScript arrays (the dimensionality of output embeddings is fixed per snapshot, typically 1536). Unit-tested against hand-computed expected values.

2. **`tokenizer.ts`**: Simple whitespace and punctuation tokenizer for vocabulary statistics. Split on whitespace, lowercase, strip punctuation. No external NLP library. Unit-tested with known inputs.

3. **`snapshot.ts`**: Snapshot creation -- embed outputs using the provided `embedFn`, compute centroid, per-dimension variance, spread, pairwise similarity statistics, output length statistics, vocabulary statistics, reservoir sampling for sample embeddings and sample outputs. Unit-tested with small synthetic inputs and mock embed functions.

4. **`metrics/centroid.ts`**: Centroid distance computation -- cosine distance between two centroid vectors, normalized.

5. **`metrics/spread.ts`**: Spread change computation -- ratio of spread values, normalized.

6. **`metrics/jsd.ts`**: Jensen-Shannon divergence -- bin sample embeddings by cosine distance to baseline centroid, compute discrete distributions, compute JSD. Verify with known distribution pairs.

7. **`metrics/psi.ts`**: Population Stability Index -- same binning as JSD, compute PSI. Verify with known distribution pairs.

8. **`metrics/output-length.ts`**: Output length drift -- Cohen's d on output lengths, variance ratio, combined score.

9. **`metrics/vocabulary.ts`**: Vocabulary drift -- cosine distance between term frequency vectors, new term ratio, combined score.

### Phase 2: Monitor and Composite (Week 2)

1. **`composite.ts`**: Weighted composite score from per-metric scores. Weight normalization when metrics are disabled. Severity classification. Explanation generation.

2. **`alert.ts`**: Threshold evaluation against per-metric thresholds and `alertSeverity`. `onDrift` callback dispatch. Cooldown logic. Error swallowing for async callbacks.

3. **`monitor.ts`**: `createMonitor(options)`, `snapshot()`, `compare()`, `setBaseline()`, `getBaseline()`, `check()`. Wire together all metric modules, composite computation, and alert dispatch.

4. **`probe.ts`**: Active probing -- run prompt N times with concurrency control (using a simple semaphore pattern with `Promise.all` and chunking). Collect outputs, create snapshot, compare to baseline.

5. **`errors.ts`**: `DriftError` class with typed error codes.

6. **`types.ts`**: All TypeScript type definitions.

7. **`index.ts`**: Public API exports.

### Phase 3: Serialization and CLI (Week 3)

1. **`serialization.ts`**: `saveSnapshot`, `loadSnapshot`, JSON schema validation. Verify round-trip fidelity.

2. **`cli.ts`**: Four commands -- `baseline`, `check`, `probe`, `report`. Argument parsing using `process.argv` directly (no external parser dependency). Human-readable and JSON output formats. Exit code logic. Built-in OpenAI embedding support for CLI convenience (the API requires a key but no additional npm dependency -- uses native `fetch`).

### Phase 4: Tests and Documentation (Week 3-4)

1. Full unit test suite for all metrics and components.
2. Integration tests with pre-computed fixture data.
3. CLI tests via `child_process.execSync`.
4. README with quick-start, API reference, and common use-case examples.
5. Performance validation: measure snapshot creation and comparison times against the targets in Section 18.

---

## 22. Example Use Cases

### Use Case 1: Nightly Production Monitoring

**Scenario**: A team operates a classification prompt that categorizes customer support tickets into 15 categories. The prompt was carefully tuned and deployed 3 months ago. The team suspects that OpenAI may have updated GPT-4o since then, but there is no way to confirm without monitoring.

**Solution**: A nightly cron job collects the day's classification outputs from production logs (typically 200-500 outputs), creates a snapshot, and compares to the baseline established at deployment. The drift report is logged as structured JSON to the observability stack. A Datadog alert fires when the composite severity reaches `medium` or above.

**Configuration**:
```typescript
const monitor = createMonitor({
  promptId: 'ticket-classifier',
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
  alertSeverity: 'medium',
  onDrift: (report) => {
    datadogLogger.warn({ event: 'prompt_drift', ...report });
  },
});

// Nightly job
const todaysOutputs = await db.query("SELECT output FROM classifications WHERE created_at > NOW() - INTERVAL '24h'");
const report = await monitor.check(todaysOutputs.map(r => r.output));
```

### Use Case 2: Model Update Detection

**Scenario**: A content generation pipeline uses Claude to produce product descriptions. Anthropic updates Claude periodically, and each update subtly changes the writing style -- slightly different sentence structure, different word choices, different paragraph length. The marketing team needs consistent brand voice.

**Solution**: A weekly probe runs the production prompt 50 times and compares to the baseline. If vocabulary drift or output length drift spikes while centroid distance remains low, it indicates a stylistic change (same topic, different expression). The team reviews the sample outputs in the drift report and decides whether to adjust the prompt or accept the new style.

**Configuration**: `alertSeverity: 'low'`, vocabulary drift threshold: 0.15, output length threshold: 0.20.

### Use Case 3: CI/CD Pre-Deployment Gate

**Scenario**: A team deploys their application weekly. The deployment includes infrastructure changes, SDK updates, and configuration changes that could indirectly affect LLM behavior (different API endpoints, different default parameters, updated SDK that changes how the prompt is formatted). They want to catch behavioral changes before they reach production.

**Solution**: A CI step in the deployment pipeline probes critical prompts against the staging LLM endpoint. Each probe runs the prompt 30 times, creates a snapshot, and compares to the baseline (stored in the repository alongside the prompt). If drift exceeds `high` severity, the pipeline fails and posts a Slack notification with the drift report.

**CLI**:
```bash
prompt-drift probe \
  --prompt-file ./prompts/classify-intent.txt \
  --baseline ./baselines/classify-intent.json \
  --runs 30 \
  --provider openai \
  --llm-model gpt-4o \
  --embed-model text-embedding-3-small \
  --alert-severity high \
  --format summary
```

### Use Case 4: SLA Monitoring Dashboard

**Scenario**: An organization provides an AI-powered API to enterprise customers with SLAs on response format consistency (JSON schema compliance), response length bounds, and semantic relevance. A silent model change that alters any of these dimensions is an SLA breach.

**Solution**: Continuous monitoring with hourly probes. Each hour, `monitor.probe()` runs the prompt 20 times. The composite drift score, per-metric scores, and severity are emitted as Prometheus metrics and visualized in a Grafana dashboard. The dashboard shows drift trends over days and weeks, enabling the operations team to detect gradual drift (not just sudden shifts) and correlate drift events with known model update schedules.

**Integration**:
```typescript
// Every hour
const result = await monitor.probe(prompt, llmFn, { runs: 20 });
prometheus.gauge('prompt_drift_composite', result.report?.driftScore ?? 0, { prompt: 'api-response' });
prometheus.gauge('prompt_drift_length', result.report?.metrics.outputLength.score ?? 0, { prompt: 'api-response' });
prometheus.gauge('prompt_drift_vocabulary', result.report?.metrics.vocabulary.score ?? 0, { prompt: 'api-response' });
```

### Use Case 5: Multi-Prompt Fleet Monitoring

**Scenario**: A platform team maintains 40+ prompts across different products. Each prompt needs independent drift monitoring with prompt-specific thresholds (the classification prompt is highly sensitive to drift; the creative writing prompt tolerates more variation).

**Solution**: One `DriftMonitor` per prompt, each with its own baseline and thresholds. A centralized monitoring service iterates over all monitors, runs checks, aggregates reports, and produces a fleet-level dashboard.

```typescript
const monitors = prompts.map(p => createMonitor({
  promptId: p.id,
  embedFn: myEmbedFn,
  embeddingModel: 'text-embedding-3-small',
  alertSeverity: p.sensitivity === 'high' ? 'low' : 'high',
  onDrift: (report) => fleetAlertHandler(report),
}));

// Daily check per prompt
for (const [i, monitor] of monitors.entries()) {
  const outputs = await getOutputs(prompts[i].id);
  await monitor.check(outputs);
}
```
