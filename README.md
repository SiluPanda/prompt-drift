# prompt-drift

Detect silent LLM output changes over time via semantic drift analysis.

[![npm version](https://img.shields.io/npm/v/prompt-drift.svg)](https://www.npmjs.com/package/prompt-drift)
[![npm downloads](https://img.shields.io/npm/dt/prompt-drift.svg)](https://www.npmjs.com/package/prompt-drift)
[![license](https://img.shields.io/npm/l/prompt-drift.svg)](https://github.com/SiluPanda/prompt-drift/blob/master/LICENSE)
[![node](https://img.shields.io/node/v/prompt-drift.svg)](https://nodejs.org)

---

## Description

LLM providers routinely update their models without explicit version changes visible to the developer. Same API endpoint, same model name, same prompt -- different behavior. `prompt-drift` monitors the statistical distribution of LLM outputs for a given prompt and alerts when that distribution shifts beyond configurable thresholds.

You provide an embedding function. `prompt-drift` handles the rest: snapshot creation, six drift metrics, composite scoring, severity classification, and alerting. Zero mandatory runtime dependencies. All statistical computations are self-contained TypeScript.

The package supports two monitoring modes:

- **Active probing** -- call `monitor.probe()` to run a prompt N times against the LLM, build a snapshot, and compare against a stored baseline.
- **Passive collection** -- gather outputs from production logs and pass them to `monitor.check()` to detect drift without additional LLM calls.

---

## Installation

```bash
npm install prompt-drift
```

Requires Node.js 18 or later.

---

## Quick Start

```typescript
import { createMonitor } from 'prompt-drift';

// Provide any embedding function -- OpenAI, Cohere, local model, etc.
const embedFn = async (text: string): Promise<number[]> => {
  const response = await openai.embeddings.create({
    model: 'text-embedding-3-small',
    input: text,
  });
  return response.data[0].embedding;
};

const monitor = createMonitor({
  embedFn,
  onDrift: (report) => {
    console.warn('Drift detected!', report.severity, report.explanation);
  },
});

// 1. Capture a baseline from known-good outputs
const baselineOutputs = [
  'The capital of France is Paris.',
  'Paris is the capital of France.',
  'France has Paris as its capital city.',
  // ... collect 20+ outputs for statistical power
];
const baseline = await monitor.snapshot(baselineOutputs, { promptId: 'capital-qa' });
monitor.setBaseline(baseline);

// 2. Later -- check new outputs for drift
const newOutputs = await collectOutputsFromProduction();
const report = await monitor.check(newOutputs);

console.log(report.severity);            // 'none' | 'low' | 'medium' | 'high' | 'critical'
console.log(report.compositeDriftScore); // 0..1
console.log(report.explanation);         // human-readable summary
```

---

## Features

- **Six drift metrics** -- centroid distance, spread change, Jensen-Shannon divergence, Population Stability Index, output length drift, and vocabulary drift, combined into a single composite score.
- **Severity classification** -- composite scores are mapped to `none`, `low`, `medium`, `high`, or `critical` severity levels with configurable thresholds.
- **Alert callbacks** -- register an `onDrift` handler that fires when drift is detected, with a configurable cooldown to prevent alert storms.
- **Active probing** -- `monitor.probe()` calls your LLM function N times, builds a snapshot, and compares against the baseline in one step.
- **Passive collection** -- `monitor.check()` accepts pre-collected outputs from production logs for zero-cost drift detection.
- **Serialization** -- `monitor.serialize()` exports the monitor state as JSON for persistence and restoration across processes.
- **Embedding-agnostic** -- bring any embedding function. OpenAI, Cohere, local models, or custom implementations all work.
- **Zero runtime dependencies** -- all math (cosine similarity, KL divergence, Jensen-Shannon divergence, histograms, percentiles) is implemented in pure TypeScript.
- **Full TypeScript support** -- ships with declaration files and source maps. All types are exported.

---

## API Reference

### `createMonitor(options: MonitorOptions): DriftMonitor`

Factory function that creates and returns a `DriftMonitor` instance.

```typescript
import { createMonitor } from 'prompt-drift';

const monitor = createMonitor({
  embedFn: myEmbedFunction,
  thresholds: { centroidDistance: 0.15 },
  weights: { centroidDistance: 0.3 },
  onDrift: (report) => alert(report),
  alertCooldownMs: 120000,
});
```

#### `MonitorOptions`

| Property | Type | Required | Default | Description |
|---|---|---|---|---|
| `embedFn` | `EmbedFn` | Yes | -- | Function that maps a string to an embedding vector. Signature: `(text: string) => Promise<number[]>`. |
| `thresholds` | `object` | No | See below | Per-metric thresholds. When a metric score exceeds its threshold, it is flagged as `exceeded` in the report. |
| `weights` | `object` | No | See below | Per-metric weights for the composite drift score. |
| `onDrift` | `(report: DriftReport) => void` | No | -- | Callback invoked when drift severity is not `none`. Subject to cooldown. |
| `alertCooldownMs` | `number` | No | `60000` | Minimum milliseconds between consecutive `onDrift` invocations. |

**Default thresholds:**

| Metric | Default |
|---|---|
| `centroidDistance` | `0.1` |
| `spreadChange` | `0.2` |
| `jsd` | `0.1` |
| `psi` | `0.2` |
| `lengthDrift` | `0.3` |
| `vocabularyDrift` | `0.3` |

**Default weights:**

| Metric | Default |
|---|---|
| `centroidDistance` | `0.2` |
| `spreadChange` | `0.15` |
| `jsd` | `0.2` |
| `psi` | `0.2` |
| `lengthDrift` | `0.15` |
| `vocabularyDrift` | `0.1` |

---

### `monitor.snapshot(outputs: string[], options?): Promise<Snapshot>`

Embeds an array of LLM output strings and computes a statistical snapshot of their distribution.

```typescript
const snap = await monitor.snapshot(outputs, { promptId: 'summarize-v2' });
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `outputs` | `string[]` | Array of LLM output strings to analyze. |
| `options.promptId` | `string` (optional) | Identifier for the prompt being monitored. Stored in the snapshot for organizational purposes. |

**Returns:** `Promise<Snapshot>` -- a statistical summary containing centroid, spread, pairwise similarity, length stats, vocabulary stats, and the original samples.

---

### `monitor.compare(baseline: Snapshot, current: Snapshot): DriftReport`

Compares two snapshots and returns a drift report. This is a synchronous operation -- all computation happens on the pre-computed snapshot statistics.

```typescript
const report = monitor.compare(baselineSnap, currentSnap);
console.log(report.severity);
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `baseline` | `Snapshot` | The reference snapshot (known-good distribution). |
| `current` | `Snapshot` | The snapshot to compare against the baseline. |

**Returns:** `DriftReport` -- per-metric results, composite score, severity, and explanation.

---

### `monitor.check(newOutputs: string[]): Promise<DriftReport>`

Builds a snapshot from the provided outputs and compares it against the stored baseline. This is the primary entry point for passive monitoring with production-collected outputs.

```typescript
monitor.setBaseline(baseline);
const report = await monitor.check(newOutputs);
```

**Throws:** `Error` with message `'No baseline set. Call setBaseline() first.'` if no baseline has been set.

**Returns:** `Promise<DriftReport>`

---

### `monitor.probe(prompt: string, llmFn: LlmFn, options?: ProbeOptions): Promise<ProbeResult>`

Actively probes a prompt by calling `llmFn(prompt)` multiple times, building a snapshot from the collected outputs, and comparing against the stored baseline if one exists.

```typescript
const llmFn = async (prompt: string) => {
  const response = await openai.chat.completions.create({
    model: 'gpt-4',
    messages: [{ role: 'user', content: prompt }],
  });
  return response.choices[0].message.content;
};

const { snapshot, report } = await monitor.probe(
  'Summarize the benefits of exercise.',
  llmFn,
  { sampleCount: 10 }
);

// report is null if no baseline has been set
if (report) {
  console.log(report.severity);
}
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `string` | The prompt text to send to the LLM. |
| `llmFn` | `LlmFn` | Function that calls the LLM. Signature: `(prompt: string) => Promise<string>`. |
| `options.sampleCount` | `number` (optional) | Number of times to call `llmFn`. Default: `5`. |
| `options.timeoutMs` | `number` (optional) | Timeout for the entire probe operation in milliseconds. |

**Returns:** `Promise<ProbeResult>` -- contains `snapshot` (always present) and `report` (`DriftReport | null`, null when no baseline is set).

---

### `monitor.setBaseline(snapshot: Snapshot): void`

Sets the reference snapshot for future `check()` and `probe()` comparisons.

```typescript
const snap = await monitor.snapshot(knownGoodOutputs);
monitor.setBaseline(snap);
```

---

### `monitor.getBaseline(): Snapshot | null`

Returns the current baseline snapshot, or `null` if none has been set.

```typescript
const baseline = monitor.getBaseline();
if (!baseline) {
  console.log('No baseline established yet.');
}
```

---

### `monitor.hasBaseline: boolean`

Read-only property. Returns `true` if a baseline snapshot has been set.

```typescript
if (monitor.hasBaseline) {
  const report = await monitor.check(newOutputs);
}
```

---

### `monitor.serialize(): string`

Serializes the monitor state (including the stored baseline) to a JSON string. Use this to persist monitor state across process restarts.

```typescript
const state = monitor.serialize();
fs.writeFileSync('monitor-state.json', state);
```

---

## Types

All types are exported from the package entry point.

```typescript
import type {
  EmbedFn,
  LlmFn,
  DriftSeverity,
  LengthStats,
  VocabularyStats,
  Snapshot,
  MetricResult,
  DriftReport,
  MonitorOptions,
  ProbeOptions,
  ProbeResult,
  DriftMonitor,
} from 'prompt-drift';
```

### `EmbedFn`

```typescript
type EmbedFn = (text: string) => Promise<number[]>;
```

### `LlmFn`

```typescript
type LlmFn = (prompt: string) => Promise<string>;
```

### `DriftSeverity`

```typescript
type DriftSeverity = 'none' | 'low' | 'medium' | 'high' | 'critical';
```

### `Snapshot`

```typescript
interface Snapshot {
  id: string;
  promptId?: string;
  timestamp: number;
  embeddingModel: string;
  centroid: number[];
  spread: number;
  variance: number[];
  meanPairwiseSimilarity: number;
  stdPairwiseSimilarity: number;
  lengthStats: LengthStats;
  vocabularyStats: VocabularyStats;
  samples: string[];
}
```

### `LengthStats`

```typescript
interface LengthStats {
  mean: number;
  stddev: number;
  median: number;
  p5: number;
  p95: number;
}
```

### `VocabularyStats`

```typescript
interface VocabularyStats {
  topK: Array<{ term: string; frequency: number }>;
  vocabSize: number;
  totalTokenCount: number;
}
```

### `MetricResult`

```typescript
interface MetricResult {
  score: number;    // normalized to [0, 1]
  exceeded: boolean; // true if score exceeds the configured threshold
}
```

### `DriftReport`

```typescript
interface DriftReport {
  centroidDistance: MetricResult;
  spreadChange: MetricResult;
  jensenShannonDivergence: MetricResult;
  populationStabilityIndex: MetricResult;
  lengthDrift: MetricResult;
  vocabularyDrift: MetricResult;
  compositeDriftScore: number;  // weighted average, [0, 1]
  severity: DriftSeverity;
  explanation: string;          // human-readable summary
}
```

### `ProbeResult`

```typescript
interface ProbeResult {
  snapshot: Snapshot;
  report: DriftReport | null; // null when no baseline is set
}
```

### `ProbeOptions`

```typescript
interface ProbeOptions {
  sampleCount?: number; // default: 5
  timeoutMs?: number;
}
```

---

## Configuration

### Thresholds

Thresholds control when individual metrics are flagged as `exceeded` in the drift report. Lower thresholds increase sensitivity.

```typescript
const monitor = createMonitor({
  embedFn,
  thresholds: {
    centroidDistance: 0.05, // more sensitive to semantic shift
    spreadChange: 0.2,
    jsd: 0.1,
    psi: 0.2,
    lengthDrift: 0.15,     // more sensitive to length changes
    vocabularyDrift: 0.3,
  },
});
```

### Weights

Weights control each metric's contribution to the composite drift score. They do not need to sum to 1 -- the composite score is a direct weighted sum of individual metric scores (each in [0, 1]).

```typescript
const monitor = createMonitor({
  embedFn,
  weights: {
    centroidDistance: 0.3,   // emphasize semantic drift
    spreadChange: 0.1,
    jsd: 0.25,
    psi: 0.15,
    lengthDrift: 0.1,
    vocabularyDrift: 0.1,
  },
});
```

### Alert Cooldown

The `alertCooldownMs` option prevents alert storms when drift is detected across multiple consecutive comparisons.

```typescript
const monitor = createMonitor({
  embedFn,
  alertCooldownMs: 300000, // at most one alert every 5 minutes
  onDrift: (report) => {
    notifySlack(`Drift detected: ${report.severity} -- ${report.explanation}`);
  },
});
```

---

## Error Handling

### No Baseline Set

Calling `monitor.check()` without a baseline throws a synchronous error:

```typescript
try {
  const report = await monitor.check(outputs);
} catch (err) {
  // Error: 'No baseline set. Call setBaseline() first.'
}
```

Calling `monitor.probe()` without a baseline does not throw. Instead, the returned `ProbeResult.report` is `null`.

### Embedding Function Errors

If the `embedFn` provided to `createMonitor` throws or rejects, the error propagates from `monitor.snapshot()`, `monitor.check()`, and `monitor.probe()`. Wrap calls in try/catch to handle embedding failures:

```typescript
try {
  const snap = await monitor.snapshot(outputs);
} catch (err) {
  console.error('Embedding failed:', err);
}
```

### Empty or Minimal Inputs

Passing an empty array to `monitor.snapshot()` will produce a snapshot with empty centroid and zero-valued statistics. For meaningful drift detection, provide at least 20 output samples.

---

## Advanced Usage

### Scheduled Probing with Cron

Run a probe on a schedule to detect model changes before they affect users:

```typescript
import { createMonitor } from 'prompt-drift';

const monitor = createMonitor({
  embedFn,
  onDrift: (report) => {
    if (report.severity === 'high' || report.severity === 'critical') {
      pagerDuty.trigger({
        summary: `Prompt drift: ${report.severity}`,
        details: report.explanation,
      });
    }
  },
});

// Load persisted baseline
const saved = JSON.parse(fs.readFileSync('baseline.json', 'utf-8'));
monitor.setBaseline(saved.baseline);

// Probe every hour
setInterval(async () => {
  const { report } = await monitor.probe('Summarize the key benefits of exercise.', llmFn, {
    sampleCount: 20,
  });
  if (report) {
    console.log(`[${new Date().toISOString()}] drift=${report.compositeDriftScore.toFixed(3)} severity=${report.severity}`);
  }
}, 3600000);
```

### CI/CD Gate

Fail a deployment pipeline when prompt behavior has drifted:

```typescript
const { report } = await monitor.probe(prompt, llmFn, { sampleCount: 30 });

if (report && (report.severity === 'high' || report.severity === 'critical')) {
  console.error(`Deployment blocked: prompt drift severity is ${report.severity}`);
  console.error(report.explanation);
  process.exit(1);
}
```

### Comparing Arbitrary Snapshots

Use `monitor.compare()` to compare any two snapshots without setting a baseline. This is useful for ad-hoc investigations:

```typescript
const snapA = await monitor.snapshot(outputsFromLastWeek);
const snapB = await monitor.snapshot(outputsFromThisWeek);
const report = monitor.compare(snapA, snapB);

console.log('Week-over-week drift:', report.compositeDriftScore.toFixed(3));
for (const [metric, result] of Object.entries(report)) {
  if (typeof result === 'object' && result !== null && 'exceeded' in result && result.exceeded) {
    console.log(`  ${metric}: ${result.score.toFixed(3)} [EXCEEDED]`);
  }
}
```

### Persisting and Restoring State

Save the monitor state (including baseline) to disk and restore it later:

```typescript
// Save
const state = monitor.serialize();
fs.writeFileSync('monitor-state.json', state);

// Restore
const restored = JSON.parse(fs.readFileSync('monitor-state.json', 'utf-8'));
if (restored.baseline) {
  monitor.setBaseline(restored.baseline);
}
```

### Inspecting Individual Metrics

Each metric in the drift report includes a `score` (normalized 0-1) and an `exceeded` flag:

```typescript
const report = monitor.compare(baseline, current);

if (report.centroidDistance.exceeded) {
  console.log('Semantic content has shifted.');
}
if (report.lengthDrift.exceeded) {
  console.log('Output verbosity has changed.');
}
if (report.vocabularyDrift.exceeded) {
  console.log('The model is using different words.');
}
```

---

## Drift Metrics

Six metrics contribute to the composite drift score. Each produces a normalized score in [0, 1].

| Metric | What It Measures | How It Works |
|---|---|---|
| **Centroid Distance** | Semantic shift in embedding space | Euclidean distance between the centroid (mean embedding vector) of the baseline and current snapshots. Normalized to [0, 1]. |
| **Spread Change** | Change in output diversity | Relative change in average distance from each embedding to the centroid. Detects whether outputs are becoming more or less varied. |
| **Jensen-Shannon Divergence** | Distribution shape change | JSD between pairwise similarity distributions of baseline and current snapshots. Captures changes in internal coherence. |
| **Population Stability Index** | Output length distribution shift | PSI computed over binned output length distributions. Detects changes in how long or short outputs are. |
| **Length Drift** | Mean output length change | Relative change in mean character count. Catches verbosity shifts. |
| **Vocabulary Drift** | Lexical shift | 1 minus the overlap ratio of top-K vocabulary terms between baseline and current snapshots. Detects when the model uses different words. |

---

## Severity Levels

| Composite Score | Severity | Interpretation |
|---|---|---|
| < 0.10 | `none` | No significant drift. Normal LLM output variation. |
| 0.10 -- 0.25 | `low` | Minor variation. Monitor for trends. |
| 0.25 -- 0.50 | `medium` | Noticeable behavioral shift. Investigate sample outputs. |
| 0.50 -- 0.75 | `high` | Significant change. Review outputs and consider re-establishing baseline. |
| >= 0.75 | `critical` | Major behavioral shift. Likely a model update. Immediate investigation recommended. |

---

## TypeScript

`prompt-drift` is written in TypeScript and ships with full type declarations (`.d.ts` files) and source maps. All public types are exported from the package entry point:

```typescript
import { createMonitor } from 'prompt-drift';
import type { DriftMonitor, DriftReport, Snapshot, MonitorOptions } from 'prompt-drift';
```

The package targets ES2022 and uses CommonJS module format.

---

## License

MIT
