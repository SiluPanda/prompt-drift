# prompt-drift

Detect silent LLM output changes over time via semantic drift analysis. Zero external runtime dependencies — you provide the embedding function.

## Install

```bash
npm install prompt-drift
```

## Quick Start

```typescript
import { createMonitor } from 'prompt-drift'

// Provide any embedding function — OpenAI, local model, etc.
const embedFn = async (text: string): Promise<number[]> => {
  // Example: call your embedding API here
  return myEmbeddingApi(text)
}

const monitor = createMonitor({
  embedFn,
  onDrift: report => {
    console.warn('Drift detected!', report.severity, report.explanation)
  },
})

// Capture a baseline snapshot from known-good outputs
const baselineOutputs = await collectLlmOutputs(prompt, 20)
const baseline = await monitor.snapshot(baselineOutputs, { promptId: 'my-prompt' })
monitor.setBaseline(baseline)

// Later — check new outputs for drift
const newOutputs = await collectLlmOutputs(prompt, 20)
const report = await monitor.check(newOutputs)
console.log(report.severity)           // 'none' | 'low' | 'medium' | 'high' | 'critical'
console.log(report.compositeDriftScore) // 0..1
console.log(report.explanation)
```

## API

### `createMonitor(options)`

Creates a `DriftMonitor` instance.

**Options:**

| Field | Type | Description |
|---|---|---|
| `embedFn` | `(text: string) => Promise<number[]>` | Required. Your embedding function. |
| `thresholds` | `object` | Per-metric thresholds. See defaults below. |
| `weights` | `object` | Per-metric weights for composite score. |
| `onDrift` | `(report: DriftReport) => void` | Called when severity is not `none`. |
| `alertCooldownMs` | `number` | Minimum ms between `onDrift` calls. Default: 60000. |

**Default thresholds:**

```typescript
{
  centroidDistance: 0.1,
  spreadChange:     0.2,
  jsd:              0.1,
  psi:              0.2,
  lengthDrift:      0.3,
  vocabularyDrift:  0.3,
}
```

### `monitor.snapshot(outputs, options?)`

Builds a statistical `Snapshot` from an array of LLM output strings.

```typescript
const snap = await monitor.snapshot(outputs, { promptId: 'optional-id' })
```

### `monitor.compare(baseline, current)`

Compares two snapshots and returns a `DriftReport` synchronously.

### `monitor.check(newOutputs)`

Builds a snapshot from `newOutputs` and compares against the stored baseline. Throws if no baseline is set.

### `monitor.probe(prompt, llmFn, options?)`

Calls `llmFn(prompt)` `sampleCount` times (default 5), builds a snapshot, and compares against baseline.

```typescript
const { snapshot, report } = await monitor.probe('Summarize X', myLlm, { sampleCount: 10 })
```

### `monitor.setBaseline(snapshot)` / `monitor.getBaseline()`

Store or retrieve the current baseline snapshot.

### `monitor.serialize()`

Returns a JSON string of the monitor state (baseline included). Useful for persistence.

### `monitor.hasBaseline`

Boolean — `true` if a baseline has been set.

## Drift Metrics

Six metrics contribute to the composite drift score (0..1):

| Metric | Description |
|---|---|
| `centroidDistance` | Euclidean distance between embedding centroids |
| `spreadChange` | Change in average distance from centroid |
| `jensenShannonDivergence` | JSD between pairwise similarity distributions |
| `populationStabilityIndex` | PSI over output length distributions |
| `lengthDrift` | Relative change in mean output length |
| `vocabularyDrift` | Jaccard distance over top-K vocabulary |

## Severity Levels

| Composite Score | Severity |
|---|---|
| < 0.10 | `none` |
| 0.10 – 0.25 | `low` |
| 0.25 – 0.50 | `medium` |
| 0.50 – 0.75 | `high` |
| >= 0.75 | `critical` |

## License

MIT
