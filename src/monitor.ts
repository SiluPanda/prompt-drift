import { randomUUID } from 'crypto'
import { centroid } from './stats'
import {
  computeSpread,
  computePairwiseSimilarity,
  computeLengthStats,
  computeVocabularyStats,
} from './stats'
import {
  DEFAULT_THRESHOLDS,
  metricCentroidDistance,
  metricSpreadChange,
  metricJSD,
  metricPSI,
  metricLengthDrift,
  metricVocabularyDrift,
} from './metrics'
import type {
  MonitorOptions,
  DriftMonitor,
  Snapshot,
  DriftReport,
  DriftSeverity,
  LlmFn,
  ProbeOptions,
  ProbeResult,
} from './types'

export function createMonitor(options: MonitorOptions): DriftMonitor {
  let baseline: Snapshot | null = null
  const thresholds = { ...DEFAULT_THRESHOLDS, ...options.thresholds }
  const weights = {
    centroidDistance: 0.2,
    spreadChange: 0.15,
    jsd: 0.2,
    psi: 0.2,
    lengthDrift: 0.15,
    vocabularyDrift: 0.1,
    ...options.weights,
  }
  let lastAlertTime = 0

  async function buildSnapshot(outputs: string[], promptId?: string): Promise<Snapshot> {
    const embeddings = await Promise.all(outputs.map(t => options.embedFn(t)))
    const c = centroid(embeddings)
    const pairwise = computePairwiseSimilarity(embeddings)
    const variance = c.map(() => 0)
    return {
      id: randomUUID(),
      promptId,
      timestamp: Date.now(),
      embeddingModel: 'provided',
      centroid: c,
      spread: computeSpread(embeddings, c),
      variance,
      meanPairwiseSimilarity: pairwise.mean,
      stdPairwiseSimilarity: pairwise.std,
      lengthStats: computeLengthStats(outputs),
      vocabularyStats: computeVocabularyStats(outputs),
      samples: outputs,
    }
  }

  function compareFn(base: Snapshot, curr: Snapshot): DriftReport {
    const cd = metricCentroidDistance(base, curr, thresholds.centroidDistance)
    const sc = metricSpreadChange(base, curr, thresholds.spreadChange)
    const jsd = metricJSD(base, curr, thresholds.jsd)
    const psi = metricPSI(base, curr, thresholds.psi)
    const ld = metricLengthDrift(base, curr, thresholds.lengthDrift)
    const vd = metricVocabularyDrift(base, curr, thresholds.vocabularyDrift)

    const composite =
      cd.score * weights.centroidDistance +
      sc.score * weights.spreadChange +
      jsd.score * weights.jsd +
      psi.score * weights.psi +
      ld.score * weights.lengthDrift +
      vd.score * weights.vocabularyDrift

    const severity: DriftSeverity =
      composite < 0.1
        ? 'none'
        : composite < 0.25
        ? 'low'
        : composite < 0.5
        ? 'medium'
        : composite < 0.75
        ? 'high'
        : 'critical'

    const metricNames = ['centroid', 'spread', 'jsd', 'psi', 'length', 'vocabulary']
    const exceeded = [cd, sc, jsd, psi, ld, vd]
      .map((m, i) => (m.exceeded ? metricNames[i] : null))
      .filter((n): n is NonNullable<typeof n> => n !== null)

    const explanation =
      exceeded.length === 0
        ? 'No significant drift detected.'
        : `Drift detected in: ${exceeded.join(', ')}. Composite score: ${composite.toFixed(3)}.`

    const report: DriftReport = {
      centroidDistance: cd,
      spreadChange: sc,
      jensenShannonDivergence: jsd,
      populationStabilityIndex: psi,
      lengthDrift: ld,
      vocabularyDrift: vd,
      compositeDriftScore: composite,
      severity,
      explanation,
    }

    if (options.onDrift && severity !== 'none') {
      const now = Date.now()
      const cooldown = options.alertCooldownMs ?? 60000
      if (now - lastAlertTime > cooldown) {
        options.onDrift(report)
        lastAlertTime = now
      }
    }

    return report
  }

  return {
    async snapshot(outputs, opts) {
      return buildSnapshot(outputs, opts?.promptId)
    },

    compare: compareFn,

    async probe(
      prompt: string,
      llmFn: LlmFn,
      probeOptions?: ProbeOptions
    ): Promise<ProbeResult> {
      const count = probeOptions?.sampleCount ?? 5
      const outputs = await Promise.all(Array.from({ length: count }, () => llmFn(prompt)))
      const snap = await buildSnapshot(outputs, prompt)
      const report = baseline ? compareFn(baseline, snap) : null
      return { snapshot: snap, report }
    },

    async check(newOutputs: string[]): Promise<DriftReport> {
      if (!baseline) throw new Error('No baseline set. Call setBaseline() first.')
      const curr = await buildSnapshot(newOutputs)
      return compareFn(baseline, curr)
    },

    setBaseline(snap: Snapshot): void {
      baseline = snap
    },

    getBaseline(): Snapshot | null {
      return baseline
    },

    serialize(): string {
      return JSON.stringify({ baseline })
    },

    get hasBaseline(): boolean {
      return baseline !== null
    },
  }
}
