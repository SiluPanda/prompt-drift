import { jensenShannonDivergence, histogramBins } from './math'
import type { Snapshot, MetricResult } from './types'

export const DEFAULT_THRESHOLDS = {
  centroidDistance: 0.1,
  spreadChange: 0.2,
  jsd: 0.1,
  psi: 0.2,
  lengthDrift: 0.3,
  vocabularyDrift: 0.3,
}

export function metricCentroidDistance(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const dist = Math.sqrt(
    base.centroid.reduce((s, v, i) => s + (v - (curr.centroid[i] ?? 0)) ** 2, 0)
  )
  return { score: Math.min(1, dist / 2), exceeded: dist > threshold }
}

export function metricSpreadChange(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const delta = Math.abs(curr.spread - base.spread) / Math.max(base.spread, 0.001)
  return { score: Math.min(1, delta), exceeded: delta > threshold }
}

export function metricJSD(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const bins = 10
  const baseHist = histogramBins(
    base.samples.map(() => base.meanPairwiseSimilarity),
    bins
  )
  const currHist = histogramBins(
    curr.samples.map(() => curr.meanPairwiseSimilarity),
    bins
  )
  const jsd = jensenShannonDivergence(baseHist, currHist)
  return { score: Math.min(1, jsd), exceeded: jsd > threshold }
}

export function metricPSI(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const bins = 10
  const baseLengths = base.samples.map(s => s.length)
  const currLengths = curr.samples.map(s => s.length)
  const bHist = histogramBins(baseLengths, bins).map(v => Math.max(v, 1e-10))
  const cHist = histogramBins(currLengths, bins).map(v => Math.max(v, 1e-10))
  const psi = bHist.reduce((s, b, i) => s + (b - cHist[i]) * Math.log(b / cHist[i]), 0)
  const normalized = Math.min(1, Math.abs(psi) / 2)
  return { score: normalized, exceeded: Math.abs(psi) > threshold }
}

export function metricLengthDrift(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const delta =
    Math.abs(curr.lengthStats.mean - base.lengthStats.mean) /
    Math.max(base.lengthStats.mean, 1)
  return { score: Math.min(1, delta), exceeded: delta > threshold }
}

export function metricVocabularyDrift(
  base: Snapshot,
  curr: Snapshot,
  threshold: number
): MetricResult {
  const baseTerms = new Set(base.vocabularyStats.topK.map(t => t.term))
  const currTerms = new Set(curr.vocabularyStats.topK.map(t => t.term))
  const intersection = [...baseTerms].filter(t => currTerms.has(t)).length
  const overlap = intersection / Math.max(baseTerms.size, currTerms.size, 1)
  const drift = 1 - overlap
  return { score: drift, exceeded: drift > threshold }
}
