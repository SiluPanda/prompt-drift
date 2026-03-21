import { mean, stddev, percentile, centroid, cosine } from './math'
import type { LengthStats, VocabularyStats } from './types'

export function computeLengthStats(outputs: string[]): LengthStats {
  const lengths = outputs.map(s => s.length).sort((a, b) => a - b)
  const m = mean(lengths)
  return {
    mean: m,
    stddev: stddev(lengths, m),
    median: percentile(lengths, 50),
    p5: percentile(lengths, 5),
    p95: percentile(lengths, 95),
  }
}

export function computeVocabularyStats(outputs: string[], topKCount = 20): VocabularyStats {
  const freqs: Record<string, number> = {}
  let total = 0
  for (const text of outputs) {
    const tokens = text.toLowerCase().match(/\b[a-z]{2,}\b/g) ?? []
    for (const t of tokens) {
      freqs[t] = (freqs[t] ?? 0) + 1
      total++
    }
  }
  const topK = Object.entries(freqs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, topKCount)
    .map(([term, frequency]) => ({ term, frequency }))
  return { topK, vocabSize: Object.keys(freqs).length, totalTokenCount: total }
}

export function computeSpread(embeddings: number[][], c: number[]): number {
  if (embeddings.length === 0) return 0
  const dists = embeddings.map(e =>
    Math.sqrt(e.reduce((s, v, i) => s + (v - c[i]) ** 2, 0))
  )
  return mean(dists)
}

export function computePairwiseSimilarity(embeddings: number[][]): { mean: number; std: number } {
  if (embeddings.length < 2) return { mean: 1, std: 0 }
  const sims: number[] = []
  for (let i = 0; i < embeddings.length; i++) {
    for (let j = i + 1; j < embeddings.length; j++) {
      sims.push(cosine(embeddings[i], embeddings[j]))
    }
  }
  const m = mean(sims)
  return { mean: m, std: stddev(sims, m) }
}

export { centroid }
