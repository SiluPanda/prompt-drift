import { randomUUID } from 'crypto'

export type EmbedFn = (text: string) => Promise<number[]>
export type LlmFn = (prompt: string) => Promise<string>
export type DriftSeverity = 'none' | 'low' | 'medium' | 'high' | 'critical'

export interface LengthStats {
  mean: number
  stddev: number
  median: number
  p5: number
  p95: number
}

export interface VocabularyStats {
  topK: Array<{ term: string; frequency: number }>
  vocabSize: number
  totalTokenCount: number
}

export interface Snapshot {
  id: string
  promptId?: string
  timestamp: number
  embeddingModel: string
  centroid: number[]
  spread: number
  variance: number[]
  meanPairwiseSimilarity: number
  stdPairwiseSimilarity: number
  lengthStats: LengthStats
  vocabularyStats: VocabularyStats
  samples: string[]
}

export interface MetricResult {
  score: number
  exceeded: boolean
}

export interface DriftReport {
  centroidDistance: MetricResult
  spreadChange: MetricResult
  jensenShannonDivergence: MetricResult
  populationStabilityIndex: MetricResult
  lengthDrift: MetricResult
  vocabularyDrift: MetricResult
  compositeDriftScore: number
  severity: DriftSeverity
  explanation: string
}

export interface MonitorOptions {
  embedFn: EmbedFn
  thresholds?: {
    centroidDistance?: number
    spreadChange?: number
    jsd?: number
    psi?: number
    lengthDrift?: number
    vocabularyDrift?: number
    composite?: number
  }
  weights?: {
    centroidDistance?: number
    spreadChange?: number
    jsd?: number
    psi?: number
    lengthDrift?: number
    vocabularyDrift?: number
  }
  onDrift?: (report: DriftReport) => void
  alertCooldownMs?: number
}

export interface ProbeOptions {
  sampleCount?: number
  timeoutMs?: number
}

export interface ProbeResult {
  snapshot: Snapshot
  report: DriftReport | null
}

export interface DriftMonitor {
  snapshot(outputs: string[], options?: { promptId?: string }): Promise<Snapshot>
  compare(baseline: Snapshot, current: Snapshot): DriftReport
  probe(prompt: string, llmFn: LlmFn, options?: ProbeOptions): Promise<ProbeResult>
  check(newOutputs: string[]): Promise<DriftReport>
  setBaseline(snapshot: Snapshot): void
  getBaseline(): Snapshot | null
  serialize(): string
  readonly hasBaseline: boolean
}

// Re-export randomUUID to avoid unused import warning in types.ts
export { randomUUID }
