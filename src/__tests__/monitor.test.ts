import { describe, it, expect } from 'vitest'
import { createMonitor } from '../monitor'
import type { Snapshot } from '../types'

// Mock embedder: returns a deterministic vector based on text content.
// "different" texts produce orthogonal-ish vectors to simulate real drift.
function makeMockEmbedFn(offset = 0) {
  return async (text: string): Promise<number[]> => {
    const seed = text.split('').reduce((s, c) => s + c.charCodeAt(0), 0) + offset
    const dim = 8
    const vec: number[] = []
    for (let i = 0; i < dim; i++) {
      vec.push(Math.sin(seed * (i + 1) * 0.1))
    }
    // Normalize
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0)) || 1
    return vec.map(v => v / norm)
  }
}

const SIMILAR_OUTPUTS = [
  'The quick brown fox jumps over the lazy dog.',
  'The quick brown fox leaps over the lazy dog.',
  'A quick brown fox jumps over the lazy dog.',
  'The quick brown fox jumps past the lazy dog.',
  'The fast brown fox jumps over the lazy dog.',
]

const DIFFERENT_OUTPUTS = [
  'Hello world this is a test of completely different content.',
  'Banana apple orange grape fruit salad recipe here.',
  'Mathematics calculus integration differential equations solutions.',
  'Programming languages Python JavaScript TypeScript Rust Go.',
  'Music theory harmony rhythm melody counterpoint fugue.',
]

describe('createMonitor', () => {
  it('snapshot() returns a Snapshot with correct structure', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS, { promptId: 'test-prompt' })

    expect(snap.id).toBeTypeOf('string')
    expect(snap.id.length).toBeGreaterThan(0)
    expect(snap.promptId).toBe('test-prompt')
    expect(snap.timestamp).toBeTypeOf('number')
    expect(snap.embeddingModel).toBe('provided')
    expect(snap.centroid).toBeInstanceOf(Array)
    expect(snap.centroid.length).toBe(8)
    expect(snap.spread).toBeTypeOf('number')
    expect(snap.meanPairwiseSimilarity).toBeTypeOf('number')
    expect(snap.stdPairwiseSimilarity).toBeTypeOf('number')
    expect(snap.lengthStats.mean).toBeTypeOf('number')
    expect(snap.lengthStats.median).toBeTypeOf('number')
    expect(snap.lengthStats.p5).toBeTypeOf('number')
    expect(snap.lengthStats.p95).toBeTypeOf('number')
    expect(snap.vocabularyStats.topK).toBeInstanceOf(Array)
    expect(snap.vocabularyStats.vocabSize).toBeGreaterThan(0)
    expect(snap.samples).toEqual(SIMILAR_OUTPUTS)
  })

  it('compare() returns DriftReport with compositeDriftScore in [0,1]', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const base = await monitor.snapshot(SIMILAR_OUTPUTS)
    const curr = await monitor.snapshot(DIFFERENT_OUTPUTS)
    const report = monitor.compare(base, curr)

    expect(report.compositeDriftScore).toBeGreaterThanOrEqual(0)
    expect(report.compositeDriftScore).toBeLessThanOrEqual(1)
    expect(report.centroidDistance.score).toBeGreaterThanOrEqual(0)
    expect(report.centroidDistance.score).toBeLessThanOrEqual(1)
    expect(report.spreadChange.score).toBeGreaterThanOrEqual(0)
    expect(report.jensenShannonDivergence.score).toBeGreaterThanOrEqual(0)
    expect(report.populationStabilityIndex.score).toBeGreaterThanOrEqual(0)
    expect(report.lengthDrift.score).toBeGreaterThanOrEqual(0)
    expect(report.vocabularyDrift.score).toBeGreaterThanOrEqual(0)
    expect(['none', 'low', 'medium', 'high', 'critical']).toContain(report.severity)
    expect(report.explanation).toBeTypeOf('string')
    expect(report.explanation.length).toBeGreaterThan(0)
  })

  it('compare(same, same) produces severity=none and low composite score', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    // compare a snapshot against itself — zero drift
    const report = monitor.compare(snap, snap)

    expect(report.severity).toBe('none')
    expect(report.compositeDriftScore).toBeLessThan(0.1)
    expect(report.centroidDistance.exceeded).toBe(false)
    expect(report.spreadChange.exceeded).toBe(false)
    expect(report.lengthDrift.exceeded).toBe(false)
    expect(report.vocabularyDrift.exceeded).toBe(false)
  })

  it('compare(base, very_different) produces higher composite score than compare(same, same)', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const base = await monitor.snapshot(SIMILAR_OUTPUTS)
    const same = await monitor.snapshot(SIMILAR_OUTPUTS)
    const different = await monitor.snapshot(DIFFERENT_OUTPUTS)

    const sameReport = monitor.compare(base, same)
    const diffReport = monitor.compare(base, different)

    expect(diffReport.compositeDriftScore).toBeGreaterThanOrEqual(sameReport.compositeDriftScore)
  })

  it('check() throws when no baseline is set', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    await expect(monitor.check(SIMILAR_OUTPUTS)).rejects.toThrow('No baseline set')
  })

  it('setBaseline/getBaseline roundtrip', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    expect(monitor.getBaseline()).toBeNull()
    expect(monitor.hasBaseline).toBe(false)

    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    monitor.setBaseline(snap)

    expect(monitor.getBaseline()).toBe(snap)
    expect(monitor.hasBaseline).toBe(true)
  })

  it('check() uses baseline and returns DriftReport', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    monitor.setBaseline(snap)

    const report = await monitor.check(DIFFERENT_OUTPUTS)
    expect(report.compositeDriftScore).toBeGreaterThanOrEqual(0)
    expect(report.compositeDriftScore).toBeLessThanOrEqual(1)
    expect(['none', 'low', 'medium', 'high', 'critical']).toContain(report.severity)
  })

  it('probe() calls llmFn N times and returns snapshot + report', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const baseSnap = await monitor.snapshot(SIMILAR_OUTPUTS)
    monitor.setBaseline(baseSnap)

    const callCount = { n: 0 }
    const llmFn = async (_prompt: string): Promise<string> => {
      callCount.n++
      return `Response number ${callCount.n} to the prompt.`
    }

    const result = await monitor.probe('test prompt', llmFn, { sampleCount: 3 })

    expect(callCount.n).toBe(3)
    expect(result.snapshot.samples).toHaveLength(3)
    expect(result.report).not.toBeNull()
    expect(result.report!.compositeDriftScore).toBeGreaterThanOrEqual(0)
  })

  it('probe() returns report=null when no baseline is set', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const llmFn = async (_prompt: string): Promise<string> => 'some output'
    const result = await monitor.probe('test prompt', llmFn, { sampleCount: 2 })
    expect(result.report).toBeNull()
    expect(result.snapshot.samples).toHaveLength(2)
  })

  it('onDrift callback fires when severity is not none', async () => {
    const driftAlerts: unknown[] = []
    const monitor = createMonitor({
      embedFn: makeMockEmbedFn(),
      alertCooldownMs: 0,
      onDrift: report => driftAlerts.push(report),
      thresholds: {
        centroidDistance: 0.001, // very sensitive — will exceed easily
        lengthDrift: 0.001,
        vocabularyDrift: 0.001,
      },
    })

    const base = await monitor.snapshot(SIMILAR_OUTPUTS)
    const different = await monitor.snapshot(DIFFERENT_OUTPUTS)
    monitor.compare(base, different)

    expect(driftAlerts.length).toBeGreaterThan(0)
  })

  it('onDrift callback respects alertCooldownMs', async () => {
    const driftAlerts: unknown[] = []
    const monitor = createMonitor({
      embedFn: makeMockEmbedFn(),
      alertCooldownMs: 999999, // very long cooldown
      onDrift: report => driftAlerts.push(report),
      thresholds: { centroidDistance: 0.001, lengthDrift: 0.001, vocabularyDrift: 0.001 },
    })

    const base = await monitor.snapshot(SIMILAR_OUTPUTS)
    const different = await monitor.snapshot(DIFFERENT_OUTPUTS)
    monitor.compare(base, different)
    monitor.compare(base, different)
    monitor.compare(base, different)

    // Cooldown prevents multiple fires
    expect(driftAlerts.length).toBe(1)
  })

  it('serialize() returns valid JSON containing baseline', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    monitor.setBaseline(snap)

    const serialized = monitor.serialize()
    const parsed = JSON.parse(serialized) as { baseline: Snapshot }
    expect(parsed.baseline).toBeDefined()
    expect(parsed.baseline.id).toBe(snap.id)
  })

  it('snapshot() without promptId has undefined promptId', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    expect(snap.promptId).toBeUndefined()
  })

  it('vocabularyDrift is 0 when comparing identical snapshots', async () => {
    const monitor = createMonitor({ embedFn: makeMockEmbedFn() })
    const snap = await monitor.snapshot(SIMILAR_OUTPUTS)
    const report = monitor.compare(snap, snap)
    expect(report.vocabularyDrift.score).toBe(0)
    expect(report.vocabularyDrift.exceeded).toBe(false)
  })
})
