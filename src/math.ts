export function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export function mean(arr: number[]): number {
  return arr.reduce((s, v) => s + v, 0) / arr.length
}

export function stddev(arr: number[], m?: number): number {
  const mu = m ?? mean(arr)
  return Math.sqrt(arr.reduce((s, v) => s + (v - mu) ** 2, 0) / arr.length)
}

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0
  const idx = (p / 100) * (sorted.length - 1)
  const lo = Math.floor(idx)
  const hi = Math.ceil(idx)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

export function centroid(embeddings: number[][]): number[] {
  if (embeddings.length === 0) return []
  const dim = embeddings[0].length
  const c = new Array(dim).fill(0) as number[]
  for (const e of embeddings) {
    for (let i = 0; i < dim; i++) c[i] += e[i]
  }
  return c.map(v => v / embeddings.length)
}

export function klDivergence(p: number[], q: number[]): number {
  let kl = 0
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0 && q[i] > 0) kl += p[i] * Math.log(p[i] / q[i])
  }
  return kl
}

export function jensenShannonDivergence(p: number[], q: number[]): number {
  const m = p.map((v, i) => 0.5 * (v + q[i]))
  return 0.5 * klDivergence(p, m) + 0.5 * klDivergence(q, m)
}

export function histogramBins(values: number[], bins: number): number[] {
  if (values.length === 0) return new Array(bins).fill(0) as number[]
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const counts = new Array(bins).fill(0) as number[]
  for (const v of values) {
    const bin = Math.min(bins - 1, Math.floor(((v - min) / range) * bins))
    counts[bin]++
  }
  return counts.map(c => c / values.length)
}
