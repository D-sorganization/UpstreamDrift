# Assessment E: Performance & Scalability

## Assessment Overview

You are a **performance engineer** conducting an **adversarial, evidence-based** performance review. Your job is to identify **computational bottlenecks, memory issues, and scalability limitations** that impact usability.

---

## Key Metrics

| Metric              | Target          | Critical Threshold            |
| ------------------- | --------------- | ----------------------------- |
| Startup Time        | <3 seconds      | >10 seconds = BLOCKER         |
| Memory Usage (idle) | <200 MB         | >1 GB = CRITICAL              |
| Operation Time      | Documented ±20% | Undocumented = MAJOR          |
| Memory Leaks        | None            | Any confirmed leak = CRITICAL |

---

## Review Categories

### A. Startup Performance

- Time from launch to interactive state
- Import time for core modules
- Lazy loading of optional features
- Cold start vs warm start comparison

### B. Computational Efficiency

- CPU profiling of core operations
- Vectorization vs loops for data operations
- Parallel execution strategies
- Caching of expensive computations

### C. Memory Management

- Memory profiling of typical workflows
- Large dataset handling (1M+ records)
- Memory cleanup after operations
- Peak vs steady-state memory usage

### D. I/O Performance

- File loading times vs file sizes
- Streaming for large files
- Network request efficiency (if applicable)
- Disk write patterns

### E. Scalability Testing

| Data Size    | Expected Time | Memory  | Status |
| ------------ | ------------- | ------- | ------ |
| 1K records   | <1s           | <100 MB | ✅/❌  |
| 10K records  | <5s           | <200 MB | ✅/❌  |
| 100K records | <30s          | <500 MB | ✅/❌  |
| 1M records   | <5min         | <2 GB   | ✅/❌  |

---

## Output Format

### 1. Performance Profile

| Operation      | P50 Time | P99 Time | Memory Peak | Status |
| -------------- | -------- | -------- | ----------- | ------ |
| Startup        | X ms     | X ms     | X MB        | ✅/❌  |
| Load file      | X ms     | X ms     | X MB        | ✅/❌  |
| Core operation | X ms     | X ms     | X MB        | ✅/❌  |

### 2. Hotspot Analysis

| Location            | % CPU Time | Issue       | Fix            |
| ------------------- | ---------- | ----------- | -------------- |
| `module.function()` | X%         | Description | Recommendation |

### 3. Remediation Roadmap

**48 hours:** Quick wins (caching, obvious bottlenecks)
**2 weeks:** Vectorization, parallel execution
**6 weeks:** Architecture changes for scalability

---

_Assessment E focuses on performance. See Assessment A for architecture and Assessment D for user experience._
