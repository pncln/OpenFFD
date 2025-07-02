# OpenFFD Hierarchical FFD Performance Analysis Report
Generated on: 2025-07-02 08:02:33
## Executive Summary
### Performance Highlights:
- Best throughput: 53220151.7 points/second
  - Configuration: 1,000,000 points, depth=1, base=(6, 6, 6)
- Lowest throughput: 20861.6 points/second
  - Configuration: 50,000 points, depth=6, base=(6, 6, 6)

### Complexity Analysis:
- Depth scaling: exponential
- Exponential base: 2.323
- Optimal hierarchy depth: 1
- Best throughput at optimal depth: 44236173.0 points/second

### Memory Analysis:
- Memory per control point: 0.00 MB
- Base memory overhead: 8.12 MB

## Recommendations
### For Interactive Applications:
- Use hierarchy depth ≤ 3 for real-time performance
- Limit subdivision factor to 2 for balanced resolution
- Consider 4×4×4 base dimensions for general use
### For High-Fidelity Applications:
- Hierarchy depth 4-5 provides good detail/performance balance
- Use larger base dimensions (6×6×6) for complex geometries
- Monitor memory usage for depths > 4
### For Large-Scale Simulations:
- Enable parallel processing for meshes > 250K points
- Use progressive refinement strategy
- Consider memory constraints when designing hierarchy

## Technical Insights
### Creation Time by Depth:
- Depth 1: 0.008 seconds average
- Depth 2: 0.016 seconds average
- Depth 3: 0.034 seconds average
- Depth 4: 0.610 seconds average
- Depth 5: 0.258 seconds average
- Depth 6: 1.858 seconds average

### Control Point Growth:
- Depth 1: 140 control points average
- Depth 2: 1471 control points average
- Depth 3: 26656 control points average
- Depth 4: 1477548 control points average
- Depth 5: 655340 control points average
- Depth 6: 5242860 control points average
