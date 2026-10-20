# Benchmark inputs

| Runs | Prompt | Image |
|---|---|---|
| FL2VA, 362 frames | [fl2va-15s.txt](fl2va-15s.txt) | [First frame](fl2va-first-frame.jpg) |
| Ref2VA PDD, 362 frames | [ref2va-15s.txt](ref2va-15s.txt) | [Reference](ref2va-reference.png) |
| Ref2VA Turbo/PDD and Base, 124 frames | [ref2va-5s.txt](ref2va-5s.txt) | [Reference](ref2va-reference.png) |

Pass the full Context-IR text through `--prompt-file`. Seeds, dimensions, steps, and GPU layouts are recorded in the [measurements](../README.md).
