# HU 200 BB roadmap results

| Date | Step | Arm | Iterations | Infosets | it/s | H2H | Slumbot BB/100 | Cost |
|------|------|-----|-----------:|---------:|-----:|-----|----------------|-----:|
| 2026-10-21 | Local benchmark (M-series laptop) | C0 hu200.toml | 600,000 | 33,647 | 939 cumulative (994/999/841 per 200k window) | - | - | $0 |
| 2026-10-21 | ccx33 benchmark (8 dedicated vCPU) | C0 hu200.toml | 600,000 | 33,637 | 3,935 | - | - | $0.01 (351 s incl. setup) |
| 2026-10-21 | Slumbot freeze check (guest) | hu_blueprint_50000000.bin, hu.toml | 50,000,000 | - | - | - | raw -107.2 ±42.5, baseline -95.1 ±41.1 (3,000 hands); PASS vs -118.3 ±42 | $0 |

Estimates at 3,935 it/s and about $0.22/h (before larger arms grow the infoset table): C0 to the 60M cap is about 4.2 h, about $1. Six to eight arms at that scale is under $10, well inside the $40 budget. Re-estimate after C0 shows how it/s holds as infosets grow.
