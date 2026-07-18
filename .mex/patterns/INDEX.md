# Pattern Index

Lookup table for all pattern files in this directory. Check here before starting any task — if a pattern exists, follow it.

<!-- This file is populated during setup (Pass 2) and updated whenever patterns are added.
     Each row maps a pattern file (or section) to its trigger — when should the agent load it?

     Format — simple (one task per file):
     | `filename.md` | One-line description of when to use this pattern |

     Format — anchored (multi-section file, one row per task):
     | `filename.md#task-first-task` | When doing the first task |
     | `filename.md#task-second-task` | When doing the second task |

     Example (from a Flask API project):
     | `add-api-client.md` | Adding a new external service integration |
     | `debug-pipeline.md` | Diagnosing failures in the request pipeline |
     | `crud-operations.md#task-add-endpoint` | Adding a new API route with validation |
     | `crud-operations.md#task-add-model` | Adding a new database model |

     Keep this table sorted alphabetically. One row per task (not per file).
     If you create a new pattern, add it here. If you delete one, remove it. -->

| Pattern | Use when |
|---------|----------|
| [add-agent.md](add-agent.md) | Adding a new decision-making agent/policy to the live engine |
| [debug-bot-misplay.md](debug-bot-misplay.md) | Bot folds strong hands, raises trash, or loses to random — encoding/scaling checklist |
| [deploy-openpoker.md](deploy-openpoker.md) | Deploying a checkpoint to openpoker.ai over WebSocket |
| [eval-checkpoint.md](eval-checkpoint.md) | Measuring a trained checkpoint vs OpenSpiel or random baselines |
| [train-strategy.md#task-neural-deep-cfr](train-strategy.md#task-neural-deep-cfr) | Running or resuming a neural Deep CFR training job |
| [train-strategy.md#task-tabular-mccfr](train-strategy.md#task-tabular-mccfr) | Running or resuming a tabular MCCFR training job |
