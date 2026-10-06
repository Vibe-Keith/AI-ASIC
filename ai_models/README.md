# ai_models

Every AI file AI-ASIC uses lives here.

| Folder | Contents |
|--------|----------|
| `llm/` | GGUF language models for the chat engine (`*.gguf`). Drop any GGUF here, or run `python -m ai_asic.cli models download qwen2.5-0.5b`. |
| `hasher/` | Trained HASHER split-models (`*.json`) saved from the Workload tab or `ai-asic workload --save`. |
| `cache/` | `responses.jsonl` (SHA-256-addressed response cache, append-only) and `transcripts.jsonl` (chat turns sealed with mined proof-of-work). Safe to delete. |
| `config.json` | Chat defaults: model file, system prompt, context size, sampling, accelerator options. |

Model weights and caches are git-ignored. Set the `AI_ASIC_MODELS` environment variable to
keep this folder somewhere else.
