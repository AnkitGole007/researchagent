# Ollama Cloud Models — Review & Comparison vs. Current QIL Primary/Fallback

> Research doc, 2026-07-29. Scope: survey every model listed at
> https://ollama.com/search?c=cloud, evaluate description/benchmarks/cost, and
> empirically compare the size-comparable candidates against this app's
> current QIL primary (`qwen/qwen3.6-27b` on Groq) and fallback
> (`google/gemma-4-26b-a4b-it:free` on OpenRouter). No production code changed
> as part of this doc — see "Recommendation" for whether anything should.

## TL;DR

**Keep the current primary (Groq `qwen3.6-27b`) and fallback (OpenRouter
`gemma-4-26b`).** Ollama Cloud has no model that's both (a) fast enough for a
step that runs on every query and (b) reliably accessible on a free account.
The one genuinely good result (`gemma4:31b-cloud`, 2.3s, best similarity score
seen all session) is 2-3x slower than Groq and belongs to a family whose own
smaller/cheaper tier (`qwen3.5:cloud`) is Pro-subscription-gated — a
structural mismatch with this app's all-free-tier architecture, not a
quality problem.

## Methodology

- Full catalog pulled live from https://ollama.com/search?c=cloud (18 models
  listed under the "cloud" filter, 2026-07-29).
- Per-model description, parameter count, context window, and "usage tier"
  (Ollama's own cost-proxy: light/medium/high, or explicit $/1M-token pricing
  for the heaviest models) read from each model's own page.
- Four candidates in the size/latency class actually relevant to this app's
  per-query QIL step (`gpt-oss:20b-cloud`, `gemma4:31b-cloud`,
  `nemotron-3-nano:30b-cloud`, `qwen3.5:cloud`) were tested live via the
  OpenAI-compatible endpoint `https://ollama.com/v1`, using the
  `OLLAMA_API_KEY` already configured in `.env` for this app's separate
  paper-summarization LLM-provider feature (`backend/runner.py`'s
  `_PROVIDER_API_BASE["ollama_cloud"]`) — no new dependency or integration
  needed, just a different base URL with the existing `openai` SDK, same
  pattern as `_call_openrouter_llm`.
- Only **one** brief (`recsys_broad_no_acronym`, the original bug-motivating
  query) was run per model, not the full 4-brief sweep used for Groq/
  OpenRouter — Ollama Cloud's free tier is explicitly "light usage" with
  session/weekly caps (see Cost section), so this deliberately conserved quota
  until it was clear which models were even viable.
- Metrics: same groundedness ratio (lexical overlap) and semantic similarity
  (local MiniLM cosine) as the Groq/OpenRouter comparisons earlier this
  session.

## Full catalog (as listed under "cloud", 2026-07-29)

| Model | Description | Params | Usage / cost | Fit for this task |
|---|---|---|---|---|
| `glm-5.2` | Z.ai flagship, long-horizon coding/agentic tasks, 1M context | 756B | **high** | No — frontier coding agent, wrong shape |
| `kimi-k3` | Moonshot's most capable model, native multimodal agentic | 2.81T | $3/$0.30/$15 per 1M (in/cached/out), **Pro/Max only** | No — not accessible on free tier at all |
| `gemma4` | Google DeepMind, frontier-per-size, reasoning + multimodal | 2B–31B (dense/MoE tags) | light–medium | **Yes — tested, see below** |
| `qwen3.5` | Alibaba, multimodal, hybrid Gated-Delta+MoE architecture | 0.8B–397B (only `:cloud`/`:397b-cloud` hosted) | **Pro-gated on cloud** | Partial — only the 397B flagship is cloud-hosted, and it's Pro-only |
| `glm-5.1` | Z.ai, agentic engineering/coding, SWE-Bench Pro-tuned | (flagship-scale) | high | No — coding-agent focus |
| `minimax-m2.7` | MiniMax, coding/agentic/productivity | 229B | medium | No — too heavy for per-query use |
| `nemotron-3-super` | NVIDIA, 120B MoE (12B active), multi-agent focus | 120B (12B active) | medium-ish | No — still heavier than needed |
| `minimax-m2.5` | MiniMax, prior-gen coding/productivity model | ~229B class | medium | No |
| `minimax-m3` | MiniMax, coding/agentic frontier, 1M context, multimodal | (flagship-scale) | medium-high | No |
| `kimi-k2.7-code` | Moonshot, coding-focused agentic, lower thinking-token usage | (flagship-scale) | high | No — coding-agent focus |
| `kimi-k2.6` | Moonshot, long-horizon coding/design/orchestration | (flagship-scale) | high | No |
| `deepseek-v4-pro` | DeepSeek, frontier MoE, large context, 3 reasoning modes | (flagship-scale) | high | No |
| `deepseek-v4-flash` | DeepSeek, efficient MoE, 3 thinking modes incl. "no thinking" | 284B total / 13B active | **medium** | No — still 13B active, heavier than our primary |
| `nemotron-3-ultra` | NVIDIA, high-throughput reasoning, long-running agents | (flagship-scale) | high | No |
| `gpt-oss` | OpenAI open-weight, reasoning/agentic, configurable effort | 20B / 120B | **light (20b)** | **Tested — fails, see below** |
| `gemini-3-flash-preview` | Google, "frontier intelligence built for speed", proprietary | closed-weight | — | Speed-oriented but proprietary/gateway model, not evaluated live this round |
| `nemotron-3-nano` | NVIDIA, hybrid Mamba-2+MoE, configurable reasoning trace | 30B total / 3.5B active (also 4B dense tag) | light-ish | **Tested — fails as-is, see below** |
| `kimi-k2.5` | Moonshot, native multimodal agentic, instant + thinking modes | (flagship-scale) | high | No |
| `mistral-large-3` | Mistral, general-purpose multimodal MoE, enterprise-grade | 675B | medium | No |

The 12 frontier-scale models (everything except `gemma4`, `gpt-oss`,
`nemotron-3-nano`, and `qwen3.5`) are all built for long-horizon coding/agentic
workloads — hours-to-tens-of-hours SWE-bench-style tasks, 1M-token contexts,
multi-agent orchestration. That's a fundamentally different shape of problem
than a single ~900-token JSON extraction that has to return in under a
couple of seconds on every search. None were tested live: their "usage tier"
badges (medium-to-high, or explicit $3-15/1M-token pricing) and multi-hundred-
billion parameter counts make them structurally the wrong tool regardless of
per-brief quality, the same reasoning that ruled out Groq's `groq/compound`
agentic models earlier this session.

## Deep dive: the four size-comparable candidates

### `gemma4:31b-cloud` — same family as the current OpenRouter fallback

Google's own benchmark table (from the model's Ollama page) for the two cloud
tags plus the smaller local-only ones:

| | Gemma 4 31B (dense) | Gemma 4 26B A4B (MoE, = current OpenRouter fallback's family) |
|---|---|---|
| MMLU Pro | 85.2% | 82.6% |
| GPQA Diamond | 84.3% | 82.3% |
| AIME 2026 (no tools) | 89.2% | 88.3% |
| LiveCodeBench v6 | 80.0% | 77.1% |

Confirms the current fallback (`gemma-4-26b-a4b-it:free` on OpenRouter) and
this candidate (`gemma4:31b-cloud` on Ollama) are close siblings — the 31B is
the dense, slightly-stronger-on-paper variant; only the 31B has a `-cloud` tag
on Ollama (the 26B MoE isn't offered through Ollama's hosted API at all, only
as a local download).

**Live result** (recsys brief): succeeded, **2296ms**, groundedness 0.2,
similarity **0.369** — the best similarity score of any model tested this
session on this specific brief (edging out the current primary's 0.358 on the
same brief). Keywords leaned toward standard recsys vocabulary
(collaborative-filtering, matrix-factorization, content-based-filtering,
rating-prediction) rather than the brief's own literal words, which is why
groundedness reads low despite the similarity being high — the same
"good paraphrase, not hallucination" pattern seen from `gemma-4-26b` on
OpenRouter earlier.

2.3s is dramatically faster than the current OpenRouter fallback's 9-16s for
the same family, but still 2-3x slower than Groq's sub-second responses for
the current primary — meaningful if this ran on every query, less so as a
fallback tier (where OpenRouter's 9-16s is already the accepted cost of a
degraded path).

### `qwen/qwen3.5` (Ollama) vs. `qwen/qwen3.6-27b` (current Groq primary)

Qwen3.5's flagship benchmark table (397B-A17B MoE, the only cloud-hosted
size) is genuinely frontier-class — competitive with GPT-5.2, Claude 4.5
Opus, and Gemini-3 Pro on MMLU-Pro (87.8%), GPQA (88.4%), and SWE-bench
Verified (76.2%). But this comparison is somewhat moot for this app:

- **`qwen3.5:cloud` returned a 403 on the live probe: "this model requires a
  subscription, upgrade for access."** The free tier does not include this
  model at all — not a quality or latency finding, a hard access wall.
- Only the 397B flagship has a `:cloud` tag; none of the smaller local tags
  (27B, 35B, 9B — any of which would be a fairer size-for-size comparison
  against the current 27B primary) are hosted on Ollama's cloud API at any
  price. To use this family via Ollama Cloud at all means running the 397B
  version, which is a different weight class entirely from what this app
  needs.

### `gpt-oss:20b-cloud` — third host, same failure

Already flagged twice this session (on native Groq hosting and on
OpenRouter) as a reasoning model that burns its whole token budget on an
internal `<think>` block before ever emitting JSON. Tested again here as the
third distinct hosting provider: **same failure** — empty response, 13.9s
elapsed before giving up. This closes the loop on that finding: it is
conclusively a property of the model itself, not any one host's
infrastructure, latency, or rate-limiting behavior.

### `nemotron-3-nano:30b-cloud` — fails as-is, cause not fully resolved

NVIDIA's own model card describes a configurable reasoning trace ("if the
user prefers the model to provide its final answer without intermediate
reasoning traces, it can be configured to do so"), similar in spirit to
qwen3.6's `reasoning_effort` fix. Live probe returned **empty content** on the
default call (5.1s). Tried NVIDIA's documented convention for disabling
Nemotron's chain-of-thought (appending `"detailed thinking off"` to the
system prompt) — **still empty content**. Given the free-tier quota
constraints, this wasn't chased further; recorded as "fails out of the box,
the standard workaround didn't fix it in one attempt," not as "provably
unfixable."

## Cost & accessibility — the deciding factor

This matters more than per-brief quality for a step that runs on *every*
query:

- Ollama Cloud's **Free** plan is explicitly "light usage" with **session
  limits that reset every 5 hours and weekly limits that reset every 7
  days** — not the effectively-unlimited-for-this-workload throughput Groq's
  free tier gives the current primary.
- Model access is gated by plan: `qwen3.5:cloud` needs Pro ($20/mo);
  `kimi-k3` needs Pro/Max plus **per-token billing on top** ($3 in / $0.30
  cached / $15 out per 1M tokens). The heavier frontier models are priced
  explicitly per token; the lighter ones (`gemma4`, `gpt-oss`,
  `nemotron-3-nano`) are metered by an abstracted "usage level"
  (light/medium/high) against the plan's included quota instead.
- Net effect: even the one candidate that performed well
  (`gemma4:31b-cloud`) would need this app's users to either stay within a
  light-usage free-tier session cap or pay $20+/mo — a real constraint this
  app's current all-free architecture doesn't have with Groq or OpenRouter's
  free tier.

## Comparison summary

| | Groq `qwen3.6-27b` (current primary) | OpenRouter `gemma-4-26b` (current fallback) | Ollama `gemma4:31b-cloud` | Ollama `gpt-oss:20b-cloud` | Ollama `nemotron-3-nano:30b-cloud` | Ollama `qwen3.5:cloud` |
|---|---|---|---|---|---|---|
| Success (this brief) | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ (403, needs Pro) |
| Latency | ~900ms | ~9-16s | **2.3s** | 13.9s (then failed) | 5.1s (then failed) | n/a |
| Similarity | 0.358 | 0.294 | **0.369** | n/a | n/a | n/a |
| Free-tier access | Yes, generous | Yes, rate-limited | Yes, "light usage" caps | Yes, "light usage" caps | Yes, "light usage" caps | **No — Pro required** |

## Recommendation

No change to production. The current primary/fallback pair remains the right
choice:

- **Nothing on Ollama Cloud beats Groq's `qwen3.6-27b` on speed** — the one
  strong quality result (`gemma4:31b-cloud`) is still 2-3x slower, before
  even accounting for free-tier session caps that Groq's free tier doesn't
  impose on this workload.
- **The direct sibling of the current fallback family performed well**
  (`gemma4:31b-cloud`, best similarity score of the session) and could be a
  reasonable *alternative* fallback to `gemma-4-26b` on OpenRouter if the
  OpenRouter leg ever became unreliable — genuinely faster (2.3s vs 9-16s)
  for a comparable quality tier. Not swapped in now since the current
  fallback isn't broken; recorded here as a validated option if that
  changes.
- **`gpt-oss` is now confirmed unreliable on all three hosts tested this
  session** (Groq, OpenRouter, Ollama) — this is a model property, not a
  hosting quirk, closing that open question from the prior comparison.
- **`qwen3.5`, the family most comparable to the current primary, isn't
  accessible on Ollama's free tier at any size** — moot for this app unless
  a paid plan is on the table, which is out of scope for a free-tier product.

## What remains open

- `nemotron-3-nano:30b-cloud`'s failure mode wasn't fully root-caused — the
  one documented reasoning-toggle attempt didn't fix it, but this wasn't
  chased further given free-tier quota limits. Low priority given it
  wouldn't beat the current primary/fallback on latency even if fixed.
- Only one brief was tested per Ollama model (vs. four for Groq/OpenRouter)
  to conserve free-tier quota — directional signal only, same caveat as
  every prior comparison this session.
- `gemini-3-flash-preview` (proprietary, gateway-hosted, no public parameter
  count) wasn't tested live — flagged as "speed-oriented" by its own
  description but not verified against this task.
