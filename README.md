<div align="center">

# 🦸 LLM Inference: Zero to Hero
### From "what is a token?" to serving LLMs fast with [SGLang](https://github.com/sgl-project/sglang), including on Intel GPUs

![chapters](https://img.shields.io/badge/chapters-3%20of%2018%20published-2a78d6?style=for-the-badge)
![path](https://img.shields.io/badge/path-zero%20→%20hero-1baf7a?style=for-the-badge)
![focus](https://img.shields.io/badge/focus-LLM%20serving-eb6834?style=for-the-badge)
![hardware](https://img.shields.io/badge/tested%20on-Intel%20Arc%20Pro%20B60-6250d6?style=for-the-badge)
![stars](https://img.shields.io/github/stars/arathi-hlab/llm-inference-zero-to-hero?style=for-the-badge&color=eda100)

✍️ by **Ashwini Rathi** · [@arathi-hlab](https://github.com/arathi-hlab)

</div>

---

## 👋 Start here

Clear, colourful, step-by-step explainers. Every chapter uses **one worked example (Llama 3 8B)**, real numbers, and diagrams that work in light and dark mode.
No prior ML background needed for Level 0. Each level builds on the one before.

```mermaid
flowchart LR
    L0["🌱 Level 0<br/>How an LLM works"]:::l0 --> L1["🚶 Level 1<br/>Generating text"]:::l1
    L1 --> L2["🏃 Level 2<br/>Making it fit"]:::l2
    L2 --> L3["🚀 Level 3<br/>Serving many users"]:::l3
    L3 --> L4["🦸 Level 4<br/>Scaling out"]:::l4
    L4 --> L5["🔷 Level 5<br/>Hands-on on Intel XPU"]:::l5
    classDef l0 fill:#1baf7a,stroke:#127a55,color:#ffffff
    classDef l1 fill:#2a78d6,stroke:#184f95,color:#ffffff
    classDef l2 fill:#eda100,stroke:#a56f00,color:#0b0b0b
    classDef l3 fill:#eb6834,stroke:#b0461d,color:#ffffff
    classDef l4 fill:#6250d6,stroke:#3d2fa0,color:#ffffff
    classDef l5 fill:#0b0b0b,stroke:#52514e,color:#ffffff
```

✅ published · 🔜 coming soon

---

## 🌱 Level 0 · Zero: how an LLM actually works

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 01 | 🔤 **[Tokens, Embeddings & Logits](chapters/01-tokens-embeddings-logits.md)** | How text becomes token IDs, IDs become vectors, and vectors become 128,256 scores for the next word | ✅ |
| 02 | 🧱 **[Inside a Transformer Block](chapters/02-inside-a-transformer-block.md)** | RMSNorm, attention, SwiGLU and the residual stream; where Llama 3 8B's 8 B parameters live, and why decode is memory-bound | ✅ |

## 🚶 Level 1 · Generating text

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 03 | 🎲 Sampling | Greedy, temperature, top-k, top-p, min-p, penalties, and SGLang's `sampling_params` | 🔜 |
| 04 | 🧠 **[KV Caching, Explained](chapters/04-kv-caching-explained.md)** | Why LLMs cache Keys & Values (not Queries), quadratic → linear attention, and the GPU-memory bill, checked on an Arc Pro B60 | ✅ |
| 05 | ⚖️ Prefill vs decode | Compute-bound vs memory-bound, TTFT vs TPOT, and why they need different kernels | 🔜 |

## 🏃 Level 2 · Making it fit

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 06 | 🎛️ GPU memory budget | Weights + KV pool + activations, and `--mem-fraction-static` explained | 🔜 |
| 07 | 🧮 Quantization | bf16 → fp8 / int4 weights, fp8 KV cache, and the accuracy trade-offs | 🔜 |
| 08 | 👥 Attention variants | MHA, GQA, MLA and sliding windows, and what each does to the KV cache | 🔜 |

## 🚀 Level 3 · Serving many users

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 09 | 📦 Continuous batching | From static batches to iteration-level scheduling | 🔜 |
| 10 | 📄 Paged KV memory | Fixed-size KV pages, the token pool, and fragmentation | 🔜 |
| 11 | 🌳 RadixAttention | How SGLang shares KV cache across requests with a common prefix | 🔜 |
| 12 | ✂️ Chunked prefill & scheduling | Keeping decode smooth while long prompts arrive | 🔜 |

## 🦸 Level 4 · Hero: scaling out

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 13 | 🔮 Speculative decoding | Draft-and-verify, EAGLE, and when it pays off | 🔜 |
| 14 | 🧩 Parallelism | Tensor, pipeline, data and expert parallelism | 🔜 |
| 15 | 🎯 Mixture-of-Experts | Routers, top-k experts, and the fused MoE kernels | 🔜 |
| 16 | 🔀 Prefill/decode disaggregation | Splitting the two phases across GPUs | 🔜 |

## 🔷 Level 5 · Hands-on: SGLang on Intel XPU

| # | Chapter | What you'll learn | |
|:-:|---|---|:-:|
| 17 | 🛠️ Running SGLang on Intel Arc | Install, launch, and serve Llama 3 8B on an Arc Pro B60, step by step | 🔜 |
| 18 | 📊 Benchmark & profile | `bench_serving`, reading throughput and latency, finding the bottleneck | 🔜 |

---

<p align="center">
<a href="chapters/04-kv-caching-explained.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decode-step-dark.svg">
  <img alt="One decode step with a KV cache" src="assets/decode-step-light.svg" width="680">
</picture>
</a>
</p>

---

<div align="center">

### ⭐ If this helped you, please star the repo. It helps others find it!

<a href="https://github.com/arathi-hlab/llm-inference-zero-to-hero"><img alt="Star this repo on GitHub" src="https://img.shields.io/badge/%E2%AD%90%20Star%20this%20repo-on%20GitHub-eda100?style=for-the-badge&logo=github&logoColor=white" height="42"></a>

</div>
