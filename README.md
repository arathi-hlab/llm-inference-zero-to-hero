<div align="center">

# 🚀 SGLang Guidance

**Clear, colourful explainers on LLM inference and serving with [SGLang](https://github.com/sgl-project/sglang), including on Intel GPUs.**

![articles](https://img.shields.io/badge/articles-1-2a78d6?style=for-the-badge)
![focus](https://img.shields.io/badge/focus-LLM%20serving-eb6834?style=for-the-badge)
![hardware](https://img.shields.io/badge/tested%20on-Intel%20Arc%20Pro%20B60-6250d6?style=for-the-badge)
![stars](https://img.shields.io/github/stars/arathi-hlab/sglang-guidance?style=for-the-badge&color=eda100)

✍️ by **Ashwini Rathi** · [@arathi-hlab](https://github.com/arathi-hlab)

</div>

---

## 📚 Articles

| | Article | What you'll learn |
|---|---|---|
| 🧠⚡ | **[KV Caching, Explained](articles/kv-caching-explained.md)** | Why LLMs cache Keys & Values (not Queries), how it turns quadratic attention into linear, and what it costs in GPU memory, with Llama 3 8B numbers checked on an Arc Pro B60 |

<p align="center">
<a href="articles/kv-caching-explained.md">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/decode-step-dark.svg">
  <img alt="One decode step with a KV cache" src="assets/decode-step-light.svg" width="680">
</picture>
</a>
</p>

---

## 🛣️ Coming next

- 🌳 RadixAttention: how SGLang shares KV cache across requests
- 🎛️ `--mem-fraction-static` and KV pool sizing, explained
- 🧮 KV cache quantization (fp8) and its trade-offs
- 🔷 Running SGLang on Intel XPU, step by step

---

<div align="center">

### ⭐ If this helped you, please star the repo. It helps others find it!

<a href="https://github.com/arathi-hlab/sglang-guidance"><img alt="Star this repo on GitHub" src="https://img.shields.io/badge/%E2%AD%90%20Star%20this%20repo-on%20GitHub-eda100?style=for-the-badge&logo=github&logoColor=white" height="42"></a>

</div>
