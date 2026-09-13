# Papers

Reference PDFs, committed so the version we worked from stays pinned. arXiv
papers get revised and the numbers move between versions, so the arXiv ID alone
is not enough to reproduce a claim.

Files are named `<arxiv-id>-<short-name>.pdf`.

## 2408.11796 — LLM Pruning and Distillation in Practice: The Minitron Approach

[`2408.11796-minitron-approach.pdf`](2408.11796-minitron-approach.pdf)

Sreenivas, Muralidharan, Joshi, Chochowski, Patwary, Molchanov, Shoeybi, Kautz,
Mahabaleshwarkar, Shen, Zeng, Kuchaiev, Chen, Suhara, Diao, Yu, Chen, Ross,
Korzekwa, Olabiyi, Aithal, Catanzaro. NVIDIA.

| | |
| --- | --- |
| Version | v3, dated 2024-12-03 |
| Pages | 11 |
| arXiv | https://arxiv.org/abs/2408.11796 |
| PDF source | https://arxiv.org/pdf/2408.11796v3 |
| Retrieved | 2026-09-12 |
| sha256 | `cbd33b05d941146cc86e4a5e83ec35a433388419165b80c1f534ad868cde7024` |

What it covers: structured pruning plus knowledge distillation to compress
Mistral NeMo 12B to 8B and Llama 3.1 8B to 4B, without access to the original
pretraining data. Introduces a teacher correction step, a light fine-tune of the
teacher on the target data distribution, before distillation. Compares depth
pruning against joint hidden/attention/MLP width pruning on the LM Evaluation
Harness.

This is the paper bonsai works from.

## Checking a file

```bash
sha256sum -c <<'SUMS'
cbd33b05d941146cc86e4a5e83ec35a433388419165b80c1f534ad868cde7024  2408.11796-minitron-approach.pdf
SUMS
```
