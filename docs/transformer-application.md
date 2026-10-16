# Complete autoregressive transformer application

A decode step consumes the new token plus every layer's previous keys and values. The shared tensor graph now carries those arrays explicitly: prefill creates them, each incremental decode concatenates one position, and the next call receives the resulting state. The full GPT-2 path includes token/position embeddings, both layer norms and residual connections, Q/K/V projection, causal QK-softmax-PV attention, MLP with tanh GELU, final normalization, tied vocabulary projection, and greedy token selection.

This is `transformer-graph/v1` over `pim-foundry.tensor-graph.v2`, not a paper simulator's timing-only transformer proxy. Functional execution consumes actual checkpoint weights and token IDs. The CPU reference path does not report hardware latency or energy. The shared platform path supplies separately labeled model predictions and explicit data-transfer costs.

## Historical pretrained validation

The observations below come from the local v0.8 development studies. Their run bundles are not distributed in this repository. The clean-checkout workflow below creates new assets and observations; it does not assume those historical directories exist.

- Public checkpoint: [distilbert/distilgpt2](https://huggingface.co/distilbert/distilgpt2), revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`; six layers, 768 hidden channels, 12 attention heads, vocabulary 50,257.
- Safetensors SHA-256: `e1ff18884359fe8beb795a5f414feb85a6ce3d929ad019c0d958c039d2b94a1b`. Loading uses the data-only safetensors parser and never executes repository code.
- Independent reference: Transformers 4.44.2, PyTorch 2.2.2+cpu, eager attention, float32, dropout disabled. Checkpoint loading checks missing/unexpected keys; legacy stored triangular attention masks are checked and replaced by the reference implementation's causal mask. The tied LM head is checked.
- Inputs: two equal-length 16-token prompts, four generated tokens per prompt; a distinct teacher-forcing evaluation pass over two sequences of 65 tokens.
- Evaluation dataset: [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, `wikitext-2-raw-v1`, test split. Selection is predetermined: join the first 32 rows with a newline and take the first 130 BPE tokens. The two 65-token sequences each contribute 64 next-token targets. Tokenization files, token selection and dataset shard are content-bound.

Observed NumPy 2.2.6 versus the independent reference:

| Check | Observed result |
|---|---:|
| Greedy token IDs | All 8 match |
| Generation logits maximum absolute error | 0.00008392334 |
| Teacher-forcing logits maximum absolute error | 0.00020599365 |
| Mean next-token NLL | 4.59892779617 nats/token |
| Reference NLL | 4.59892878034 nats/token |
| Absolute NLL difference | 0.00000098417 |
| Fixture perplexity | 99.3777053424 |

Every layer's K and V caches and the independently recomputed full-prefix logits are also compared. The application result records each check and fails quality if any numerical comparison or NLL-preservation threshold fails. The validation policy is absolute/relative logit tolerance 0.0003 and maximum NLL change 0.0001. The reference implementation itself differs between cached and fully recomputed logits by up to 0.00006103516 because reduction shapes differ.

**The perplexity above covers 128 targets from a fixed prefix. It is not full WikiText test perplexity or evidence of general model quality.** The checkpoint is an existing trained model; no synthetic/randomly initialized model is used to establish the reported task-quality result. Small seeded models are used only for invariant tests.

## State and shape contract

`GPT2Session.forward()` accepts a nonempty rank-2 integer token batch. The first call is prefill; subsequent calls append to the same equal-length batch. The positional offset comes from the stored cache length. Causal masking permits query position `past_length + i` to access only keys at or before that position. Sequence length cannot exceed the learned position embedding capacity. Batch size cannot change until `reset()`.

`generate()` always resets the session, executes prefill, selects the first token by vocabulary argmax, and performs `max_new_tokens - 1` incremental calls. The returned cache therefore contains the prompt and every generated token except the final token, which has not yet been consumed by the network. Reset discards prior K/V arrays and call records. Independent requests cannot inherit a previous request's state.

Tests verify cached decode against fresh full-prefix recomputation; no influence from future tokens; batched versus independent requests; reset isolation; context overflow without state mutation; invalid token IDs; configuration/weight coverage; greedy token selection; and the NLL shift/denominator using analytic cases. The application currently uses equal-length batches without padding, float32 source arithmetic/storage, learned GPT-2 positions and a tied output head. An explicit variant can surround selected linear matmuls with float16 casts and restore float32 before subsequent operators. Unsupported model architectures or attention variants fail explicitly.

## Execution and reproducibility

The independent reference uses PyTorch 2.2.2+cpu with NumPy 1.26.4, while the Foundry runtime uses NumPy 2.2.6. Keep them in separate environments so installing the reference dependencies does not change the runtime numerical environment. The historical reference was prepared on Linux with Python 3.12 and the versions below. These commands run from the repository root; the `.venv` runtime is also used by the main quickstart.

```sh
python3.12 -m venv .venv-reference
.venv-reference/bin/python -m pip install numpy==1.26.4 transformers==4.44.2 safetensors==0.8.0 pyarrow==19.0.1 msgpack==1.1.1
.venv-reference/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.2+cpu'

python3.12 -m venv .venv
.venv/bin/python -m pip install '.[tensor]'

# Explicitly retrieve pinned model/tokenizer/data assets and prepare the HF oracle.
OPENBLAS_NUM_THREADS=1 .venv-reference/bin/python scripts/evaluate_transformer_application.py \
  --assets runs/transformer-assets --download --prepare-only

# Execute the complete Foundry graph, quality checks, sealed result, and replay.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_transformer_application.py \
  --assets runs/transformer-assets --reuse-reference --out runs/transformer-run
```

The preparation script verifies pinned asset hashes, uses local-only model loading, and does not execute model-repository code. `--download` permits the explicit network step; omit it when the exact raw assets are already present. `--prepare-only` generates the NPZ weights, reference arrays, inputs, and dataset manifest without running the application. The later `--reuse-reference` execution needs only the runtime environment and those prepared assets. Downloads, converted weights, and source/asset snapshots require additional disk space. Each study output directory must be new.

To evaluate the same full application under an illustrative GPU analytical profile, first create the platform file. Its rates, 4 GiB capacity, and zero energy coefficients are model assumptions, not hardware measurements.

```sh
.venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
from pim_foundry.tensor_platform import default_platform
Path('runs/transformer-assets/model-platform.json').write_text(
    json.dumps(default_platform('gpu'), indent=2) + '\n')
PYTHON
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_transformer_application.py \
  --assets runs/transformer-assets --reuse-reference \
  --platform runs/transformer-assets/model-platform.json --device gpu \
  --out runs/transformer-platform-run
```

Every application bundle snapshots the model/configuration, token inputs, reference arrays, dataset manifest, and package source; replay checks output metrics, quality, and identities. The recorded dependency versions bound numerical interpretation; float32 BLAS reduction order may vary across environments, so numerical tolerance and exact replay identity are separate checks.

For modeled platform runs, weights and cache transfers are counted under the declared per-call cold execution policy. K/V values persist functionally in host memory between calls. Current modeled TTFT and mean inter-token latency include those transfers; they do not claim persistent device-resident KV or observed GPU/PIM execution. Mapping float32 GPT-2 directly to a backend supporting only int8 or float16 is rejected. The supplied `fp16_matmul_nodes` variant inserts explicit float32→float16 input casts, a float16-storage matmul, and float16→float32 output conversion. This variant is checked against the original reference with a declared quality tolerance.


## Complete application through native HBM

The historical `runs/transformer-native-hbm-v08-release` study was a second, separate acceptance test; that bundle is not included in the public checkout. The public `sshleifer/tiny-gpt2` Flax checkpoint at revision `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be` is parsed as data-only MessagePack; no pickle or repository code is executed. Its SHA-256 is `3cd4987249615ca870f2c18e657a62f0962aecad0eac790dd89362227310fd30`. Flax Conv1D kernels are transposed according to Transformers 4.44.2's `FlaxConv1D` implementation, and complete PyTorch checkpoint coverage is checked before independent reference execution.

The tiny model has two layers and two hidden channels. It is a functional fixture; its provenance and size do not establish meaningful language quality. The trained DistilGPT2 experiment above remains the substantive model-quality preservation check.

The tiny fixture runs a four-token prefill, one incremental decode, and a separate 17-token teacher-forcing pass. Eight QKV/attention-output/MLP linear nodes per forward are explicitly converted to float16 and mapped to the HBM backend. Other operations retain float32 and use CPU functional execution with an analytical GPU profile. Each application run executes 32 native HBM simulator tiles, including the split needed for the 17-position evaluation pass. Actual intermediate tensor values cross between these backends; the final output is not replaced with the reference.

The native mapping and an all-GPU analytical baseline execute the same explicitly cast graph on the same declared platform, with the same model, data, precision policy and accounting contract. Both have sealed replays and pass the independent reference checks. Observed native results: both greedy tokens match; evaluation-logit maximum absolute error 0.0000038147; NLL absolute difference 0.000000094697. The chosen tolerance (0.002 absolute/relative logits, 0.001 NLL change) is declared in the manifest before execution.

The conditional generation-latency model ratio is **1.01403 native mapping / analytical GPU mapping**: this fixture does not show a modeled speedup. It does not measure hardware performance. HBM command timing retains the patched DCC runtime's explicitly documented completion assumption; the GPU cost profile is uncalibrated. The metric includes per-call transfers, functional KV state lives on the host between calls, and greedy argmax is an executed/shared-IR node included in the model schedule.

The native script requires a separately built HBM runtime at the example platform's expected `runs/dcc-patched-final/runtime.json` path. See [native runtime prerequisites](native-quickstart.md#建置原生-runtimes). It currently has no runtime-path override and no `--prepare-only` option: its first invocation prepares the tiny-model reference **and executes the native study**. Do this only after obtaining the source assets above and building the native runtime.

```sh
# Prepare pinned tiny-model assets/reference, then execute in the reference environment.
OPENBLAS_NUM_THREADS=1 .venv-reference/bin/python scripts/evaluate_transformer_native.py \
  --assets runs/transformer-tiny-assets --source-assets runs/transformer-assets \
  --download --out runs/transformer-native-reference-env

# Reuse those assets for a new native study under the NumPy 2.2.6 runtime.
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_transformer_native.py \
  --assets runs/transformer-tiny-assets --source-assets runs/transformer-assets \
  --reuse-reference --out runs/transformer-native-run
```

A negative control in `runs/transformer-quality-negative-v08` changes one reference greedy-token ID while preserving model/input bytes. The application returns `quality_failed`, and the bundle remains verifiable. This confirms that a claimed reference answer is tested, not automatically accepted.

The final release studies use NumPy 2.2.6: `runs/transformer-application-v08-release`, `runs/transformer-platform-v08-release`, and `runs/transformer-native-hbm-v08-release`. Native platform identity binds the runtime-descriptor content hash and is checked for consistency across every prefill, decode and evaluation call. Unknown energy is omitted with an explicit diagnostic rather than converted to zero. The native acceptance script supports `--download` for immutable public tiny-model assets.
