# Adaptive Visual Multi-Agent CVRP (AVMA-CVRP)

AVMA-CVRP is a visual-only multi-agent research pipeline for Capacitated Vehicle Routing Problems (CVRP). The multimodal model constructs, critiques, scores, repairs, and perturbs routes from rendered images. Python is the deterministic controller and observer layer: it renders the visual problem, checks feasibility and capacity, detects structural stagnation, audits Hybrid 2-opt behavior, and records metrics without exposing hidden numeric problem data to the model.

## Research question

The main experiment compares **4 visual demand encodings × 2 placements**:

| Encoding | Collision/map placement | Side-panel placement |
|---|---|---|
| Size | `size_collision` | `size_sidepanel` |
| Bar | `bar_collision` | `bar_sidepanel` |
| Dot density | `dotdensity_collision` | `dotdensity_sidepanel` |
| Color | `color_collision` | `color_sidepanel` |

The side-panel condition changes the **placement** of the demand encoding, not its semantic meaning.

Final visual-condition configs are under:

```text
configs/main_8method/
├── size_collision.yaml
├── size_sidepanel.yaml
├── bar_collision.yaml
├── bar_sidepanel.yaml
├── dotdensity_collision.yaml
├── dotdensity_sidepanel.yaml
├── color_collision.yaml
└── color_sidepanel.yaml
```

## Information firewall

The model may use only information visible in the provided images and the agent instructions, including:

- visible customer positions and node IDs,
- the visually marked depot,
- visible route connections,
- visual demand encodings,
- the visual empty/full capacity reference,
- and the visibly displayed vehicle count.

The model must not receive hidden numeric problem information such as:

- coordinates,
- distance matrices,
- numerical demands,
- numerical vehicle capacity or route loads,
- numerical edge or route lengths,
- known optimums or optimal routes,
- optimality gaps,
- GBest,
- textual current-route input,
- missing-node lists,
- or validation reasons.

Python may compute these values for validation and analysis, but they are not returned to the model.

## Multi-agent protocol

```text
Problem image
    |
    v
Initializer
    |
    v
Current route image
    |
    v
Critic -> 3 independent candidate calls
    |
    v
Candidate route images
    |
    v
Visual Scorer
(no pre-scorer validity filtering)
    |
    v
Selected route
    |
    v
Python feasibility/capacity audit
   |                    |
 valid                invalid
   |                    |
   |             Visual Repair (max 2)
   |                    |
   +------------> valid working route
                        |
                        v
              structural stagnation?
                 |             |
                 no            yes
                 |              |
               Critic       Hybrid
                            one LLM 2-opt
                                 |
                                 v
                               Critic
                                 |
                       later stagnation
                                 |
                                 v
                         Diversity Restart
```

Frozen protocol behavior:

- Critic produces **3 independent candidates**.
- Renderable invalid candidates are still shown to the Scorer.
- Unparseable or unrenderable outputs may be retried because no candidate image can be produced.
- Selected invalid routes go to visual Repair.
- Repair uses at most **2 attempts**, then falls back to Diversity Restart.
- Structural stagnation uses a **5-route window** with exact repetition and/or mean edge-set similarity `>= 0.90`.
- First stagnation triggers Hybrid.
- Hybrid performs exactly one intra-route LLM 2-opt; Python audits but does not repair the claim.
- Later stagnation triggers Diversity Restart.
- Primary benchmark scope is CVRPLIB `EUC_2D`.

## Prompt versions

Prompt sets are versioned independently from the visual condition:

```text
prompts/
├── cvrp_capacity_v1/
├── cvrp_capacity_v2/
└── cvrp_capacity_v3/
```

All main configs default to:

```text
cvrp_capacity_v3
```

A different prompt version can be selected without changing the visual config:

```powershell
--prompt-set cvrp_capacity_v1
--prompt-set cvrp_capacity_v2
--prompt-set cvrp_capacity_v3
```

The effective prompt version and prompt hashes are recorded in run provenance.

## Versioned scale policies

Image workspace scale is also independent from the visual condition and prompt version.

Default policy:

```text
data/cvrplib/scale_policies/benchmark_scale_v1.json
```

Frozen V1 values:

| Instance | Workspace scale |
|---|---:|
| `P-n21-k2.vrp` | `1.000x` |
| `A-n37-k5.vrp` | `1.000x` |
| `E-n51-k5.vrp` | `1.000x` |
| `X-n110-k13.vrp` | `2.092x` |
| `X-n204-k19.vrp` | `3.000x` |
| `X-n298-k31.vrp` | `3.000x` |
| `X-n393-k38.vrp` | `3.000x` |

The same instance scale is applied across:

- all 8 visual conditions,
- all prompt versions,
- the shared problem image,
- route images,
- Critic candidates,
- Scorer images,
- Repair images,
- and Hybrid images.

Future scale policies can be added without changing code, for example:

```text
data/cvrplib/scale_policies/benchmark_scale_v2.json
```

and selected with:

```powershell
--scale-policy data/cvrplib/scale_policies/benchmark_scale_v2.json
```

A missing instance entry is treated as an error; there is no silent fallback to `1x`.

## Current benchmark instances

```text
data/cvrplib/
├── P-n21-k2.vrp
├── A-n37-k5.vrp
├── E-n51-k5.vrp
├── X-n110-k13.vrp
├── X-n204-k19.vrp
├── X-n298-k31.vrp
├── X-n393-k38.vrp
└── scale_policies/
    └── benchmark_scale_v1.json
```

The 500-customer level is currently held out.

## Setup

PowerShell:

```powershell
cd .\avma_cvrp_experiment

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pytest -q
```

Only the API key for the selected provider is required in `.env`.

Example:

```text
GEMINI_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
OPENROUTER_API_KEY=
```

## API-free validation

Validate the full config / prompt / instance / render path without making an API call:

```powershell
python run_adaptive_multi_agent.py `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --max-vehicles 2 `
  --validate-only
```

Use V2 prompts with the exact same visual condition:

```powershell
python run_adaptive_multi_agent.py `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --prompt-set cvrp_capacity_v2 `
  --provider gemini `
  --model gemini-3.6-flash `
  --max-vehicles 2 `
  --validate-only
```

## Main run

Default prompt set: `cvrp_capacity_v3`
Default scale policy: `benchmark_scale_v1`

Example:

```powershell
python run_adaptive_multi_agent.py `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --max-vehicles 2
```

For reproducible named runs, use `--run-id`.

Resume an interrupted provider/model run with the same frozen conditions:

```powershell
python run_adaptive_multi_agent.py `
  --run-id <existing-run-id> `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --max-vehicles 2 `
  --resume
```

The runner rejects incompatible reuse of a run when frozen experiment conditions change.

## Outputs

New active benchmark runs are written under:

```text
output/runs/<run-id>/
```

The shared run records:

- `run.json` provenance,
- the model-facing `problem.png`,
- prompt/config/problem hashes and metadata,
- effective render policy,
- scale-policy name/hash,
- and effective pixel dimensions.

Provider/model-specific state, traces, route images, and analysis artifacts live below the shared run directory.

Historical pre-main runs can be kept separately under:

```text
output/archive_runs/
```

Classical solver outputs can be kept under:

```text
output/baseline/
```

## Analysis

Generate the compact run report with:

```powershell
python run_analysis.py --run-dir output/runs/<run-id>
```

When a shared run contains multiple provider/model results:

```powershell
python run_analysis.py `
  --run-dir output/runs/<run-id> `
  --provider gemini `
  --model gemini-3.6-flash
```

Primary reported metrics include:

- initializer first-shot feasibility,
- capacity violation count,
- capacity excess severity,
- repair rate / attempts / success,
- valid Critic candidate rate,
- final valid rate,
- Scorer oracle agreement,
- selection regret,
- valid-only distance / gap / crossings,
- tokens,
- latency,
- calls,
- and estimated cost.

`observed_oracle_best` is analysis-only: it means the best valid objective value Python happened to observe among generated candidates. It never changes the model's search path.

## Baseline

A deterministic classical CVRP baseline can be run with:

```powershell
python run_baseline.py `
  --instance data/cvrplib/<instance>.vrp
```

## Experimental freeze

For a main benchmark batch, freeze together:

- visual-condition configs,
- prompt version,
- scale-policy version,
- model,
- media resolution,
- repair / Critic counts,
- restart and stagnation policy,
- seed / replicate policy,
- and benchmark instance set.

New prompt or scale experiments should create a new version instead of modifying an already-used frozen version.
