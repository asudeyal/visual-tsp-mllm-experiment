# Adaptive Visual Multi-Agent TSP (AVMA-TSP)

AVMA-TSP is a visual-only multi-agent research pipeline for TSP. The MLLM creates, critiques, scores, repairs and perturbs tours from rendered images. Python is the deterministic experiment/controller layer: it renders images, checks feasibility, detects structural stagnation, audits the Hybrid 2-opt claim and records objective metrics. Numeric objective information is never returned to the model.

## Architecture

```text
TSP problem image
      |
      v
Initializer
      |
      v
Current route image
      |
      v
Critic -> 3 candidate route images
      |
      v
Visual Scorer  (no pre-scorer validity filtering)
      |
      v
Selected route
      |
      v
Python feasibility audit
   |             |
 valid         invalid
   |             |
   |          Visual Repair (max 2)
   |             |
   +------> valid working route
                 |
                 v
        Structural stagnation?
          |             |
          no            yes
          |              |
        Critic       Hybrid Agent
                     one LLM 2-opt
                          |
                          v
                        Critic
                          |
                  stagnation again
                          |
                          v
                  Diversity Restart
```

### Information firewall

The model may receive only problem/route/candidate images, visible node labels and agent instructions. It must not receive coordinates, distance matrices, numerical edge/tour lengths, gaps, known optimums, GBest, textual current-route input, missing-node lists or validation reasons.

Python may calculate those values for **observer/audit purposes**. Distance or gap never selects a candidate or triggers an agent. Feasibility may trigger Repair; structural route repetition/similarity may trigger Hybrid/Restart.

## Frozen v1 behavior

- Critic: 3 independent candidate calls.
- Renderable invalid candidates are **not filtered** before Scorer.
- Unparseable or unrenderable outputs are retried because no candidate image can be produced from them.
- Scorer sees the original problem image plus every candidate image; candidate display order is shuffled deterministically.
- Selected invalid route -> strict visual Repair using only original problem + invalid route images.
- Repair: maximum 2 attempts; then Diversity Restart.
- Structural stagnation: 5-route window, exact-repeat signal and/or mean edge-set similarity >= 0.90.
- First structural stagnation -> Hybrid.
- Hybrid performs exactly one 2-opt **inside the LLM** and returns one route.
- Python audits whether the claimed Hybrid output is truly a single 2-opt but does not fix or replace it.
- A later structural stagnation after Hybrid -> Diversity Restart.
- Primary validation scope: TSPLIB `EUC_2D`.

## Protocol configs and providers

Protocol YAML files are reusable across instances and models:

```text
configs/
├── smoke_v1.yaml
├── pilot10_v1.yaml
├── pilot20_v1.yaml
├── main50_v1.yaml
└── main100_v1.yaml
```

YAML stores the experimental protocol. Provider, model and seed are run-specific CLI parameters. Included providers are `gemini`, `groq`, `mistral` and `openrouter`. The chosen model must support image input.

Environment variables:

```text
GEMINI_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
OPENROUTER_API_KEY=
```

Only the key for the active provider is required.

## Setup (PowerShell)

```powershell
cd .\adaptive_visual_tsp_experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

Place official TSPLIB EUC_2D instances under `data/tsplib/`.

## Shared multi-provider run

A new run gets a compact id such as:

```text
260827-eil_51_p10
```

The date, instance and protocol identify the shared experiment. Provider/model names live below that run instead of being repeated in the run id.

First model:

```powershell
python run_adaptive_multi_agent.py `
  --config configs/pilot10_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/tsplib/eil51.tsp `
  --reference-optimum 426
```

A second provider/model can use the same run id and therefore the exact same shared `problem_model.png`:

```powershell
python run_adaptive_multi_agent.py `
  --run-id 260827-eil_51_p10 `
  --config configs/pilot10_v1.yaml `
  --provider groq `
  --model <vision-model> `
  --instance data/tsplib/eil51.tsp `
  --reference-optimum 426
```

The runner rejects a reused run id if the instance, protocol config, prompts or seed differ. To make a replicate with different conditions, choose a different `--run-id`.

API-free validation uses the same shared run/input policy:

```powershell
python run_adaptive_multi_agent.py `
  --config configs/pilot10_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/tsplib/eil51.tsp `
  --reference-optimum 426 `
  --validate-only
```

Resume one provider/model:

```powershell
python run_adaptive_multi_agent.py `
  --run-id 260827-eil_51_p10 `
  --config configs/pilot10_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/tsplib/eil51.tsp `
  --reference-optimum 426 `
  --resume
```

## Outputs

New runs use:

```text
output/runs/260827-eil_51_p10/
├── run_manifest.json
├── inputs/
│   └── problem_model.png
└── providers/
    ├── gemini/
    │   └── gemini-3.6-flash/
    │       ├── run_manifest.json
    │       ├── initializer/
    │       ├── iterations/
    │       ├── checkpoint.json
    │       ├── summary.json
    │       └── analysis/
    └── <provider>/
        └── <model>/
```

`inputs/problem_model.png` is rendered once at the shared-run level. All compared models therefore receive the same physical problem image file. Route/candidate images remain provider/model-specific outputs but use the same frozen rendering protocol.

Historical single-model runs are not migrated or renamed and remain analyzable in their original layout.

## Analysis

For a shared run with one completed model, this is enough:

```powershell
python run_analysis.py --run-dir output/runs/260827-eil_51_p10
```

When multiple models exist, select one:

```powershell
python run_analysis.py `
  --run-dir output/runs/260827-eil_51_p10 `
  --provider gemini `
  --model gemini-3.6-flash
```

Each provider/model keeps exactly four analysis outputs:

```text
analysis/
├── summary.json
├── iterations.csv
├── selected_vs_oracle.png
└── analysis_report.txt
```

Historical single-model runs still work with their old direct path:

```powershell
python run_analysis.py --run-dir output/runs/<legacy-run-folder>
```

`observed_oracle_best` means the best valid distance Python happened to observe among generated candidates. It is **analysis-only** and never changes the model's search path.

## Baseline

```powershell
python run_baseline.py `
  --instance data/tsplib/<instance>.tsp `
  --reference-optimum <value> `
  --time-limit 2
```

## Prompt lifecycle

Prompts are versioned under `prompts/v1/`. Pilot prompt/parameter changes must create an explicit new prompt/config version. Once the main benchmark phase starts, prompts and experiment parameters should be frozen rather than tuned per benchmark instance.
