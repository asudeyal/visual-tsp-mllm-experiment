# AVMA-CVRP

Implementation and experimental artifacts for the manuscript:

**AVMA-CVRP: An Adaptive Visual Multi-Agent System for the Capacitated Vehicle Routing Problem**

Authors: **Nimet Asude Yalçın, Ayşe Nihal Kaymak, and Furkan Yener**

AVMA-CVRP is a visual-only multi-agent framework for the Capacitated Vehicle Routing Problem (CVRP). The multimodal model operates on rendered problem and route images, while a deterministic Python observer performs feasibility checks, objective evaluation, structural-stagnation detection, logging, and analysis without exposing hidden numerical problem data to the model.

## Manuscript experiment

The final manuscript evaluates **4 visual demand encodings × 2 layouts**, giving **8 visual conditions**:

| Encoding | Collision-Aware | Side-Panel |
|---|---|---|
| Size | `size_collision` | `size_sidepanel` |
| Bar | `bar_collision` | `bar_sidepanel` |
| Dot Density | `dotdensity_collision` | `dotdensity_sidepanel` |
| Color | `color_collision` | `color_sidepanel` |

The visual-condition configurations are stored under:

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

The side-panel condition changes the spatial placement of the demand representation, not its semantic meaning.

## Benchmark instances used in the manuscript

| Instance | Customers | Vehicles | Capacity | BKS |
|---|---:|---:|---:|---:|
| `P-n16-k8` | 15 | 8 | 35 | 450 |
| `P-n19-k2` | 18 | 2 | 160 | 212 |
| `P-n21-k2` | 20 | 2 | 160 | 211 |
| `E-n23-k3` | 22 | 3 | 4500 | 569 |
| `A-n32-k5` | 31 | 5 | 100 | 784 |
| `B-n38-k6` | 37 | 6 | 100 | 805 |
| `P-n50-k8` | 49 | 8 | 120 | 631 |
| `E-n76-k7` | 75 | 7 | 220 | 682 |

The corresponding CVRPLIB instance files are available under `data/cvrplib/`.

Additional CVRPLIB files in that directory were used during development and are not part of the final manuscript benchmark set.

## Experimental artifacts

All **64 runs reported in the manuscript** are stored under:

```text
output/runs/
```

The final experiment contains:

- 8 benchmark instances,
- 8 visual conditions per instance,
- 1 recorded run per instance-condition pair,
- for a total of **64 manuscript runs**.

Run IDs follow the pattern:

```text
main-v3-<instance>-<encoding>-<layout>-r01
```

Examples:

```text
main-v3-a32-bar-collision-r01
main-v3-a32-bar-sidepanel-r01
main-v3-p21-size-collision-r01
main-v3-p19-bar-sidepanel-r01
```

Each run directory contains the core model-facing problem image, provenance metadata, route/candidate images, and state and trace records. Derived analysis artifacts are included where generated and can be regenerated with `run_analysis.py`.

Historical pilot and development runs that are **not part of the final manuscript results** are stored separately under:

```text
output/archive_runs/
```

Classical solver outputs, when present, are stored under:

```text
output/baseline/
```

## Reproducibility settings

The manuscript experiments use the following frozen settings:

| Setting | Value |
|---|---|
| Model | `gemini-3.7-flash` |
| Prompt set | `cvrp_capacity_v3` |
| Random seed | `42` |
| Critic candidates per iteration | `3` |
| Maximum direct Repair attempts | `2` |
| Maximum Diversity Restart attempts | `3` |
| Structural-stagnation window | `5` |
| Mean edge-set similarity threshold | `0.90` |
| Maximum unique routes in stagnation window | `2` |
| Media resolution | `high` |

The effective prompt text, prompt hashes, configuration hash, instance hash, render policy, and run metadata are recorded in each run's provenance.

The manuscript runs were executed with the required CLI override `--model gemini-3.7-flash`. The model recorded in each provider's `state.json` and `trace.jsonl` is authoritative for the executed API calls, even where a YAML configuration retains a different model value.

The base YAML configurations specify 10 iterations. Runs extended beyond that base horizon were continued with `--resume --iterations <target>`. The executed target and completed iteration counts are recorded in each run's `state.json`.

## Information Firewall

The methodological design separates information available to the MLLM from numerical information used by the deterministic observer.

The model may use only visible information and role instructions, including:

- visible customer positions and node IDs,
- the depot marker,
- visible route connections,
- visual demand encodings,
- the visual full-capacity reference,
- and the visibly displayed vehicle count.

The model is not given hidden numerical problem information such as:

- exact coordinates,
- distance matrices,
- numerical customer demands,
- numerical vehicle capacity or route loads,
- numerical edge or route lengths,
- BKS/optimum or optimal routes,
- optimality gaps,
- observer GBest,
- missing-node lists,
- or validation/failure reasons.

The Python observer may compute these values for feasibility validation, objective calculation, adaptive control, and analysis, but they are not returned to the model.

## Multi-agent protocol

The final framework uses the following roles:

1. **Initializer Agent** — constructs the first complete route set from the problem image.
2. **Critic Agent** — generates three independent candidate route sets per iteration.
3. **Visual Scorer** — ranks the candidate images using only visible evidence.
4. **Repair Agent** — attempts to restore feasibility without receiving the numerical failure reason.
5. **Hybrid Agent** — performs one LLM-guided intra-route 2-opt move after the first structural-stagnation event.
6. **Diversity Restart Agent** — generates a fresh route set after repeated stagnation or exhausted repair/recovery paths.

Important protocol rules:

- Critic produces **3 independent candidates** per iteration.
- Renderable candidates are shown to the Visual Scorer without numerical feasibility pre-filtering.
- Selected invalid routes may receive at most **2 Repair attempts**.
- Structural stagnation is evaluated over the last **5** working route sets.
- Stagnation is triggered when the number of unique canonical route sets is at most **2**, or the mean consecutive edge-set similarity is at least **0.90**.
- The first stagnation event triggers Hybrid.
- A later stagnation event in the same search phase triggers Diversity Restart.
- Diversity Restart uses at most **3 attempts** before the incumbent is retained when available.

## Prompt versions

Prompt sets are versioned under:

```text
prompts/
├── cvrp_capacity_v1/
├── cvrp_capacity_v2/
└── cvrp_capacity_v3/
```

The manuscript experiments use:

```text
cvrp_capacity_v3
```

Earlier prompt versions are retained for development history and are not the prompt set reported in the final manuscript experiments.

## Repository structure

```text
avma_cvrp_experiment/
├── README.md
├── requirements.txt
├── run_adaptive_multi_agent.py
├── run_analysis.py
├── run_baseline.py
├── run_refinement_analysis.py
├── configs/
│   └── main_8method/
├── data/
│   └── cvrplib/
├── prompts/
│   ├── cvrp_capacity_v1/
│   ├── cvrp_capacity_v2/
│   └── cvrp_capacity_v3/
├── src/
├── tests/
└── output/
    ├── runs/          # all final manuscript runs
    ├── archive_runs/  # pilot/development runs
    └── baseline/      # classical baseline outputs
```

## Setup

The experimental pipeline is Python-based. Dependency requirements are defined in `requirements.txt`.

PowerShell example:

```powershell
cd .\avma_cvrp_experiment

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pytest -q
```

For Gemini execution, define the API key locally, for example in an untracked `.env` file:

```text
GEMINI_API_KEY=
```

Do not commit API keys or other credentials.

## API-free validation

The full configuration, prompt, instance, and rendering path can be validated without making an API request:

```powershell
python run_adaptive_multi_agent.py `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --provider gemini `
  --model gemini-3.7-flash `
  --max-vehicles 2 `
  --reference-optimum 211 `
  --validate-only
```

## Example manuscript-style run

```powershell
python run_adaptive_multi_agent.py `
  --instance data/cvrplib/P-n21-k2.vrp `
  --config configs/main_8method/bar_collision.yaml `
  --provider gemini `
  --model gemini-3.7-flash `
  --max-vehicles 2 `
  --reference-optimum 211 `
  --run-id main-v3-p21-bar-collision-r01
```

For named runs, `--run-id` should match the frozen experiment naming convention.

## Analysis

Generate the per-run analysis report with:

```powershell
python run_analysis.py `
  --run-id main-v3-p21-bar-collision-r01 `
  --provider gemini `
  --model gemini-3.7-flash
```

Analysis artifacts are written below the corresponding provider/model run directory, typically including:

```text
analysis/
├── report.txt
└── search_progress.png
```

The analysis layer reports feasibility, objective values, BKS gaps, Observer GBest, Selected GBest, selection regret, structural metrics, recovery events, API calls, token usage, and latency.

Observer-side metrics never alter the model's search decisions.

## Development files

Some source files, benchmark instances, prompt versions, archived runs, and experimental utilities are retained to preserve the development history of the project.

For the final manuscript, the authoritative experimental artifacts are:

- `configs/main_8method/`
- `prompts/cvrp_capacity_v3/`
- the 8 benchmark instances listed above,
- and all 64 run directories under `output/runs/`.

## Citation

A manuscript-specific release will be created before submission so that the exact code and artifact state used for the paper can be referenced independently of future changes to `main`.

Planned release tag:

```text
v1.0-mdpi-submission
```

Until that release is created, use the repository path for development access.
