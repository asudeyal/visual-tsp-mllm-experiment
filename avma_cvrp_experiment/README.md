# Adaptive Visual Multi-Agent CVRP (AVMA-CVRP)

AVMA-CVRP is a visual-only multi-agent research pipeline for Capacitated Vehicle Routing Problems (CVRP). The MLLM creates, critiques, scores, repairs and perturbs routes from rendered images containing demand encodings. Python is the deterministic experiment/controller layer: it renders images with visual demand encodings (size, bar, dot density, or color), checks capacity and tour feasibility, detects structural stagnation, audits the Hybrid 2-opt claim and records objective metrics. Numeric objective information is never returned to the model.

## Architecture

```text
CVRP problem image (with demand encoding)
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
Python capacity & feasibility audit
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
Information firewall
The model may receive only problem/route/candidate images, visible node labels, demand encodings and agent instructions. It must not receive coordinates, distance matrices, numerical demands, vehicle capacities, numerical edge/tour lengths, gaps, known optimums, GBest, textual current-route input, missing-node lists or validation reasons.

Python may calculate those values for observer/audit purposes. Distance, demand, capacity, or gap never selects a candidate or triggers an agent. Feasibility or capacity violation may trigger Repair; structural route repetition/similarity may trigger Hybrid/Restart.

Frozen v1 behavior
Critic: 3 independent candidate calls.

Renderable invalid candidates (including capacity violations) are not filtered before Scorer.

Unparseable or unrenderable outputs are retried because no candidate image can be produced from them.

Scorer sees the original problem image plus every candidate image; candidate display order is shuffled deterministically.

Selected invalid route -> strict visual Repair using only original problem + invalid route images.

Repair: maximum 2 attempts; then Diversity Restart.

Structural stagnation: 5-route window, exact-repeat signal and/or mean edge-set similarity >= 0.90.

First structural stagnation -> Hybrid.

Hybrid performs exactly one 2-opt inside the LLM and returns one route.

Python audits whether the claimed Hybrid output is truly a single 2-opt but does not fix or replace it.

A later structural stagnation after Hybrid -> Diversity Restart.

Primary validation scope: CVRPLIB EUC_2D.

Protocol configs and providers
Protocol YAML files are reusable across instances and models:

Plaintext
configs/
├── cvrp_pilot10_size_v1.yaml
├── cvrp_pilot10_bar_v1.yaml
├── cvrp_pilot10_dot_density_v1.yaml
└── cvrp_pilot10_color_v1.yaml
YAML stores the experimental protocol and demand encoding method (size, bar, dot_density, or color). Provider, model and seed are run-specific CLI parameters. Included providers are gemini, groq, mistral and openrouter. The chosen model must support image input.

Environment variables:

Plaintext
GEMINI_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
OPENROUTER_API_KEY=
Only the key for the active provider is required.

Setup (PowerShell)
PowerShell
cd .\avma_cvrp_experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
Place official CVRPLIB EUC_2D instances under data/cvrplib/.

Shared multi-provider run
A new run gets a compact id such as:

Plaintext
260831-e-n_51-k_5_p10_size
The date, instance, encoding and protocol identify the shared experiment. Provider/model names live below that run instead of being repeated in the run id.

First model:

PowerShell
python run_adaptive_multi_agent.py `
  --config configs/cvrp_pilot10_size_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/cvrplib/E-n51-k5.vrp
A second provider/model can use the same run id and therefore the exact same shared problem.png:

PowerShell
python run_adaptive_multi_agent.py `
  --run-id 260831-e-n_51-k_5_p10_size `
  --config configs/cvrp_pilot10_size_v1.yaml `
  --provider groq `
  --model <vision-model> `
  --instance data/cvrplib/E-n51-k5.vrp
The runner rejects a reused run id if the instance, protocol config, prompts or seed differ. To make a replicate with different conditions, choose a different --run-id.

API-free validation uses the same shared run/input policy:

PowerShell
python run_adaptive_multi_agent.py `
  --config configs/cvrp_pilot10_size_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/cvrplib/E-n51-k5.vrp `
  --validate-only
Resume one provider/model:

PowerShell
python run_adaptive_multi_agent.py `
  --run-id 260831-e-n_51-k_5_p10_size `
  --config configs/cvrp_pilot10_size_v1.yaml `
  --provider gemini `
  --model gemini-3.6-flash `
  --instance data/cvrplib/E-n51-k5.vrp `
  --resume
Outputs
New runs use:

Plaintext
output/runs/260831-e-n_51-k_5_p10_size/
├── run_manifest.json
├── inputs/
│   └── problem.png
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
inputs/problem.png is rendered once at the shared-run level with the chosen demand encoding. All compared models therefore receive the same physical problem image file. Route/candidate images remain provider/model-specific outputs but use the same frozen rendering protocol.

Analysis
For a shared run with one completed model, this is enough:

PowerShell
python run_analysis.py --run-dir output/runs/260831-e-n_51-k_5_p10_size
When multiple models exist, select one:

PowerShell
python run_analysis.py `
  --run-dir output/runs/260831-e-n_51-k_5_p10_size `
  --provider gemini `
  --model gemini-3.6-flash
Each provider/model keeps exactly four analysis outputs:

Plaintext
analysis/
├── summary.json
├── iterations.csv
├── selected_vs_oracle.png
└── analysis_report.txt
observed_oracle_best means the best valid distance Python happened to observe among generated candidates. It is analysis-only and never changes the model's search path.

Baseline
PowerShell
python run_baseline.py `
  --instance data/cvrplib/<instance>.vrp
Prompt lifecycle
Prompts are versioned under prompts/cvrp_capacity_v1/. Pilot prompt/parameter changes must create an explicit new prompt/config version. Once the main benchmark phase starts, prompts and experiment parameters should be frozen rather than tuned per benchmark instance.