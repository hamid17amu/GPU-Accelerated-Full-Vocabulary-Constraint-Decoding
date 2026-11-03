# GPU-Accelerated Full-Vocabulary Constraint Decoding — starter code

Starter code for the project described in `GPU_Project_pirapu.pdf`.

What ships here is the **baseline you are asked to beat**: symbol table
creation and constraint decoding **iteration 1**, enforced with **lazy
widening** (check the Top-64 tokens, widen to Top-1024 only if too few
survive). The full-vocabulary CPU and CUDA versions are tasks 2–4; they are
deliberately not implemented.

> Later iterations of the research pipeline (failure-adaptive constraints
> learned from compile and test feedback) are unpublished and not included.

## Contents

| Path | What it is |
|---|---|
| `care_constraints.py` | Symbol table loading, site extraction, static constraint checking, lazy-widening `LogitsProcessor` |
| `run_decode.py` | Minimal driver: FIM prompt → constrained decode → exact-match score |
| `java/` | `GroundedSymbolTable.java` — builds a symbol table for one hole (JavaParser + symbol solver) |
| `symbol_cache/` | Pre-computed symbol tables for all 166 bugs (`Chart_1.v2.json`, …) |
| `static_context_cache/` | Pre-computed class-skeleton context blocks used in the prompt |
| `data/defects4j_single_line_with_idx_augmented.jsonl` | 166 single-line Defects4J bugs with the buggy/fixed function and the FIM split |
| `lora_full_function_skeleton_4096/` | Fine-tuned LoRA adapter for CodeLlama-7b-Instruct (r=8, q_proj/v_proj, 4096-token window) |

You do **not** need Defects4J or `javac` to run the decoder: the symbol tables
and context blocks are pre-computed. You need them only to regenerate a table
or to measure plausible patches (task 5).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The adapter is layered on `codellama/CodeLlama-7b-Instruct-hf`, which is downloaded
from the Hugging Face Hub on first use (~13 GB; set `HF_HOME` to point at a
scratch disk). Pass `--base-model /path/to/local/CodeLlama-7b-Instruct-hf` if you
already have it. Inference in bf16 needs about 15 GB of GPU memory.

## Running

Inspect a prompt and its symbol table — no model, no GPU:

```bash
python run_decode.py --bug Chart_1 --dry-run
```

Unconstrained baseline:

```bash
python run_decode.py --limit 10 --constraints none --num-beams 10 --num-candidates 10
```

Constrained decoding with lazy widening:

```bash
python run_decode.py --limit 10 --constraints static --num-beams 10 --num-candidates 10 \
    --report-stats --out results_lazy.json
```

`--report-stats` prints, per bug, how many decoding steps ran, how many had to
widen from Top-64 to Top-1024, how many failed open because nothing survived,
how many candidate tokens were checked, and how long the checking took. Those
counters are your CPU reference numbers for the evaluation in task 5.

Useful flags: `--top-k` / `--wide-k` (the two widening stages, default 64 and
1024), `--num-beams`, `--temperature` (0 = greedy), `--max-new-tokens`,
`--no-context` to drop the class skeleton from the prompt.

## Where your code goes

`care_constraints.py` has two hooks:

- **`RowFilter.allowed(gen_ids, cand_ids)`** — the hot loop. One call per beam
  per decoding step; it runs `StaticConstraints.check` once per candidate
  token. This is the function to move to the GPU.
- **`make_logits_processor(...)`** — the per-row widening policy. Keep this one
  as the baseline and add a full-vocabulary variant beside it, so a run can
  select the strategy and the evaluation can compare them.

For task 1 (how often is a valid token inside Top-64 / Top-1024 / beyond), note
that `allowed()` already receives the candidate ids in descending score order,
so the rank of the first surviving token is its index in that list. A token
beyond Top-1024 is only visible once you check the whole vocabulary — which is
the point of the task.

## Regenerating a symbol table

Only needed to add a bug or change the extractor's output.

```bash
cd java && mvn -q package        # -> target/symbol-table.jar
cd ..

defects4j checkout -p Chart -v 1b -w /tmp/Chart_1b
java -jar java/target/symbol-table.jar \
    --file /tmp/Chart_1b/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java \
    --line 1564 \
    --source-root /tmp/Chart_1b/source \
    --classpath "$(defects4j export -p cp.compile -w /tmp/Chart_1b)" \
    --out symbol_cache/Chart_1.v2.json --pretty
```

`--line` is 1-based and refers to the file **as checked out**, not the line
number in the dataset. `care_constraints.hole_line_number()` computes it from
the hole's position inside the buggy function. Omitting `--classpath` still
works but resolves fewer receivers, so the constraints are weaker.

## Notes on the constraint semantics

- **Fail open.** A position is constrained only when the symbol table resolves
  it. An unknown receiver, a non-enum switch, or a file with wildcard imports
  (for `new`) decodes freely. A constraint therefore never removes a patch that
  `javac` would have accepted.
- **Partial names stay alive.** A name still being generated only has to be a
  *prefix* of some legal member, so `calendar.getT` survives and
  `calendar.getZ` does not.
- **String literals and comments are blanked** before sites are extracted, so
  text inside them is never constrained.
