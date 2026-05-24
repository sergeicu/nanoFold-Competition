# nanoFold: A Protein Folding Slowrun

nanoFold is a data-efficiency competition for protein structure prediction. It is inspired by the nanoGPT slowrun: everyone trains under a fixed budget, and the leaderboard rewards models that learn more structure from the same amount of data.

The core bet is simple: biological data is expensive. Text and image models often improve by consuming more data, but protein structure data is far more constrained, far harder to generate, and far more sensitive to leakage. If we want better biological foundation models, we need architectures and training methods that make stronger use of the data we already have.

This repo turns that idea into a benchmark. Participants get the same official train set, the same sample budget, and the same hidden evaluation path. Progress should come from better biological priors, better inductive biases, better objectives, better curricula, and better optimization under scarcity.

Note - we are training for 'limited' track only. Always run with `--limited` flag. 

Your TODO task is clearly defined here - at the bottom of this file - under '# Your task' 

## How The Slowrun Works

The `limited` track trains for fixed sample budgets. The hidden leaderboard evaluates multiple checkpoints and rank by `foldscore_auc_hidden`: trapezoidal area under hidden FoldScore versus cumulative samples seen.

That makes early learning matter. A model that gets useful structure after 2,000 samples should beat a model that only wakes up at the end, even if their final scores are close. This is the pressure that should surface architectures with better biological priors.

FoldScore is a CASP15-inspired raw score over the CASP metrics that can be
computed reproducibly from the official atom14 output contract and official
residue identities:

```text
FoldScore =
  0.25*GDT_HA-Ca
+ 0.09375*(lDDT-all-atom14 + CADaa-atom14 + SG-atom14 + SC-atom14)
+ 0.125*(MolProbity-clash-atom14 + BB-atom14 + DipDiff-atom14)
```

`GDT_HA-Ca` measures global C-alpha fold accuracy with threshold-specific GDT
superpositions. The remaining terms measure local all-atom agreement, contact
preservation, SphereGrinder local motifs, chi1/chi2 side-chain geometry,
atom-name-aware heavy-atom clashes, phi/psi/omega backbone geometry, and
three-residue C-alpha/O distance windows. The weights are the CASP15 formula
weights renormalized over the structure-derived components supported by
nanoFold. `ASE` requires submitted confidence estimates and `reLLG_lddt`
requires a crystallographic molecular-replacement scoring path, so they are not
part of the official rank score.

The final hidden FoldScore is the tie-breaker for fixed-budget tracks and the primary rank metric for `unlimited`. Public validation exists for debugging, not ranking.

## What This Repo Provides

- official track policy in `tracks/limited.yaml`
- minAlphaFold2-derived preprocessing for A3M/mmCIF alignment, atom14 labels, residue constants, and template plumbing
- sealed prediction/scoring entrypoints that keep hidden labels away from submission code
- a strict submission API with `build_model`, `build_optimizer`, and `run_batch`
- dataset fingerprints and manifest checks so official data changes are visible
- an optional Modal GPU runner with local-disk data staging for public-data training
- a pinned minAlphaFold2 tiny reference submission plus a template submission that pass the official atom14 contract

## Tracks At A Glance

Source of truth: `tracks/*.yaml`. All tracks use the same official public train set, public validation set, sealed hidden validation set, CASP15-inspired FoldScore, and no-template policy. They differ in how much training budget is allowed and what scientific question the leaderboard is meant to answer.

| Track | Purpose | Fixed training budget | Rank metric | Use this when | Submit with |
|---|---|---:|---|---|---|
| `limited` | Main accessible slowrun leaderboard | `20,000` samples (`10,000` steps x effective batch `2`) | `foldscore_auc_hidden` | you want the primary competition result under a small, reproducible budget | `--track limited` |
| `research_large` | Larger fixed-data research leaderboard | `100,000` samples (`50,000` steps x effective batch `2`) | `foldscore_auc_hidden` | you want to study whether an approach still wins with more optimization while using the same data | `--track research_large` |
| `unlimited` | Open-ended fixed-data research leaderboard | unrestricted training budget and model size | `final_hidden_foldscore` | you want the best final hidden structure quality while keeping the hidden set sealed and the public data contract fixed | `--track unlimited` |

For all three tracks, set `track: <track_id>` in `submissions/<name>/config.yaml`, validate with `python scripts/validate_submission.py --submission submissions/<name> --track <track_id> --strict`, and open a submission PR naming the intended track. Submit separate configs or separate submission directories when one method targets multiple tracks. Maintainers create accepted leaderboard entries after sealed hidden evaluation; participant PRs should not edit leaderboard artifacts.

Leaderboard entries include a linked `Name` column pointing to the accepted
submission directory and a `Team` column for identity. Use a lab, company,
project team, or individual researcher name consistently across submissions. If
`--team` is omitted during a PR-triggered GitHub Actions run, nanoFold falls
back to the PR author's GitHub username. Manual maintainer automation can set
`NANOFOLD_PR_AUTHOR=<github-username>` for the same fallback.

The `limited` constants are:

| Item | Value |
|---|---:|
| Train chains | `10,000` |
| Public val chains | `1,000` |
| Hidden val chains | `1,000` |
| Seed | `0` |
| Crop size | `256` |
| MSA depth | `192` |
| Effective batch size | `2` |
| Max steps | `10,000` |
| Sample budget | `20,000` |
| Residue budget | `5,120,000` |
| Parameter cap | `100,000,000` trainable parameters |
| Tie-breaker | `final_hidden_foldscore` |

## Split Curation

Official splits treat proteins as grouped biological examples, not IID rows in a text file:

- candidates are filtered by length, resolution, monomer status, and strict sequence content
- candidates with non-standard amino-acid tokens are excluded from the official split
- MMseqs2, or a locked TSV produced with the same MMseqs2 settings, defines sequence-homology groups
- train, public val, and hidden val are cluster-disjoint and PDB-entry-disjoint
- representatives are chosen by structure quality before length, so duplicate groups do not over-contribute low-quality chains
- structure metadata is required before splitting; the official allocation is stratified by secondary-structure class, broad domain architecture, length bin, and resolution bin
- metadata sources include pinned CATH, SCOPe, ECOD, and RCSB-style structural classifications when those records cover a chain
- required OpenFold feature-asset availability is enforced before split generation
- the lock records source hashes, filtering counts, clustering mode, grouping policy, stratification fields, split quality metrics, and per-split alpha/beta/mixed distributions

The metadata builder projects broad secondary-structure fractions from domain architecture classes so every eligible chain has a deterministic split signal before any label mmCIFs are downloaded. All metadata signals are used only for split balancing and audit reports, never for scoring.

## Data Format



Official preprocessing writes split artifacts per chain:
- features: `data/processed_features/<encoded_chain_id>.npz`
- labels: `data/processed_labels/<encoded_chain_id>.npz`

`<encoded_chain_id>` is nanoFold's filesystem-safe chain stem. It preserves
case-sensitive PDB chain IDs on every supported filesystem.

Required feature keys:
- `aatype`, `msa`, `deletions`
- `residue_index` — `(L,)` int32, contiguous 0..L-1 and available during sealed inference
- `between_segment_residues` — `(L,)` int32, zero for single-chain examples
- `template_aatype`, `template_ca_coords`, `template_ca_mask`

Required label keys:
- `ca_coords`, `ca_mask`
- `atom14_positions` — `(L, 14, 3)` float32, full atom14 layout
- `atom14_mask` — `(L, 14)` bool, True where coordinate was present in mmCIF

Additional label metadata:
- `residue_index` — `(L,)` int32, duplicated for convenience when present
- `resolution` — `()` float32, Å (0.0 if unknown)

Preprocessing run metadata is captured in `<processed_features_dir>/preprocess_meta.json`. Its SHA256 is folded into the dataset fingerprint, so changes to preprocessing flags, projection thresholds, dependency metadata, or source revision are visible to the verifier.

Official scoring requires atom14 labels, and submissions must return `pred_atom14` shaped `(B, L, 14, 3)`. The runtime derives the Cα view from atom14 slot 1 for diagnostics.

The official tracks disable templates by preprocessing with `T=0`; template-enabled tracks require explicit leakage filters.

Config schema uses:
- `data.processed_features_dir`
- `data.processed_labels_dir`


read more here - more here - nanoFold-Competition/docs/DATA.md


# Training 

```
cd /root/nanoFold-Competition/
source .venv/bin/activate

CUBLAS_WORKSPACE_CONFIG=:4096:8 python train.py \
  --config submissions/minalphafold2/config.yaml \
  --track limited --reset-run

# training should take around 15mins. 
# Try to create a copy of submissions/minalphafold2/ and edit config file to do a dry run on a very small number of steps - .e.g change steps from 1,500 to 50. 
# verify that it works 


```

# Score Public Validation

After training has written a checkpoint, generate public validation predictions:

```bash
mkdir -p runs/minalphafold2_reference/_forbid_labels

python predict.py \
  --config submissions/minalphafold2/config.yaml \
  --ckpt runs/minalphafold2_reference/checkpoints/ckpt_last.pt \
  --split val \
  --track limited \
  --official \
  --forbid-labels-dir runs/minalphafold2_reference/_forbid_labels \
  --pred-out-dir runs/minalphafold2_reference/public_predictions \
  --save runs/minalphafold2_reference/predict_val.json
```

Then score those predictions:

```bash
python score.py \
  --prediction-summary runs/minalphafold2_reference/predict_val.json \
  --features-dir data/processed_features \
  --labels-dir data/processed_labels \
  --save runs/minalphafold2_reference/eval_val.json \
  --per-chain-out runs/minalphafold2_reference/per_chain_scores_val.jsonl
```

Public validation is for debugging and iteration. The leaderboard ranking uses the sealed hidden validation path.


# Official codebase (easy to read)

Sample code for training is given here `submissions/minalphafold2/` 

It implements code from this repository - https://github.com/ChrisHayduk/minAlphaFold2

minAlphaFold2 is a direct, readable AlphaFold2-style implementation with an Evoformer trunk, recycling, invariant point attention, and structure-module atom14 coordinate generation. It gives nanoFold a biological-prior reference submission with a compact AlphaFold2-style training stack: masked-MSA, distogram, backbone and all-atom FAPE, torsion, and pLDDT objectives.


# Your task 

1. Do a dry run with very minimal data (change the max-steps in config - e.g. 50 steps instead of 1500). Verify that training works. Verify that predict.py works on this dry run. Verify that score.py works on this dry run. Verify that everything loads on GPU correctly always. 
2. Perform investigation on the output format and what it means. Write a detailed guide in a separate .md file of your understanding. 
3. Create a plan of how you are going to track performance of training and outputs (score.py and predictions.py) such that you can improve model performance. 
4. If necessary - i have installed weights and biases on this machine with my account - and wandb mcp - you can use this ability to put certain checkpoints to monitor training - and then read the logs to see how you can improve the cod. 
5. ONce you understand how to do a dry training run, what kind of outputs are produced by predict.py and score.py, and how to insert wandb logs to monitor runs - then you must produce a plan - detailed plan - on how to beat the baseline performance. 
6. To run a baseline performance you run training run on 1500 steps - 
the exact config copy of it can be found here - submissions/minalphafold2_REFERENCE/config.yaml 

7. Create a detailed plan for how you may beat the baseline. 
8. TO help you - perform deep research into how this repo actually works - for this you may find useful to read this repo -  https://github.com/ChrisHayduk/minAlphaFold2  -which is implemented in the refrence submission - `submissions/minalphafold2/` . 


9. Make sure you create detailed plans as .md and reports of your understand of what you need to do (task.md) and your understand of the outputss and everything else - all in one folder called serge_v1/. It must be in root of this repository. 

remember that all the data was already downloaded and all the dependencies installed. do not attempt to do this here. 

10. Finally - baseline run already exists here - 

ls runs/minalphafold2_reference/ with checkpoints - check it out. 
It was executed using this code - ls submissions/minalphafold2



ultrathink. 