# Designing Convergent Overlapping Genes - README

![Scrolling MSA](assets/msa_scroll_side_by_side.gif)

## Project Overview
This project is focused on **generating convergent (tail-to-tail) overlapping genes**, with the main functionality being to predict and optimize overlapping DNA sequences for two given amino acid sequences. This is done using transformer-based models trained specifically for this purpose. An overlap length from 199 to 312 nucleotides may be user specified when generating and optimizing the overlap. The multi-objective optimization process integrates secondary structure predictions using S4PRED, long-range contacts using ESM-2 contact maps, alignment scores, and substitution scores. 

A more detailed overview can be found in the preprint located here: **Designing Convergent Overlapping Genes with Transformer Encoder Models and Lightweight Structural Proxies**: https://doi.org/10.1101/2025.11.07.687268 

If you use the code from this repository or the results, please cite the preprint.

## CLI Live Dashboard Preview
<a href="assets/tui.png">
  <img src="assets/tui.png" alt="CLI TUI snapshot" width="550">
</a>

## Getting Started (CLI-First)

The **command-line interface (CLI)** is the primary and recommended way to run this project. It is designed for local GPU execution and provides the best performance, reproducibility, and control over caching, batch runs, and visualization.

### Why CLI?

- Inference is **not generally VRAM-limited**, unless large ESM-2 models are explicitly enabled for contact map generation.
- A modern consumer-grade NVIDIA GPU (high CUDA core count) will typically outperform the GPUs available in Google Colab for both inference and overlap generation.
- Local execution avoids Colab session limits and enables persistent caching of intermediate results.
- The CLI provides full access to live **TUI** and **radar** dashboards for long optimization runs.

See the CLI documentation below for installation and usage instructions.

### Google Colab (Optional / Quick Start)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/protosome/convergent_overlaps_aa_change/blob/main/convergent_overlapping_gene_generation.ipynb)

A Google Colab notebook is provided as a convenience for rapid experimentation or environments without a local GPU. This is currently the fastest way to get running without local setup, but it is not the preferred long-term workflow. Access by clicking the Colab icon above.

### Recommended fresh download (important)
Use this exact download flow before CLI setup/run:

```bash
wget https://github.com/protosome/convergent_overlaps_aa_change/archive/refs/heads/main.zip
unzip main.zip
cd convergent_overlaps_aa_change-main
```

### Quick start

```bash
python run_overlap_cli.py \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results \
  --overlap-length-selected 310
```

### Validate environment only

```bash
python run_overlap_cli.py --check-only \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results
```

### Full CLI options (reference)
- `-h, --help`: show help.
- `--working-dir WORKING_DIR`: output directory for run artifacts and archives.
- `--excel-path EXCEL_PATH`: input `.xlsx`, `.xls`, or `.csv` path.
- `--overlap-length-selected OVERLAP_LENGTH_SELECTED`: one length or comma-separated lengths (example: `310` or `310,311`).
- `--start-row START_ROW`: 1-based start row.
- `--end-row END_ROW`: 1-based end row; use `0` to process to end.
- `--use-row-weights` / `--no-use-row-weights`: enable/disable per-row `ss/sub/aln/esm` weights from the input file.
- `--ss-w SS_W`: override SS weight.
- `--sub-w SUB_W`: override substitution weight.
- `--aln-w ALN_W`: override alignment weight.
- `--esm-w ESM_W`: override ESM weight.
- `--normalize-override-weights` / `--no-normalize-override-weights`: normalize override weights to sum to 1.
- `--first-pass-iters FIRST_PASS_ITERS`: first-pass optimization iterations.
- `--second-pass-iters SECOND_PASS_ITERS`: second-pass optimization iterations.
- `--reset-caches-per-row` / `--no-reset-caches-per-row`: reset internal caches between rows.
- `--archive` / `--no-archive`: enable/disable zip archive creation.
- `--notebook-path NOTEBOOK_PATH`: notebook file used as function runtime source.
- `--check-only` / `--no-check-only`: validate dependencies/paths and exit.
- `--setup` / `--no-setup`: run dependency + S4PRED weights setup before run/check.
- `--setup-only` / `--no-setup-only`: run setup and exit.
- `--weights-url WEIGHTS_URL`: override S4PRED weights tarball URL.
- `--force-weights-download` / `--no-force-weights-download`: re-download weights even if present.
- `--debug-checkpoints` / `--no-debug-checkpoints`: print detailed model-loading and inference checkpoints.
- `--auto-setup` / `--no-auto-setup`: auto-run setup if required deps/weights are missing.
- `--esm-device {auto,cuda,cpu}`: ESM device selection.
- `--esm-batch-size ESM_BATCH_SIZE`: ESM batch size (lower is safer for CUDA stability).
- `--esm-autocast` / `--no-esm-autocast`: fp16 autocast for ESM CUDA path.
- `--require-esm` / `--no-require-esm`: fail fast if ESM cannot be computed.
- `--display-mode {plain,tui,radar}`: `plain` keeps sequential logs; `tui` shows a live dashboard; `radar` adds forward/reverse window tracks.
- `--display-style {plain,boxed}`: style for live views (`tui`/`radar`); `boxed` provides fixed-height panel layout.
- `--plot-metric METRIC[,METRIC,...]`: one or more metrics used for live trend(s) in `tui`/`radar` mode. Allowed: `Combined,ESM_avg,SS_avg,Align1,Align2,Sub1,Sub2`.

### Command examples

Run with dual forward/reverse window radar:

```bash
python run_overlap_cli.py \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results \
  --overlap-length-selected 310 \
  --display-mode radar \
  --display-style boxed \
  --plot-metric Combined,SS_avg,ESM_avg
```

Strict ESM-on-CUDA run (fail if ESM fails):

```bash
python run_overlap_cli.py \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results \
  --overlap-length-selected 310 \
  --esm-device cuda \
  --no-esm-autocast \
  --esm-batch-size 1 \
  --require-esm \
  --debug-checkpoints
```

Force setup only:

```bash
python run_overlap_cli.py --setup --setup-only \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results
```

Process multiple overlap lengths and full row range:

```bash
python run_overlap_cli.py \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results \
  --overlap-length-selected 310,311,312 \
  --start-row 1 \
  --end-row 0
```

Run with live terminal dashboard and trend plot:

```bash
python run_overlap_cli.py \
  --excel-path test_results/aa_1_aa_2.xlsx \
  --working-dir test_results \
  --overlap-length-selected 310 \
  --display-mode tui \
  --plot-metric Combined,SS_avg,ESM_avg
```

### Live view modes (TUI/Radar)
The CLI includes two live visual modes designed for long optimization runs:

- `--display-mode tui`: live status + metrics + trends + events.
- `--display-mode radar`: same as `tui`, plus forward/reverse window tracks.

Recommended style:
- `--display-style boxed` (default): fixed-size panels so the layout does not jump.

Current plot control approach:
- Choose one or more trend metrics with `--plot-metric`.
- Rendering is intentionally simple and stable for long runs; there are no extra plot width/height/mode flags in the current CLI.

#### Radar legend
- `.` initial/unprocessed region
- `@` processed in attempt 1
- `#` processed in attempt 2+
- `>` active window on forward track
- `<` active window on reverse track

#### What the live panels show
- `Status`: attempt/window position, elapsed time, throughput (windows/sec), ETA, and startup timing note for the first window/pass (updated once per window, not continuously live).
- `Window Radar` (`radar` mode): active AA range and forward/reverse window advancement.
- `Metrics`: latest `Combined`, `SS1`, `SS2`, `SS_avg`, `ESM1`, `ESM2`, `ESM_avg`, plus deltas vs previous window and best-so-far.
- `Trends`: one compact per-metric trend line for each metric in `--plot-metric`.
- `Attempt Summary`: dedicated row for `Final metrics this attempt ...` so it stays visible.
- `Recent Events`: latest optimizer events plus SS prediction progress lines (long lines are wrapped across rows).

#### First pass timing note
During the first window of the first pass, the UI shows a note that this step may take longer because initial prediction/inference warm-up is happening (here, the model may be slow to find a compatible sequence to generate both amino acid sequences givne the constraints). Subsequent windows and passes should proceed (much) more quickly.

#### End-of-row recap
At row completion, a final boxed summary is printed with:
- final metrics
- best window metrics
- output file path

#### Color notes
- The live views use ANSI colors by default.
- If your shell sets `NO_COLOR`, output will be plain text.
- In very limited terminals, `--display-style plain` may render more cleanly.

### GPU compatibility note (important)
If you see errors like:
- `NVIDIA ... sm_120 is not compatible with the current PyTorch installation`
- `CUDA error: no kernel image is available for execution on the device`

then your PyTorch CUDA build does not support your GPU architecture. Install a newer PyTorch CUDA wheel (often nightly for very new GPUs), then verify support:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list() if torch.cuda.is_available() else "cpu-only")
PY
```

Your GPU arch (for example `sm_120`) must appear in `torch.cuda.get_arch_list()` for ESM on CUDA to work.

### Tokenization and Text Vectorization
To work with the sequences, you will need to tokenize and vectorize them. This is done automatically with the provided tokenizers. Tokenization is necessary to convert sequences into a numerical format that the model can understand, while vectorization ensures that all sequences are of consistent length for processing.

### Input File Format

Provide an Excel (`.xlsx`) or CSV file with two amino acid sequences (minimum length: **105 amino acids each**). Batched sequences can be run by including additional sequence pairs in additional rows.

- **Required columns:**
  - `aa_seq_1`  
  - `aa_seq_2`

- **Optional preservation columns:**  
  You may mark specific amino acids to **preserve** during overlap generation by placing them in square brackets.  

  **Example:**  
  - Sequence: `MRTSSRT`  
  - Preserved form: `MR[TS]SRT` → amino acids **TS** will be preserved.  

  Add these bracketed sequences to:  
  - `aa_seq_1_brackets`  
  - `aa_seq_2_brackets`

> ⚠️ If both sequences specify preserved residues at the same relative position, this can create conflicts and reduce the chance of generating valid overlaps.  

If preservation of specific amino acids is not required, you can simply **copy the same input sequences** into the bracket columns (or do not include the bracket columns in the input file).

---

### Demo Table

| aa_seq_1     | aa_seq_2     | aa_seq_1_brackets | aa_seq_2_brackets |
|--------------|--------------|-------------------|-------------------|
| MRTSSRT...   | QLGDVKP...   | MR[TS]SRT...      | QLGD[VK]P...      |
| AGPLMNQ...   | RTYKSDH...   | AGPLMNQ...        | RTYKSDH...        |

*(“...” indicates continuation; each sequence must be ≥105 amino acids long.)*

### Output
After running, one excel file will be generated per row from the input file. Each file will contain every amino acid pair from the windowed optimization process. To start, the recommendation is to sort for the higest value in the "combined score" column. However, given the stochastic nature of the optimization process, it is not safe to assume that the final sequence pair predicted will be the best.

From the output file, the "**translated_integrated_seq_1**" and "**translated_integrated_seq_2**" columns contain the full integrated amino acid sequences, and the "**integrated_seq_1**" and "**integrated_seq_2**" columns contain the integrated full DNA sequences for both amino acid sequences, respectively. These are the DNA sequences that include the convergent overlaps.

## Repository Structure
- **convergent_overlapping_gene_generation.ipynb**: Main notebook for prediction of convergent overlaps.
- **s4pred/**: Directory containing scripts and model weights for secondary structure prediction.
- **aa_change_model_set/**: Directory containing model data files.
- **protsub_matrix.py** and **blosum62_matrix.py**: Modules for similarity calculations.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- **S4PRED**: For secondary structure prediction.
- **ESM-2**: For contact-map embeddings.
- **BioPython**: For sequence analysis and manipulation.

## Dependencies

This project makes use of the following external repositories and packages. If you use this work in academic research, please also cite the corresponding publications where applicable:

- S4PRED (secondary structure prediction). Repository: https://github.com/psipred/s4pred. Reference: Moffat L, Jones DT. Increasing the accuracy of single sequence prediction methods using a deep semi-supervised learning framework. Xu J, editor. Bioinformatics. 2021 Nov 5;37(21):3744–51.  

- ESM-2 (Evolutionary Scale Modeling). Repository: https://github.com/facebookresearch/esm and https://huggingface.co/facebook/esm2_t12_35M_UR50D (for models). Reference: Lin Z, Akin H, Rao R, et al. Language models of protein sequences at the scale of evolution enable accurate structure prediction. Science. 2023 Mar 17;379(6637):1123–30.

- Biopython. Repository: https://github.com/biopython/biopython. Reference: Cock PJA, Antao T, Chang JT, et al. Biopython: freely available Python tools for computational molecular biology and bioinformatics. Bioinformatics, 2009.

## Citation

If you use this in your work, please cite the following:

- **Designing Convergent Overlapping Genes with Transformer Encoder Models and Lightweight Structural Proxies**.
Jason K. Morgan; bioRxiv, 11-2025, DOI: https://doi.org/10.1101/2025.11.07.687268


---
This README should provide you with the information you need to get started with the **Designing Convergent Overlapping Genes** project. If you encounter any issues or have questions, please feel free to reach out or open an issue on GitHub.
