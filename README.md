# Dino-Nuclei

This archive contains the reference implementation of **Dino-Nuclei**, a cell instance segmentation framework that combines a DINOv3 backbone with a SAM-style mask decoder, a token gating module, a gated cosine similarity prompt synthesis mechanism, and a dual-stream decoder with cross-attention. The codebase is adapted from [CellViT](https://github.com/TIO-IKIM/CellViT); the overall pipeline (data loading, training loop, post-processing, evaluation) follows CellViT, while the model, prompt synthesis, decoder, and fine-tuning strategy implement the design described in the accompanying dissertation.

On the PanNuke dataset, the configuration shipped in `./configs/examples/cell_segmentation/try1.yaml` reproduces the main result reported in the dissertation (DINOv3-Huge backbone with LoRA, dual-stream decoder).

---

## 1. What is Included in This Archive

This archive contains **source code and configuration files only**. The following are **not** included and must be obtained separately, as described in the relevant sections below:

- The PanNuke dataset (Section 4).
- DINOv3 backbone weights (downloaded automatically from `timm` on first use; see Section 3).
- Trained model checkpoints (must be produced by running training; see Section 6).

The total uncompressed size of this archive is therefore on the order of a few MB.

## 2. Requirements

- A CUDA-capable GPU. The default configuration trains on a single NVIDIA RTX A5000 (24 GB); a full training run at the DINOv3-Huge scale takes 8–12 hours per fold.
- A working `conda` (or `mamba`) installation.
- Python and PyTorch versions are pinned in `environment.yml`; the file has been updated to current package versions, so create the environment from it directly rather than installing dependencies by hand.

## 3. Installation

Extract the archive and create the conda environment:

```bash
# Extract the archive (adjust the filename if different)
unzip dino-nuclei.zip
cd dino-nuclei

# Create the conda environment
conda env create -f environment.yml
conda activate <env-name>     # the environment name is set inside environment.yml
```

No manual download of the DINOv3 backbone weights is required: they are pulled automatically from `timm` on first use. The `model.pretrained_encoder` field in the YAML configuration is a legacy entry inherited from the CellViT pipeline and is ignored in this codebase — it can be left as-is.

## 4. Dataset Preparation

We use the PanNuke dataset in the preprocessed form expected by the CellViT data loader, which this codebase reuses without modification. The high-level steps are:

1. Download the three official PanNuke folds from the [PanNuke project page](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke).
2. Run the PanNuke preprocessing pipeline as documented in the [CellViT PanNuke guide](https://github.com/TIO-IKIM/CellViT/blob/main/docs/readmes/pannuke.md). This produces a directory containing the per-tile images, instance maps, and type labels in the layout the data loader expects.
3. Point the configuration file at the resulting directory by editing `data.dataset_path` in your YAML config.

If you follow the CellViT guide step by step, the resulting directory structure is already correct — no further reorganisation is needed on this side.

## 5. Configuration

All experimental settings are controlled by a single YAML configuration file. A reference configuration that reproduces the main result of the dissertation is provided at:

```
./configs/examples/cell_segmentation/try1.yaml
```

The fields you are most likely to edit are summarised below; everything else can usually be left at its default value.

### 5.1 Paths and logging

| Field | Meaning |
|---|---|
| `data.dataset_path` | Path to the preprocessed PanNuke directory (Section 4). |
| `data.train_folds` / `val_folds` / `test_folds` | Which of the three PanNuke folds to use for each split. The default `[0] / [2] / [1]` corresponds to the `(train: 0, val: 2, test: 1)` permutation. |
| `logging.mode` | Set to `"offline"` to disable Weights & Biases. WandB is optional; offline mode logs locally and requires no account. |
| `logging.wandb_dir` / `log_dir` | Directories for WandB and local log files. Both must already exist. |
| `gpu` | List of GPU device indices, e.g. `[0]` for single-GPU or `[0, 1]` for multi-GPU. |

### 5.2 Switching the DINOv3 backbone

The four backbone variants used in the size ablation (Section 4.4.2 of the dissertation) are switched by editing four fields together. The values must match the chosen variant; mismatches between `embed_dim`, `depth`, and `num_heads` will raise an error at model construction time.

| Variant | `model.backbone` | `model.embed_dim` | `model.depth` | `model.num_heads` |
|---|---|---|---|---|
| DINOv3-Small (~22 M) | `dinov3-s` | 384 | 12 | 6 |
| DINOv3-Base (~87 M) | `dinov3-b` | 768 | 12 | 12 |
| DINOv3-Large (~300 M) | `dinov3-l` | 1024 | 24 | 16 |
| DINOv3-Huge (~1.1 B) | `dinov3-h` | 1280 | 40 | 24 |

The reference YAML is set to DINOv3-Huge. To run, for example, the Small variant, change those four fields and leave everything else untouched.

### 5.3 Other notable training options

| Field | Meaning |
|---|---|
| `training.batch_size` | Reduce if you hit out-of-memory errors with larger backbones. |
| `training.epochs` | Default 100, matching the dissertation. |
| `training.unfreeze_epoch` | Epoch at which the LoRA-adapted backbone parameters become trainable; before this epoch only the decoder is optimised. Default 25. |
| `training.mixed_precision` | FP16 training. Enabled by default. |

The complete set of hyperparameters used to produce the reported results is documented in Appendix A of the dissertation.

## 6. Training and Evaluation

The same entry script handles both training and inference: training runs to completion and the script then evaluates the resulting checkpoint on the test fold automatically.

```bash
python ./cell_segmentation/run_cellvit.py \
    --config ./configs/examples/cell_segmentation/try1.yaml
```

The test-fold metrics (Dice, Jaccard, mPQ, bPQ, mDQ, mSQ, detection precision / recall / F1, and per-tissue / per-nuclei-type breakdowns) are written to the run's log directory at the end of the run, and additionally streamed to WandB if online mode is enabled.

To evaluate a previously trained checkpoint without retraining, set `eval_checkpoint` in the YAML to the desired checkpoint file and re-run the same command.

## 7. Reproducing the Main Results

The shipped configuration `try1.yaml` reproduces the headline numbers reported in Chapter 4 of the dissertation: DINOv3-Huge backbone with LoRA, dual-stream decoder with cross-attention, gated cosine similarity prompt, full multi-task loss, and 100 training epochs on PanNuke.

Two things are worth flagging if you are aiming to match the published numbers exactly:

- **Single-fold vs. six-fold averaging.** The main results in the dissertation are reported on the `(train: 0, val: 1, test: 2)` permutation, which corresponds to setting `train_folds: [0]`, `val_folds: [1]`, `test_folds: [2]` in the YAML. To reproduce the six-fold mean used for the CellViT and DINOv3-Small + SAM-decoder rows (Appendix C), cycle through all six permutations of the three folds and average the resulting metrics.
- **Random seed.** The default seed (`random_seed: 19`) matches the one used for the reported runs. PanNuke results are mildly seed-sensitive, particularly on the long-tailed Dead-cell category; expect small variations across seeds.

## 8. Archive Layout

```
.
├── cell_segmentation/
│   └── run_cellvit.py              # main entry point (training + evaluation)
├── configs/
│   └── examples/
│       └── cell_segmentation/
│           └── try1.yaml           # reference configuration (DINOv3-Huge + LoRA)
├── environment.yml                 # conda environment specification
└── README.md
```

The model definition, loss functions, post-processing, and data loading utilities are organised under the same module structure as the upstream CellViT repository; the new components introduced by Dino-Nuclei (token gating module, gated cosine similarity, projection neck, dual-stream decoder, cross-attention fusion, LoRA integration) are added alongside the original CellViT model definition.

## 9. Acknowledgements

This implementation builds on the [CellViT](https://github.com/TIO-IKIM/CellViT) codebase. The DINOv3 backbones are loaded through [`timm`](https://github.com/huggingface/pytorch-image-models), and LoRA adaptation is provided by the [HuggingFace PEFT](https://github.com/huggingface/peft) library. We thank the authors of CellViT, DINOv3, and the Segment Anything Model for releasing their code and pretrained weights.

## 10. Citation

If you use this code in your research, please cite the accompanying dissertation:

```bibtex
@mastersthesis{lyu2026dinonuclei,
  author  = {Ziji Lyu},
  title   = {Dino-Nuclei: Vision Transformer with Segment Anything Decoder for Precise Cell Instance Segmentation},
  school  = {University of Nottingham Ningbo China},
  year    = {2026},
  type    = {BSc Dissertation}
}
```
