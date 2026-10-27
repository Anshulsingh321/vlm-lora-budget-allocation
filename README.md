# Modality-Specific LoRA Budget Allocation in Vision-Language Models

An empirical study of how to allocate a fixed LoRA parameter budget between
the vision and text components of a pretrained vision-language model.

## Research question

Does the distribution of a fixed adaptation budget across the vision and text
towers affect image-text retrieval performance?

## Experimental setup

- **Base model:** `openai/clip-vit-base-patch32`
- **Adaptation:** LoRA targeting `q_proj` and `v_proj`
- **Parameter budgets:** 294,912 and 589,824 trainable parameters
- **Vision/text allocations:** 100/0, 75/25, 50/50, 25/75, and 0/100
- **Random seeds:** 42, 123, and 456
- **Evaluation datasets:** Flickr30k and the COCO Karpathy 5K evaluation set
- **Primary metric:** Recall@1 (R@1)

The experiment covers 5 allocation strategies × 2 budgets × 3 seeds,
evaluated on both datasets.

## Results

Mean R@1 (%) across three seeds, reported as mean ± sample standard deviation.

| Dataset | Budget | 100/0 | 75/25 | 50/50 | 25/75 | 0/100 |
|---|---:|---:|---:|---:|---:|---:|
| Flickr30k | 294,912 | 76.58 ± 0.39 | 78.87 ± 0.05 | 78.68 ± 0.31 | 78.69 ± 0.40 | 75.00 ± 0.25 |
| Flickr30k | 589,824 | 76.28 ± 0.29 | 78.47 ± 0.45 | 78.08 ± 0.13 | 78.62 ± 0.26 | 74.74 ± 0.23 |
| COCO Karpathy 5K | 294,912 | 46.15 ± 0.21 | 47.54 ± 0.18 | 47.58 ± 0.39 | 47.56 ± 0.22 | 45.38 ± 0.41 |
| COCO Karpathy 5K | 589,824 | 45.75 ± 0.34 | 47.36 ± 0.30 | 47.46 ± 0.06 | 47.16 ± 0.24 | 44.96 ± 0.19 |

Frozen CLIP baseline R@1:
- Flickr30k: 69.13%
- COCO Karpathy 5K: 40.29%

## Main observations

1. Mixed vision/text allocations outperformed both single-modality extremes
   in mean R@1 across all four dataset-budget settings.
2. The best-performing mixed allocation depended on the dataset and budget.
   Differences among some mixed allocations were small.
3. The 294,912-parameter budget had higher mean R@1 than the
   589,824-parameter budget for every allocation in these experiments.
   This is an observation from this experiment, not evidence that smaller
   budgets are universally better.
4. Results are based on three seeds. Reported standard deviations describe
   observed seed variation; they do not by themselves establish statistical
   significance.

## Repository structure

- `configs/`: experiment configurations
- `src/`: model, evaluation, and utility code
- `scripts/`: training, evaluation, verification, and analysis scripts
- `results/`: per-run outputs and aggregate research results
- `notebooks/`: exploratory work
- `tests/`: tests

Trained adapter weights and local datasets are excluded from version control.

## Reproducing the experiments

1. Create and activate a Python environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Obtain the datasets separately and configure their local paths.
4. Review `configs/train.yaml` and the relevant scripts before running training
   or evaluation; dataset paths and runtime settings may need to be adapted
   to your environment.
5. Use the analysis scripts to regenerate summaries from the experiment outputs.

Dataset files and trained model checkpoints are not distributed in this repository.

## Scope and limitations

This project studies fixed, predefined LoRA allocations. It does not implement
an adaptive algorithm that automatically chooses the optimal allocation.
Conclusions are limited to the tested model, parameter budgets, datasets,
seeds, and evaluation setup.

## License

A project license has not yet been specified.
