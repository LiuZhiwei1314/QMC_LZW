# QMC_LZW output analysis

Tools for reading `outputs/<run>/tensorboard/events...`, exporting scalar CSV files,
plotting training curves, and summarizing the final stable segment.

## Usage

From the `QMC_LZW` directory:

```bash
conda run -p /vepfs-mlp2/c20250516/250504030/env/qmc python output_analysis/analyze_run.py \
  --run outputs/carbon_spinblock_test_LZW5182100 \
  --tail 5000
```

Outputs are written to `<run>/analysis/` by default:

- `csv/*.csv`: one CSV per TensorBoard scalar tag
- `scalars_combined.csv`: selected scalar tags joined by step
- `training_curves.png`: loss, variance, pmove, and mcmc width
- `loss_tail.png`: tail-region loss with mean and standard-error band
- `summary.json`: statistics for the selected tail window

Use `--out some/path` to choose a different output directory.
