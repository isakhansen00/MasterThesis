# Training

## Data setup

The training script expects all processed data to be placed in a single folder, with one set of files per year. In practice, this should look something like:

```text
project_root/
├── src/model/trainer.py
├── training_data/
│   ├── ais_points_2016.csv
│   ├── voyages_2016.csv
│   ├── vessels_2016.csv
│   ├── voyage_catches_2016.csv   (optional)
│   ├── ais_points_2017.csv
│   └── ...
```

For each year you include, the script looks for:
- `ais_points_YYYY.csv`
- `voyages_YYYY.csv`
- `vessels_YYYY.csv`

Catch files are optional. If they are missing, the model will still run, but some catch-based features will be empty.

---

## Running the model

All experiments are run from the project root. At minimum, you only need to specify the data directory and years:

```bash
python src/model/trainer.py --data-dir training_data --years 2016 2017 2018 2019 2020 2022 2023
```

A typical full run looks like:

```bash
python src/model/trainer.py \
  --data-dir output_dca80 \
  --years 2016 2017 2018 2019 2020 2022 2023 \
  --encoder-length 64 \
  --encoder-window-mode tail \
  --batch-size 64 \
  --epochs 30 \
  --model-type tft \
  --train-random-hide \
  --train-hide-min-hours 1 \
  --train-hide-max-hours 7 \
  --hide-last-hours 3
```

---

## What’s happening

The model is trained on partially observed trajectories.

- During training, the end of each voyage is randomly hidden (e.g. 1–7 hours)
- During validation and testing, a fixed cutoff is used (e.g. last 3 hours hidden)

This setup ensures that:
- the model learns to predict destinations before arrival
- evaluation remains consistent across experiments

The encoder length (e.g. 64) controls how much of the observed trajectory is used as input.

---

## Experiment variants

Most experiments are controlled through simple flags.

### Port prediction (default)

Standard destination port classification:

```bash
--model-type tft
```

---

### Multitask ETA + port

```bash
--eta-multitask
```

Adds ETA prediction as an auxiliary task..

---

### Separate ETA and port models

```bash
--two-stage-eta-port
```

- First trains an ETA model  
- Then trains the port model independently  

ETA is evaluated separately.

---

### Feature ablation

```bash
--remove-statics mmsi season catch_entropy
```

Removes selected static features to assess their impact.

---

### Robustness / cutoff sweep

```bash
--sweep-hide-hours
```

Evaluates model performance across different hidden horizons after training.

---

## Outputs

During training:
- training loss and accuracy
- validation loss and accuracy

After training:
- best model (based on validation accuracy) is restored
- final test performance is reported

Optional:
- ETA error metrics (if enabled)
- accuracy vs. hidden horizon plots

---

## Summary

- Place all yearly CSV files in one folder  
- Run the script from the project root  
- Set `--data-dir` to that folder  
- Use random hiding during training and fixed hiding for evaluation  
- Control experiments through command-line flags  

This is sufficient to reproduce all experiments.

