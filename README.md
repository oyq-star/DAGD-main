# DGAD

Diffusion-Driven Group Anomaly Detection in Spatiotemporal Trajectories: Robust Masked Sequence Imputation for Enhanced Pattern Discovery

Abstract:
Detecting group anomalies in spatiotemporal trajectories is vital for smart city security but remains challenging due to noise and complex dynamics. Existing diffusion models often suffer from "conditional contamination," where observed anomalies inadvertently guide reconstruction, masking the very deviations they aim to detect . To address this, we propose Diffusion-Driven Group Anomaly Detection (DGAD). Specifically, we introduce an Interleaved Window Masking Strategy that segments data to enforce mutual supervision and reveal latent patterns. We pair this with an Unconditional Imputation Mechanism that conditions generation on forward noise instead of partial observations. This prevents anomaly leakage and significantly widens the divergence between normal and abnormal behaviors. Furthermore, a Denoising Weighted Voting module aggregates outputs across diffusion steps to mitigate uncertainty and enhance stability.  Extensive experiments on synthetic and real-world datasets show that DGAD consistently outperforms state-of-the-art methods, improving F1-score by 1.1\% on average and 1.8\% in high-noise conditions.

# Datasets
To use the data mentioned in our paper, firstly download Machine.zip from https://drive.google.com/drive/folders/1ryksJEKALBBxv_Eb8eiSW_t_xOU1xy13?usp=sharing and put it in /scr/datasets.
Please unzip datasets.zip in datasets.

To use the new dataset, perform the following steps:

1. Upload {dataset_name}_train.pkl, {dataset_name}_test.pkl, and {dataset_name}_test_label.pkl
2. Add code to experiment_configs.py and run_experiments.py, including the feature_dim parameter.
3. Add the feature_dim parameter to evaluate.py.

# Train and inference
Replace the dataset by adjusting the configuration information in experiment_configs.py.
To reproduce the results mentioned in our paper, first, make sure you have torch and pyyaml installed in your environment. Then, use the following command to train:
```shell
python DGAD_main.py
python evaluate.py
```

# compute_score
```shell
python compute_score.py
python ensemble_proper.py
```