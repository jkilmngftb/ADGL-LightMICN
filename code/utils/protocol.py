"""Single source of truth for the frozen matched-comparison protocol."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_XLSX = ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
ADJ_CSV = ROOT / "dataset" / "adjacency_matrix.csv"
RESULTS = ROOT / "results" / "comparison"
CHECKPOINTS = ROOT / "checkpoints" / "baselines"

SEED = 2026
SEQ_LEN = 12
COMMON_WINDOW = 48
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 256
MAX_EPOCHS = 500
PATIENCE = 50
MIN_DELTA = 1e-3
L2_LAMBDA = 1.5e-3
LR_HALVE_EVERY = 50
LR_FLOOR = 1e-5

MODEL_ORDER = (
    "persistence", "lstm", "tcn", "dlinear", "itransformer",
    "gcn_mlp", "tgcn", "dcrnn", "graph_wavenet", "mtgnn",
)

PROPOSED_MODEL = "adgl_lightmicn"

MODEL_CONFIG = {
    "seq_len": SEQ_LEN,
    "hidden_dim": HIDDEN_DIM,
    "heads": HEADS,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "node_embedding_dim": 16,
    "top_k": 4,
    "sparse_lambda": 0.5,
    "light_micn_conv_kernels": (2, 4),
    "light_micn_layers": 1,
    "temporal_token_kernel_size": 3,
    "seed": SEED,
}

PARAMETER_COUNTS = {
    "full": 183758,
    "fixed_graph": 167374,
    "no_top_k": 183758,
    "no_l1": 183758,
    "graph_node_embedding": 186046,
    "no_decomposition": 183745,
    "single_scale": 130318,
    "tcn_encoder": 110209,
    "tcn_base": 89665,
    "adgl_tcn": 110209,
    "lightmicn_base": 163214,
    "adgl_lightmicn": 183758,
    "fixed_tcn": 93825,
    "fixed_lightmicn": 167374,
}

FROZEN_PROTOCOL = {
    "seed": SEED,
    "seq_len": SEQ_LEN,
    "common_window": COMMON_WINDOW,
    "learning_rate": LEARNING_RATE,
    "hidden_dim": HIDDEN_DIM,
    "heads": HEADS,
    "dropout": DROPOUT,
    "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS,
    "patience": PATIENCE,
    "min_delta": MIN_DELTA,
    "min_delta_role": "early_stopping_only",
    "checkpoint_rule": "save_every_strict_validation_rmse_improvement",
    "optimizer": "Adam",
    "lr_schedule": "halve_every_50_epochs_floor_1e-5",
    "l2_lambda": L2_LAMBDA,
    "split": "chronological_70_10_20",
    "checkpoint_selection": "strict_validation_rmse_improvement_within_each_run",
    "test_dates": "aligned_to_common_window_48",
    "proposed_model": {
        "temporal_mode": "micn",
        "micn_kernels": [2, 4],
        "micn_layers": 1,
        "trend_initialization": 1.0 / 12.0,
        "dynamic_graph": True,
        "graph_node_embedding": False,
        "top_k": 4,
        "sparse_lambda": 0.5,
        "graph_propagation": True,
        "target_calendar": True,
        "residual_prediction": True,
        "station_conditioned_head": True,
        "parameter_count_reference": 183758,
    },
}
