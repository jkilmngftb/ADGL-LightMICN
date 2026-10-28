"""Dynamic-graph analysis of the frozen MICN.

Loads the trained checkpoint (no retraining), runs the aligned Test inputs, and
inspects the learned sample-specific adjacency: sparsity, which station pairs are
strengthened, variability across windows, and agreement with the static adjacency.
"""
import json
from types import SimpleNamespace
from dataclasses import fields

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from exp.error_dynamic_graph.workflow_utils import PACKAGE_ROOT, configure_device, align_to_reference_dates, COMMON_WINDOW

RUN_DIR = PACKAGE_ROOT / "checkpoints" / "error_dynamic_graph" / "runs" / "w12_k2-4_lr0p0010_layers1"
XLSX = PACKAGE_ROOT / "dataset" / "CMA_Jiangsu_daily_2017_2026.xlsx"
ADJ_CSV = PACKAGE_ROOT / "dataset" / "adjacency_matrix.csv"
OUT = PACKAGE_ROOT / "results" / "error_dynamic_graph" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


def station_names():
    frame = pd.read_excel(XLSX, usecols=[0, 2])
    frame.columns = ["sid", "name"]
    frame = frame[frame["sid"].astype(str) != "sid"]
    pairs = dict(zip(frame["sid"].astype(str), frame["name"].astype(str)))
    ids = sorted(pairs, key=lambda s: int(s) if str(s).isdigit() else 0)
    return ids, [pairs[i] for i in ids]


def main():
    configure_device("gpu")
    from models.adgl_lightmicn import ModelConfig
    from utils.training import set_seed, _inputs, _predict
    from data_provider.data_loader import load_common_xlsx
    from exp.error_dynamic_graph.variants import build_model

    from utils.protocol import MODEL_CONFIG
    config_values = MODEL_CONFIG
    set_seed(int(config_values["seed"]))

    bundle = load_common_xlsx(XLSX, ADJ_CSV, seq_len=int(config_values["seq_len"]), allow_positional_adj=True)
    reference = load_common_xlsx(XLSX, ADJ_CSV, seq_len=COMMON_WINDOW, allow_positional_adj=True)
    align_to_reference_dates(bundle, reference, splits=("val", "test"))

    values = vars(SimpleNamespace(**config_values))
    model_config = ModelConfig(
        **{field.name: values[field.name] for field in fields(ModelConfig) if field.name in values},
        num_nodes=bundle.num_nodes,
    )
    model = build_model(model_config, bundle.adj, "full")
    model([x[:1] for x in _inputs(bundle, "test", target_calendar=True)], training=False)
    model.load_weights(str(RUN_DIR / "best.weights.h5"))

    prediction, details = _predict(model, bundle, "test", return_adgl=True)
    adjacency = details["adjacency"].astype(np.float64)     # [B, N, N] row-normalized
    selected = details["selected_gate"].astype(np.float64)  # [B, N, N] pre-normalization
    raw = details["raw_gate"]
    raw = raw.astype(np.float64) if raw is not None else None
    batch, nodes = adjacency.shape[0], adjacency.shape[1]
    print("test windows:", batch, "nodes:", nodes)

    ids, names = station_names()
    labels = [names[i] if i < len(names) else str(i) for i in range(nodes)]

    support = selected > 0.0
    mean_adjacency = adjacency.mean(axis=0)
    edge_frequency = support.mean(axis=0)
    mean_selected = np.where(support, selected, 0.0).sum(axis=0) / np.maximum(support.sum(axis=0), 1.0)
    raw_mean = raw.mean(axis=0) if raw is not None else None

    off_diag = ~np.eye(nodes, dtype=bool)
    density = float(support.sum(axis=2).mean() / (nodes - 1))
    edges_per_row = support.sum(axis=2).mean(axis=0)

    # window-to-window turnover of the directed Top-k supports (consecutive windows)
    inter = np.logical_and(support[1:], support[:-1]).sum(axis=(1, 2))
    union = np.logical_or(support[1:], support[:-1]).sum(axis=(1, 2))
    turnover = float(np.mean(1.0 - inter / np.maximum(union, 1)))

    static = np.asarray(bundle.adj)
    static_mask = static > 0
    learned_directed = edge_frequency > 0
    overlap_directed = float(np.logical_and(learned_directed, static_mask).sum() / max(learned_directed.sum(), 1))

    # undirected pair strength: mean adjacency over both directions, off-diagonal
    pair = (mean_adjacency + mean_adjacency.T) / 2.0
    pair[np.eye(nodes, dtype=bool)] = 0.0
    pairs = []
    for i in range(nodes):
        for j in range(i + 1, nodes):
            pairs.append({
                "station_i": labels[i], "station_j": labels[j],
                "index_i": i, "index_j": j,
                "mean_adjacency": float(pair[i, j]),
                "edge_frequency": float((edge_frequency[i, j] + edge_frequency[j, i]) / 2.0),
                "static_edge": bool(static_mask[i, j]),
            })
    top_pairs = pd.DataFrame(pairs).sort_values("mean_adjacency", ascending=False)
    top_pairs.to_csv(OUT / "dynamic_graph_top_pairs.csv", index=False, encoding="utf-8-sig")

    directed = []
    for i in range(nodes):
        for j in range(nodes):
            if i == j:
                continue
            directed.append({
                # Rows receive information and columns provide it: A_ij is j -> i.
                "source": labels[j], "target": labels[i], "index_source": j, "index_target": i,
                "edge_frequency": float(edge_frequency[i, j]),
                "mean_adjacency": float(mean_adjacency[i, j]),
                "mean_selected_gate": float(mean_selected[i, j]),
                "static_edge": bool(static_mask[i, j]),
            })
    top_directed = pd.DataFrame(directed).sort_values(
        ["edge_frequency", "mean_adjacency"], ascending=False)
    top_directed.to_csv(OUT / "dynamic_graph_top_directed_edges.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(mean_adjacency, index=labels, columns=labels).to_csv(
        OUT / "dynamic_graph_mean_adjacency.csv", encoding="utf-8-sig")
    pd.DataFrame(edge_frequency, index=labels, columns=labels).to_csv(
        OUT / "dynamic_graph_edge_frequency.csv", encoding="utf-8-sig")

    stats = {}
    try:
        stats = {k: float(v) for k, v in model.graph_statistics().items()}
    except Exception as exc:  # pragma: no cover
        stats = {"error": repr(exc)}

    summary = {
        "config_id": config_values.get("variants") and "w12_k2-4_lr0p0010_layers1",
        "checkpoint": str(RUN_DIR.relative_to(PACKAGE_ROOT)),
        "test_windows": int(batch),
        "nodes": int(nodes),
        "top_k": int(model_config.top_k),
        "density_offdiag": density,
        "mean_edges_per_row": float(edges_per_row.mean()),
        "edge_weight_sum_mean": float(adjacency.sum(axis=(1, 2)).mean()),
        "adjacency_entropy_mean": float(np.mean(-np.sum(
            np.where(adjacency > 0, adjacency * np.log(adjacency + 1e-8), 0.0), axis=-1))),
        "top_k_edge_turnover": turnover,
        "static_edge_fraction_of_learned_top_all": overlap_directed,
        "static_adjacency_density_all": float(static_mask.sum() / (nodes * nodes - nodes)),
        "top_pairs": top_pairs.head(10).to_dict(orient="records"),
        "model_graph_statistics": stats,
    }
    (OUT / "dynamic_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- heatmaps ----
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(mean_adjacency, cmap="magma")
    ax.set_xticks(range(nodes), labels, rotation=90)
    ax.set_yticks(range(nodes), labels)
    ax.set_title("Learned dynamic adjacency (mean over %d Test windows)" % batch)
    fig.colorbar(image, ax=ax, label="row-normalized weight")
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_graph_mean_adjacency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(edge_frequency, cmap="viridis")
    ax.set_xticks(range(nodes), labels, rotation=90)
    ax.set_yticks(range(nodes), labels)
    ax.set_title("Edge selection frequency (Top-k support over windows)")
    fig.colorbar(image, ax=ax, label="frequency")
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_graph_edge_frequency.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, matrix, title in zip(
        axes,
        [static_mask.astype(float), mean_adjacency, edge_frequency],
        ["Static adjacency (adjacency_matrix.csv)", "Learned mean adjacency",
         "Learned edge frequency"],
    ):
        image = ax.imshow(matrix, cmap="magma")
        ax.set_xticks(range(nodes), labels, rotation=90)
        ax.set_yticks(range(nodes), labels)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "dynamic_graph_vs_static.png", dpi=200)
    plt.close(fig)

    print(json.dumps({k: summary[k] for k in
                      ["density_offdiag", "mean_edges_per_row", "top_k_edge_turnover",
                       "static_edge_fraction_of_learned_top_all"]}, ensure_ascii=False, indent=2))
    print("top pairs:")
    print(top_pairs.head(10).to_string(index=False))
    print("[written] to", OUT)


if __name__ == "__main__":
    main()
