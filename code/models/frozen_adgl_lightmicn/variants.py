"""Single backbone, explicit one-component ablation definitions."""
SPECS = {
    "full": {},
    "fixed_graph": {"dynamic": False, "sparse_lambda": 0.0},
    "no_top_k": {"top_k": None},
    "no_l1": {"sparse_lambda": 0.0},
    "graph_node_embedding": {"use_embedding": True},
    "no_decomposition": {"micn_use_decomposition": False},
    "single_scale": {},
}

def derive_single_scale_kernels(candidate_kernels):
    """Remove the larger MICN scale while retaining the candidate's local scale."""
    kernels = tuple(int(value) for value in candidate_kernels)
    if len(kernels) < 2:
        raise ValueError("single_scale requires a multi-scale candidate")
    return (min(kernels),)


def effective_spec(name, candidate_kernels=(2, 4)):
    spec = {"dynamic": True, "use_embedding": False, "top_k": 4,
            "temporal_mode": "micn", "residual_output": True,
            "use_target_calendar": True, "station_conditioned_head": True,
            "use_graph_propagation": True, "micn_use_decomposition": True,
            "sparse_lambda": .5, "kernels": tuple(candidate_kernels), **SPECS[name]}
    if name == "single_scale":
        spec["kernels"] = derive_single_scale_kernels(candidate_kernels)
    return spec

def build_model(config, adjacency, name):
    from .models import GraphTemporalForecast
    spec=effective_spec(name)
    return GraphTemporalForecast(config, adj=adjacency,
        **{k:v for k,v in spec.items() if k not in ("sparse_lambda","kernels")},
        name="adgl_micn_station_head_no_embedding_residual_calendar_forecast")
