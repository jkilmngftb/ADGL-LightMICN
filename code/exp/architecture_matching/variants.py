"""Architecture-matched ADGL x temporal-backbone macro 2x2 definitions."""
SPECS = {
    "tcn_base": {
        "graph_mode": "none",
        "use_spatial_module": False,
        "temporal_mode": "tcn",
        "sparse_lambda": 0.0,
    },
    "adgl_tcn": {
        "graph_mode": "dynamic",
        "use_spatial_module": True,
        "temporal_mode": "tcn",
    },
    "lightmicn_base": {
        "graph_mode": "none",
        "use_spatial_module": False,
        "temporal_mode": "micn",
        "sparse_lambda": 0.0,
    },
    "adgl_lightmicn": {
        "graph_mode": "dynamic",
        "use_spatial_module": True,
        "temporal_mode": "micn",
    },
    "fixed_tcn": {
        "graph_mode": "fixed",
        "dynamic": False,
        "use_spatial_module": True,
        "temporal_mode": "tcn",
        "sparse_lambda": 0.0,
    },
    "fixed_lightmicn": {
        "graph_mode": "fixed",
        "dynamic": False,
        "use_spatial_module": True,
        "temporal_mode": "micn",
        "sparse_lambda": 0.0,
    },
}

def derive_single_scale_kernels(candidate_kernels):
    """Remove the larger MICN scale while retaining the candidate's local scale."""
    kernels = tuple(int(value) for value in candidate_kernels)
    if len(kernels) < 2:
        raise ValueError("single_scale requires a multi-scale candidate")
    return (min(kernels),)


def effective_spec(name, candidate_kernels=(2, 4)):
    spec = {"graph_mode": "dynamic", "dynamic": True,
            "use_embedding": False, "top_k": 4,
            "use_spatial_module": True,
            "temporal_mode": "micn", "residual_output": True,
            "use_target_calendar": True, "station_conditioned_head": True,
            "use_graph_propagation": True, "micn_use_decomposition": True,
            "sparse_lambda": .5, "kernels": tuple(candidate_kernels), **SPECS[name]}
    return spec

def build_model(config, adjacency, name):
    from models.architecture_matched import GraphTemporalForecast
    spec=effective_spec(name)
    return GraphTemporalForecast(config, adj=adjacency,
        **{k:v for k,v in spec.items() if k not in ("sparse_lambda","kernels")},
        name="adgl_micn_station_head_no_embedding_residual_calendar_forecast")
