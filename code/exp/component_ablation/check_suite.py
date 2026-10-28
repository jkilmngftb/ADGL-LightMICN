"""Forward/backward audit of matched ablations; no formal training."""
from pathlib import Path
from dataclasses import fields
import sys, json, hashlib, os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parent
PACKAGE_ROOT=ROOT.parents[1]
sys.path.insert(0,str(PACKAGE_ROOT))
from exp.component_ablation.train_variant import make_args
from exp.component_ablation.variants import SPECS, effective_spec, build_model, derive_single_scale_kernels
from exp.component_ablation.speed_runtime import make_train_step
import numpy as np
import tensorflow as tf
from models.adgl_lightmicn import ModelConfig
from utils.training import set_seed,_inputs,_loss_terms
from data_provider.data_loader import load_common_xlsx

args=make_args()
assert derive_single_scale_kernels((2,4)) == (2,)
assert derive_single_scale_kernels((3,5)) == (3,)
bundle=load_common_xlsx(args.data_xlsx,args.adj,seq_len=12,allow_positional_adj=True)
reference=load_common_xlsx(args.data_xlsx,args.adj,seq_len=48,allow_positional_adj=True)
for split in ('train','val'):
    dates=getattr(bundle,'date_'+split)
    ref=getattr(reference,'date_'+split)
    idx=np.flatnonzero(np.isin(dates,ref))
    assert np.array_equal(dates[idx],ref)
    for name in ('x_','y_','y_raw_','calendar_','date_','sample_weight_'):
        value=getattr(bundle,name+split)
        if value is not None: setattr(bundle,name+split,value[idx])
inputs=[tf.convert_to_tensor(x[:2]) for x in _inputs(bundle,'train',target_calendar=True)]
rows=[]
for variant in SPECS:
    tf.keras.backend.clear_session()
    set_seed(2026)
    spec=effective_spec(variant, (2, 4))
    values=vars(args).copy()
    values.update(light_micn_conv_kernels=spec['kernels'],sparse_lambda=spec['sparse_lambda'])
    config=ModelConfig(**{f.name:values[f.name] for f in fields(ModelConfig) if f.name in values},num_nodes=bundle.num_nodes)
    model=build_model(config,bundle.adj,variant)
    model(inputs,training=False)
    encoder=model.temporal_encoder
    trend=getattr(encoder,'trend_regression',None)
    if trend is not None: trend.kernel.assign(tf.ones_like(trend.kernel)/12)
    prediction=model(inputs,training=False).numpy()
    assert prediction.shape==(2,bundle.num_nodes) and np.isfinite(prediction).all()
    assert model.station_conditioned_head==spec['station_conditioned_head']
    assert model.use_graph_propagation==spec['use_graph_propagation']
    if variant=='no_decomposition':
        assert trend is None and not encoder.use_decomposition
        assert all(not block.use_decomposition for block in encoder.blocks)
    if variant=='single_scale': assert encoder.conv_kernels==derive_single_scale_kernels((2,4))
    if model.dynamic_adjacency is not None:
        adjacency=model._adjacency.numpy()
        edge_count=(adjacency>0).sum(axis=-1)
        assert np.all(edge_count==(14 if spec['top_k'] is None else 4))
    with tf.GradientTape() as tape:
        pred=model(inputs,training=True)
        terms=_loss_terms(model,tf.convert_to_tensor(bundle.y_train[:2]),pred,
            tf.convert_to_tensor(bundle.sample_weight_train[:2]),spec['sparse_lambda'])
    grads=tape.gradient(terms['total'],model.trainable_variables)
    active=[g for g in grads if g is not None]
    assert active and all(np.isfinite(tf.convert_to_tensor(g).numpy()).all() for g in active)
    captured={}
    def capture(layer,key):
        original_call=layer.call
        def call(*args,**kwargs):
            result=original_call(*args,**kwargs)
            captured[key]=result
            return result
        layer.call=call
    if getattr(encoder,'seasonal_projection',None) is not None:
        capture(encoder.seasonal_projection,'seasonal')
    if trend is not None: capture(trend,'trend')
    compiled=make_train_step(tf,model,tf.keras.optimizers.Adam(.0005),_loss_terms,spec['sparse_lambda'])
    compiled_terms=compiled(inputs,tf.convert_to_tensor(bundle.y_train[:2]),tf.convert_to_tensor(bundle.sample_weight_train[:2]))
    assert all(np.isfinite(float(v)) for v in compiled_terms.values())
    after=model(inputs,training=False).numpy()
    assert np.isfinite(after).all()
    assert all(np.isfinite(v.numpy()).all() for v in captured.values())
    rows.append(dict(variant=variant,parameters=model.count_params(),
        output_shape=list(pred.shape),finite_gradients=True,compiled_step_passed=True,effective_spec=spec,
        task_loss=float(terms['task']),weighted_sparse_loss=float(terms['weighted_sparse'])))
    print('PASS',variant,flush=True)

report=dict(passed=True,formal_training_performed=False,variants=rows,
    audit_device='cpu',gpu_tested=False,execution='tf_function',smoke_updates_per_variant=1,
    dates_sha256={s:hashlib.sha256(np.asarray(getattr(bundle,'date_'+s)).astype(str).tobytes()).hexdigest() for s in ('train','val')})
audit_path=PACKAGE_ROOT/'test_results'/'component_ablation'/'SUITE_AUDIT.json'
audit_path.parent.mkdir(parents=True,exist_ok=True)
audit_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('SUITE AUDIT PASS',flush=True)
