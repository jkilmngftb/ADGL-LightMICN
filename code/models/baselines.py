"""Matched one-step baselines for the frozen 12-step MICN experiment.

The implementations preserve the defining mechanism of each neural baseline
while exposing one contract: [B,12,15,8] -> [B,15].
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ModelConfig:
    num_nodes: int = 15
    seq_len: int = 12
    input_dim: int = 8
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.1
    target_feature: int = 0
    node_embedding_dim: int = 16


def _row_transition(adj):
    graph = np.maximum(np.asarray(adj, np.float32), 0.0)
    graph = graph + np.eye(graph.shape[0], dtype=np.float32)
    return graph / np.maximum(graph.sum(1, keepdims=True), 1e-6)


def _sym_norm(adj):
    graph = np.maximum(np.asarray(adj, np.float32), 0.0)
    graph = 0.5 * (graph + graph.T) + np.eye(graph.shape[0], dtype=np.float32)
    degree = np.maximum(graph.sum(1), 1e-6)
    inv = degree ** -0.5
    return (inv[:, None] * graph * inv[None, :]).astype(np.float32)


class SharedLSTM(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.nodes = cfg.num_nodes
        self.encoder = tf.keras.layers.LSTM(cfg.hidden_dim, dropout=cfg.dropout)
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        b = tf.shape(x)[0]
        series = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                            [b * self.nodes, tf.shape(x)[1], tf.shape(x)[3]])
        pred = self.head(self.encoder(series, training=training))
        return tf.reshape(pred, [b, self.nodes])


class CausalBlock(tf.keras.layers.Layer):
    def __init__(self, units, dilation, dropout):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(units, 2, padding="causal", dilation_rate=dilation,
                                            activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(units, 2, padding="causal", dilation_rate=dilation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = tf.keras.layers.LayerNormalization()
        self.projection = tf.keras.layers.Dense(units)

    def call(self, x, training=None):
        residual = self.projection(x)
        hidden = self.conv1(x)
        hidden = self.dropout(self.conv2(hidden), training=training)
        return self.norm(tf.nn.relu(residual + hidden))


class TCNForecast(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.nodes = cfg.num_nodes
        self.blocks = [CausalBlock(cfg.hidden_dim, d, cfg.dropout) for d in (1, 2, 4)]
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        b = tf.shape(x)[0]
        hidden = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                            [b * self.nodes, tf.shape(x)[1], tf.shape(x)[3]])
        for block in self.blocks:
            hidden = block(hidden, training=training)
        return tf.reshape(self.head(hidden[:, -1]), [b, self.nodes])


class DLinearForecast(tf.keras.Model):
    """Channel-wise DLinear adaptation; forecasts all channels and selects temperature."""
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        channels = cfg.num_nodes * cfg.input_dim
        self.trend_weight = self.add_weight(name="trend_weight", shape=(channels, cfg.seq_len),
                                            initializer=tf.keras.initializers.Constant(1.0 / cfg.seq_len))
        self.season_weight = self.add_weight(name="season_weight", shape=(channels, cfg.seq_len),
                                             initializer=tf.keras.initializers.Constant(1.0 / cfg.seq_len))
        self.trend_bias = self.add_weight(name="trend_bias", shape=(channels,), initializer="zeros")
        self.season_bias = self.add_weight(name="season_bias", shape=(channels,), initializer="zeros")
        self.target_indices = tf.constant(
            [node * cfg.input_dim + cfg.target_feature for node in range(cfg.num_nodes)], tf.int32)

    def call(self, inputs, training=None):
        del training
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        flat = tf.reshape(x, [tf.shape(x)[0], self.cfg.seq_len, -1])
        trend = tf.nn.avg_pool1d(flat, ksize=3, strides=1, padding="SAME")
        seasonal = flat - trend
        trend = tf.transpose(trend, [0, 2, 1])
        seasonal = tf.transpose(seasonal, [0, 2, 1])
        output = tf.reduce_sum(trend * self.trend_weight[None], -1) + self.trend_bias
        output += tf.reduce_sum(seasonal * self.season_weight[None], -1) + self.season_bias
        return tf.gather(output, self.target_indices, axis=1)


class StationITransformer(tf.keras.Model):
    """iTransformer-style station-variate tokens for multistation forecasting."""
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.embedding = tf.keras.layers.Dense(cfg.hidden_dim)
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.heads, key_dim=cfg.hidden_dim // cfg.heads, dropout=cfg.dropout)
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(2 * cfg.hidden_dim, activation="gelu"),
            tf.keras.layers.Dropout(cfg.dropout),
            tf.keras.layers.Dense(cfg.hidden_dim),
        ])
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        b = tf.shape(x)[0]
        tokens = tf.reshape(tf.transpose(x, [0, 2, 1, 3]),
                            [b, self.cfg.num_nodes, self.cfg.seq_len * self.cfg.input_dim])
        tokens = self.embedding(tokens)
        attended = self.attention(tokens, tokens, training=training)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.ffn(tokens, training=training))
        return tf.squeeze(self.head(tokens), -1)


class GCNMLP(tf.keras.Model):
    def __init__(self, cfg, adj, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.support = tf.constant(_sym_norm(adj), tf.float32)
        self.input_projection = tf.keras.layers.Dense(cfg.hidden_dim, activation="relu")
        self.graph_projection = tf.keras.layers.Dense(cfg.hidden_dim, activation="relu")
        self.temporal = tf.keras.Sequential([
            tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            tf.keras.layers.Dropout(cfg.dropout),
            tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
        ])
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        b = tf.shape(x)[0]
        hidden = self.input_projection(x)
        hidden = self.graph_projection(tf.einsum("ij,btjf->btif", self.support, hidden))
        hidden = tf.reshape(tf.transpose(hidden, [0, 2, 1, 3]),
                            [b, self.cfg.num_nodes, self.cfg.seq_len * self.cfg.hidden_dim])
        return tf.squeeze(self.head(self.temporal(hidden, training=training)), -1)


class GraphGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, support):
        super().__init__()
        self.units = units
        self.support = tf.constant(support, tf.float32)
        self.gates = tf.keras.layers.Dense(2 * units)
        self.candidate = tf.keras.layers.Dense(units)

    def call(self, x, state):
        combined = tf.concat([x, state], -1)
        combined = tf.einsum("ij,bjf->bif", self.support, combined)
        reset, update = tf.split(tf.nn.sigmoid(self.gates(combined)), 2, -1)
        candidate_input = tf.concat([x, reset * state], -1)
        candidate_input = tf.einsum("ij,bjf->bif", self.support, candidate_input)
        proposal = tf.nn.tanh(self.candidate(candidate_input))
        return update * state + (1.0 - update) * proposal


class TGCNForecast(tf.keras.Model):
    def __init__(self, cfg, adj, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cell = GraphGRUCell(cfg.hidden_dim, _sym_norm(adj))
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        state = tf.zeros([tf.shape(x)[0], self.cfg.num_nodes, self.cfg.hidden_dim], x.dtype)
        for x_t in tf.unstack(x, axis=1):
            state = self.cell(x_t, state)
        return tf.squeeze(self.head(self.dropout(state, training=training)), -1)


class MTGraphConstructor(tf.keras.layers.Layer):
    """Directed antisymmetric Top-k graph constructor used by MTGNN."""
    def __init__(self, cfg, alpha=3.0, top_k=4):
        super().__init__()
        self.alpha = float(alpha)
        self.top_k = min(int(top_k), cfg.num_nodes)
        shape = (cfg.num_nodes, cfg.node_embedding_dim)
        self.embedding1 = self.add_weight(name="embedding1", shape=shape,
                                          initializer="glorot_uniform")
        self.embedding2 = self.add_weight(name="embedding2", shape=shape,
                                          initializer="glorot_uniform")
        self.linear1 = tf.keras.layers.Dense(cfg.node_embedding_dim)
        self.linear2 = tf.keras.layers.Dense(cfg.node_embedding_dim)

    def call(self, inputs=None):
        del inputs
        nodevec1 = tf.nn.tanh(self.alpha * self.linear1(self.embedding1))
        nodevec2 = tf.nn.tanh(self.alpha * self.linear2(self.embedding2))
        score = tf.matmul(nodevec1, nodevec2, transpose_b=True)
        score -= tf.matmul(nodevec2, nodevec1, transpose_b=True)
        adjacency = tf.nn.relu(tf.nn.tanh(self.alpha * score))
        indices = tf.math.top_k(adjacency, k=self.top_k).indices
        mask = tf.reduce_max(tf.one_hot(indices, depth=tf.shape(adjacency)[1],
                                        dtype=adjacency.dtype), axis=1)
        return adjacency * mask


class MTGNNForecast(tf.keras.Model):
    """One-layer MTGNN task adaptation with inception and bidirectional mix-hop."""
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.alpha = 0.05
        self.graph = MTGraphConstructor(cfg, alpha=3.0, top_k=4)
        self.input_projection = tf.keras.layers.Dense(cfg.hidden_dim)
        branch_units = cfg.hidden_dim // 4
        self.filter_convs = [tf.keras.layers.Conv1D(branch_units, k, padding="causal")
                             for k in (2, 3, 6, 7)]
        self.gate_convs = [tf.keras.layers.Conv1D(branch_units, k, padding="causal")
                           for k in (2, 3, 6, 7)]
        self.forward_projection = tf.keras.layers.Dense(cfg.hidden_dim)
        self.backward_projection = tf.keras.layers.Dense(cfg.hidden_dim)
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(1)

    def _mixprop(self, x, adjacency, projection):
        adjacency = adjacency + tf.eye(self.cfg.num_nodes, dtype=x.dtype)
        adjacency = adjacency / tf.maximum(tf.reduce_sum(adjacency, axis=1, keepdims=True), 1e-6)
        state = x
        outputs = [state]
        for _ in range(2):
            propagated = tf.einsum("ij,btjf->btif", adjacency, state)
            state = self.alpha * x + (1.0 - self.alpha) * propagated
            outputs.append(state)
        return projection(tf.concat(outputs, axis=-1))

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        hidden = self.input_projection(x)
        shape = tf.shape(hidden)
        flat = tf.reshape(tf.transpose(hidden, [0, 2, 1, 3]),
                          [shape[0] * shape[2], shape[1], self.cfg.hidden_dim])
        filters = tf.concat([tf.nn.tanh(layer(flat)) for layer in self.filter_convs], -1)
        gates = tf.concat([tf.nn.sigmoid(layer(flat)) for layer in self.gate_convs], -1)
        temporal = self.dropout(filters * gates, training=training)
        temporal = tf.transpose(tf.reshape(
            temporal, [shape[0], shape[2], shape[1], self.cfg.hidden_dim]), [0, 2, 1, 3])
        adjacency = self.graph(hidden)
        graph = self._mixprop(temporal, adjacency, self.forward_projection)
        graph += self._mixprop(temporal, tf.transpose(adjacency), self.backward_projection)
        hidden = self.norm(hidden + graph)
        return tf.squeeze(self.head(hidden[:, -1]), -1)


class DiffusionConv(tf.keras.layers.Layer):
    def __init__(self, units, adj, steps=2):
        super().__init__()
        self.forward = tf.constant(_row_transition(adj), tf.float32)
        self.backward = tf.constant(_row_transition(np.asarray(adj).T), tf.float32)
        self.steps = steps
        self.projection = tf.keras.layers.Dense(units)

    def call(self, x):
        signals = [x]
        for support in (self.forward, self.backward):
            state = x
            for _ in range(self.steps):
                state = tf.einsum("ij,bjf->bif", support, state)
                signals.append(state)
        return self.projection(tf.concat(signals, -1))


class DCGRUCell(tf.keras.layers.Layer):
    def __init__(self, units, adj):
        super().__init__()
        self.gates = DiffusionConv(2 * units, adj)
        self.candidate = DiffusionConv(units, adj)

    def call(self, x, state):
        reset, update = tf.split(tf.nn.sigmoid(self.gates(tf.concat([x, state], -1))), 2, -1)
        proposal = tf.nn.tanh(self.candidate(tf.concat([x, reset * state], -1)))
        return update * state + (1.0 - update) * proposal


class DCRNNForecast(tf.keras.Model):
    def __init__(self, cfg, adj, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.cell = DCGRUCell(cfg.hidden_dim, adj)
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.head = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        state = tf.zeros([tf.shape(x)[0], self.cfg.num_nodes, self.cfg.hidden_dim], x.dtype)
        for x_t in tf.unstack(x, axis=1):
            state = self.cell(x_t, state)
        return tf.squeeze(self.head(self.dropout(state, training=training)), -1)


class AdaptiveSupport(tf.keras.layers.Layer):
    def __init__(self, cfg):
        super().__init__()
        self.left = self.add_weight(name="left", shape=(cfg.num_nodes, cfg.node_embedding_dim),
                                    initializer="glorot_uniform")
        self.right = self.add_weight(name="right", shape=(cfg.node_embedding_dim, cfg.num_nodes),
                                     initializer="glorot_uniform")

    def call(self, inputs=None):
        del inputs
        return tf.nn.softmax(tf.nn.relu(tf.matmul(self.left, self.right)), axis=-1)


class GraphWaveBlock(tf.keras.layers.Layer):
    def __init__(self, units, dilation, dropout):
        super().__init__()
        self.units = units
        self.filter_conv = tf.keras.layers.Conv1D(units, 2, padding="causal", dilation_rate=dilation)
        self.gate_conv = tf.keras.layers.Conv1D(units, 2, padding="causal", dilation_rate=dilation)
        self.graph_projection = tf.keras.layers.Dense(units)
        self.skip_projection = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, hidden, supports, training=None):
        shape = tf.shape(hidden)
        flat = tf.reshape(tf.transpose(hidden, [0, 2, 1, 3]),
                          [shape[0] * shape[2], shape[1], self.units])
        temporal = tf.nn.tanh(self.filter_conv(flat)) * tf.nn.sigmoid(self.gate_conv(flat))
        temporal = tf.transpose(tf.reshape(temporal, [shape[0], shape[2], shape[1], self.units]),
                                [0, 2, 1, 3])
        signals = [temporal]
        for support in supports:
            state = temporal
            for _ in range(2):
                state = tf.einsum("ij,btjf->btif", support, state)
                signals.append(state)
        graph = self.dropout(self.graph_projection(tf.concat(signals, -1)), training=training)
        return self.norm(hidden + graph), self.skip_projection(temporal)


class GraphWaveNetForecast(tf.keras.Model):
    def __init__(self, cfg, adj, **kwargs):
        super().__init__(**kwargs)
        self.fixed = tf.constant(_row_transition(adj), tf.float32)
        self.adaptive = AdaptiveSupport(cfg)
        self.input_projection = tf.keras.layers.Dense(cfg.hidden_dim)
        self.blocks = [GraphWaveBlock(cfg.hidden_dim, d, cfg.dropout) for d in (1, 2, 4)]
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dense(cfg.hidden_dim, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, inputs, training=None):
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        hidden = self.input_projection(x)
        skip = 0.0
        supports = (self.fixed, self.adaptive(hidden))
        for block in self.blocks:
            hidden, current_skip = block(hidden, supports, training=training)
            skip = skip + current_skip
        # Include the final graph-residual state so the last graph block is not
        # disconnected from the forecasting loss.
        return tf.squeeze(self.head(tf.nn.relu(skip[:, -1] + hidden[:, -1]),
                                    training=training), -1)


NEURAL_MODELS = ("lstm", "tcn", "dlinear", "itransformer", "gcn_mlp",
                 "tgcn", "dcrnn", "graph_wavenet", "mtgnn")


def build_model(name: str, cfg: ModelConfig, adjacency):
    builders = {
        "lstm": lambda: SharedLSTM(cfg, name="lstm"),
        "tcn": lambda: TCNForecast(cfg, name="tcn"),
        "dlinear": lambda: DLinearForecast(cfg, name="dlinear"),
        "itransformer": lambda: StationITransformer(cfg, name="itransformer"),
        "gcn_mlp": lambda: GCNMLP(cfg, adjacency, name="gcn_mlp"),
        "tgcn": lambda: TGCNForecast(cfg, adjacency, name="tgcn"),
        "dcrnn": lambda: DCRNNForecast(cfg, adjacency, name="dcrnn"),
        "graph_wavenet": lambda: GraphWaveNetForecast(cfg, adjacency, name="graph_wavenet"),
        "mtgnn": lambda: MTGNNForecast(cfg, name="mtgnn"),
    }
    if name not in builders:
        raise KeyError(name)
    return builders[name]()
