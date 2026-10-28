"""Keras layers and model builders for ADGL-LightMICN and matched variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf


@dataclass
class ModelConfig:
    num_nodes: int
    seq_len: int = 24
    hidden_dim: int = 64
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    node_embedding_dim: int = 16
    top_k: int = 4
    sparse_lambda: float = 1e-3
    tcn_dilations: tuple = (1, 2, 4)
    light_micn_conv_kernels: tuple = (2, 4)
    light_micn_layers: int = 1
    temporal_token_kernel_size: int = 3


class CausalResidualBlock(tf.keras.layers.Layer):
    """Causal dilated residual block for one station's time series."""

    def __init__(self, channels: int, dilation: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.conv1 = tf.keras.layers.Conv1D(self.channels, 3, padding="causal", dilation_rate=int(dilation))
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.conv2 = tf.keras.layers.Conv1D(self.channels, 3, padding="causal", dilation_rate=int(dilation))
        self.norm2 = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.residual_projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.channels:
            self.residual_projection = tf.keras.layers.Conv1D(self.channels, 1, padding="same")
        super().build(input_shape)

    def call(self, inputs, training=None):
        residual = inputs if self.residual_projection is None else self.residual_projection(inputs)
        hidden = self.conv1(inputs)
        hidden = tf.nn.relu(self.norm1(hidden))
        hidden = self.dropout(hidden, training=training)
        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        return tf.nn.relu(hidden + residual)


class TemporalConvEncoder(tf.keras.layers.Layer):
    """Stacked causal convolutions; the final state summarizes all available history."""

    def __init__(self, units: int, dropout: float = 0.1, dilations=(1, 2, 4), **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.dilations = tuple(int(x) for x in dilations)
        self.input_projection = tf.keras.layers.Conv1D(self.units, 1, padding="same")
        self.blocks = [
            CausalResidualBlock(self.units, dilation, dropout=dropout)
            for dilation in self.dilations
        ]
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training=None, return_sequences=False):
        hidden = self.input_projection(inputs)
        for block in self.blocks:
            hidden = block(hidden, training=training)
        hidden = self.norm(hidden)
        return hidden if return_sequences else hidden[:, -1, :]


class DynamicAdjacency(tf.keras.layers.Layer):
    """Sample-specific, window-level adjacency with differentiable Top-k values."""

    def __init__(self, num_nodes: int, hidden_dim: int, embedding_dim: int = 16,
                 top_k: Optional[int] = 4, use_embedding: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.top_k = None if top_k is None else int(top_k)
        self.use_embedding = bool(use_embedding)
        if self.top_k is not None and (self.top_k < 1 or self.top_k >= self.num_nodes):
            raise ValueError("top_k 必须在 1 和 num_nodes-1 之间")
        self.node_embedding = None
        self.query_projection = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)
        self.key_projection = tf.keras.layers.Dense(self.hidden_dim, use_bias=False)

    def build(self, input_shape):
        if self.use_embedding:
            self.node_embedding = self.add_weight(
                name="node_embedding",
                shape=(self.num_nodes, self.embedding_dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
                trainable=True,
            )
        super().build(input_shape)

    def call(self, node_states, training=None, return_details=False):
        del training
        # node_states: [B, T, N, D]. One graph is generated per input window.
        batch = tf.shape(node_states)[0]
        mean_state = tf.reduce_mean(node_states, axis=1)
        last_state = node_states[:, -1, :, :]
        if self.use_embedding:
            embedding = tf.broadcast_to(
                self.node_embedding[None, :, :],
                [batch, self.num_nodes, self.embedding_dim],
            )
            context = tf.concat([mean_state, last_state, embedding], axis=-1)
        else:
            context = tf.concat([mean_state, last_state], axis=-1)
        query = self.query_projection(context)
        key = self.key_projection(context)
        scores = tf.einsum("bid,bjd->bij", query, key)
        scores = scores / tf.sqrt(tf.cast(self.hidden_dim, node_states.dtype))

        off_diagonal = 1.0 - tf.eye(self.num_nodes, dtype=node_states.dtype)[None, :, :]
        raw_gate = tf.nn.sigmoid(scores) * off_diagonal
        if self.top_k is None:
            selection = off_diagonal
        else:
            masked_scores = scores + (1.0 - off_diagonal) * tf.cast(-1e9, scores.dtype)
            indices = tf.math.top_k(masked_scores, k=self.top_k).indices
            selection = tf.reduce_sum(
                tf.one_hot(indices, depth=self.num_nodes, dtype=node_states.dtype), axis=2
            )
        selected_gate = raw_gate * selection
        adjacency = selected_gate / (tf.reduce_sum(selected_gate, axis=-1, keepdims=True) + 1e-6)

        if return_details:
            return adjacency, selected_gate, raw_gate
        return adjacency


class StationConditionedPredictionHead(tf.keras.layers.Layer):
    """Shared linear head plus a low-rank station-specific correction.

    The shared term preserves the original prediction head.  A small station
    embedding generates a station-specific kernel and bias correction from
    shared parameter pools, following the node-adaptive parameter idea while
    avoiding a separate full MLP for every station.
    """

    def __init__(self, num_nodes: int, hidden_dim: int,
                 embedding_dim: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)

    def build(self, input_shape):
        self.shared_kernel = self.add_weight(
            name="shared_kernel",
            shape=(self.hidden_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.shared_bias = self.add_weight(
            name="shared_bias", shape=(), initializer="zeros", trainable=True
        )
        self.station_embedding = self.add_weight(
            name="station_embedding",
            shape=(self.num_nodes, self.embedding_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )
        self.kernel_pool = self.add_weight(
            name="kernel_pool",
            shape=(self.embedding_dim, self.hidden_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
        )
        self.bias_pool = self.add_weight(
            name="bias_pool",
            shape=(self.embedding_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, node_hidden):
        tf.debugging.assert_rank(node_hidden, 3)
        tf.debugging.assert_equal(
            tf.shape(node_hidden)[1], self.num_nodes,
            message="站点条件预测头的节点数不匹配",
        )
        tf.debugging.assert_equal(
            tf.shape(node_hidden)[2], self.hidden_dim,
            message="站点条件预测头的隐藏维度不匹配",
        )
        station_kernel_delta = tf.einsum(
            "ne,eh->nh", self.station_embedding, self.kernel_pool
        )
        station_bias_delta = tf.einsum(
            "ne,e->n", self.station_embedding, self.bias_pool
        )
        shared = tf.einsum("bnh,h->bn", node_hidden, self.shared_kernel)
        adaptive = tf.einsum(
            "bnh,nh->bn", node_hidden, station_kernel_delta
        )
        return shared + adaptive + self.shared_bias + station_bias_delta[None, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_nodes": self.num_nodes,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
        })
        return config


class GraphTemporalForecast(tf.keras.Model):
    """Shared graph-temporal backbone used by the structural ablations."""

    def __init__(self, config: ModelConfig, adj: Optional[np.ndarray] = None,
                 dynamic: bool = True, use_embedding: bool = True,
                  top_k: Optional[int] = 4, temporal_mode: str = "tcn",
                  residual_output: bool = False,
                  use_target_calendar: bool = False,
                  station_conditioned_head: bool = False,
                  use_graph_propagation: bool = True, **kwargs):
        micn_use_decomposition = kwargs.pop("micn_use_decomposition", True)
        super().__init__(**kwargs)
        self.config = config
        self.num_nodes = int(config.num_nodes)
        self.hidden_dim = int(config.hidden_dim)
        self.use_dynamic_graph = bool(dynamic)
        self.temporal_mode = temporal_mode
        self.residual_output = bool(residual_output)
        self.use_target_calendar = bool(use_target_calendar)
        self.station_conditioned_head = bool(station_conditioned_head)
        self.use_graph_propagation = bool(use_graph_propagation)
        self.input_projection = tf.keras.layers.Dense(self.hidden_dim)
        self.dynamic_adjacency = (
            DynamicAdjacency(
                self.num_nodes,
                self.hidden_dim,
                embedding_dim=config.node_embedding_dim,
                top_k=top_k,
                use_embedding=use_embedding,
                name="dynamic_adjacency",
            ) if self.use_dynamic_graph else None
        )
        if not self.use_dynamic_graph:
            if adj is None:
                raise ValueError("固定图消融需要 adj")
            fixed = np.asarray(adj, dtype=np.float32)
            fixed = 0.5 * (fixed + fixed.T)
            np.fill_diagonal(fixed, 0.0)
            row_sum = fixed.sum(axis=1, keepdims=True)
            fixed = fixed / np.maximum(row_sum, 1e-6)
            self.fixed_adjacency = tf.constant(fixed, dtype=tf.float32)
        else:
            self.fixed_adjacency = None
        self.graph_projection = tf.keras.layers.Dense(self.hidden_dim, activation="relu")
        if temporal_mode == "tcn":
            self.temporal_encoder = TemporalConvEncoder(
                self.hidden_dim, dropout=config.dropout, dilations=config.tcn_dilations
            )
        elif temporal_mode == "micn":
            from layers.lightmicn import MICNTemporalEncoder

            self.temporal_encoder = MICNTemporalEncoder(
                seq_len=config.seq_len,
                num_nodes=config.num_nodes,
                hidden_dim=config.hidden_dim,
                conv_kernels=config.light_micn_conv_kernels,
                num_layers=config.light_micn_layers,
                dropout=config.dropout,
                name="micn_temporal_encoder",
                use_decomposition=micn_use_decomposition,
            )
        else:
            raise ValueError(f"未知 temporal_mode: {temporal_mode}")
        if self.use_target_calendar:
            self.calendar_projection = tf.keras.layers.Dense(
                self.hidden_dim, activation="relu", name="target_calendar_projection"
            )
            self.calendar_fusion = tf.keras.layers.Dense(
                self.hidden_dim, activation="relu", name="target_calendar_fusion"
            )
        else:
            self.calendar_projection = None
            self.calendar_fusion = None
        if self.station_conditioned_head:
            self.output_projection = StationConditionedPredictionHead(
                self.num_nodes,
                self.hidden_dim,
                embedding_dim=config.node_embedding_dim,
                name="station_conditioned_prediction_head",
            )
        else:
            self.output_projection = tf.keras.layers.Dense(1)
        self._selected_gate = None
        self._adjacency = None
        self._raw_gate = None

    def call(self, inputs, training=None, return_adjacency=False):
        temperature = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        hidden = self.input_projection(temperature)
        batch = tf.shape(hidden)[0]
        if self.use_dynamic_graph:
            adjacency, selected_gate, raw_gate = self.dynamic_adjacency(
                hidden, training=training, return_details=True
            )
        else:
            adjacency = tf.broadcast_to(
                self.fixed_adjacency[None, :, :],
                [batch, self.num_nodes, self.num_nodes],
            )
            selected_gate = adjacency
            raw_gate = None
        identity = tf.eye(self.num_nodes, dtype=hidden.dtype)[None, :, :]
        graph = adjacency + identity
        degree = tf.reduce_sum(graph, axis=-1)
        inv_sqrt = tf.math.rsqrt(tf.maximum(degree, tf.cast(1e-6, hidden.dtype)))
        graph = inv_sqrt[:, :, None] * graph * inv_sqrt[:, None, :]
        propagated_hidden = (
            tf.einsum("bij,btjf->btif", graph, hidden)
            if self.use_graph_propagation else hidden
        )
        graph_hidden = self.graph_projection(propagated_hidden)
        length = tf.shape(graph_hidden)[1]
        series = tf.transpose(graph_hidden, [0, 2, 1, 3])
        if self.temporal_mode == "tcn":
            series = tf.reshape(series, [-1, length, self.hidden_dim])
            temporal = self.temporal_encoder(series, training=training)
            temporal = tf.reshape(temporal, [batch, self.num_nodes, self.hidden_dim])
        elif self.temporal_mode == "micn":
            temporal = self.temporal_encoder(graph_hidden, training=training)
        else:
            raise ValueError(f"未知 temporal_mode: {self.temporal_mode}")
        if self.use_target_calendar:
            if not isinstance(inputs, (tuple, list)) or len(inputs) < 3:
                raise ValueError("calendar 变体需要目标日期周期特征")
            calendar_hidden = self.calendar_projection(inputs[2])
            calendar_hidden = tf.broadcast_to(
                calendar_hidden[:, None, :],
                [tf.shape(temporal)[0], self.num_nodes, self.hidden_dim],
            )
            temporal = self.calendar_fusion(
                tf.concat([temporal, calendar_hidden], axis=-1)
            )
        if self.station_conditioned_head:
            prediction = self.output_projection(temporal)
        else:
            prediction = tf.squeeze(self.output_projection(temporal), axis=-1)
        if self.residual_output:
            prediction = temperature[:, -1, :, 0] + prediction
        self._selected_gate = selected_gate
        self._adjacency = adjacency
        self._raw_gate = raw_gate
        if return_adjacency:
            return prediction, adjacency, selected_gate, raw_gate
        return prediction

    def sparsity_loss(self):
        if self._raw_gate is None:
            return tf.constant(0.0, dtype=tf.float32)
        off_diagonal = 1.0 - tf.eye(self.num_nodes, dtype=self._raw_gate.dtype)[None, :, :]
        denominator = tf.reduce_sum(off_diagonal)
        per_sample = tf.reduce_sum(self._raw_gate * off_diagonal, axis=[-2, -1])
        return tf.reduce_mean(per_sample / tf.maximum(denominator, 1.0))

    def graph_statistics(self):
        if self._adjacency is None:
            return {}
        adjacency = self._adjacency
        entropy = -tf.reduce_sum(
            tf.where(adjacency > 0.0, adjacency * tf.math.log(adjacency + 1e-8), tf.zeros_like(adjacency)),
            axis=-1,
        )
        if self.dynamic_adjacency is None:
            offdiag = 1.0 - tf.eye(
                self.num_nodes, dtype=adjacency.dtype
            )[None, :, :]
            support_ratio = tf.reduce_mean(
                tf.reduce_sum(
                    tf.cast(adjacency > 0.0, adjacency.dtype) * offdiag,
                    axis=[-2, -1],
                )
                / tf.maximum(
                    float(self.num_nodes * (self.num_nodes - 1)), 1.0
                )
            )
        else:
            top_k = self.dynamic_adjacency.top_k
            support_ratio = tf.constant(
                (top_k if top_k is not None else self.num_nodes - 1)
                / (self.num_nodes - 1),
                dtype=adjacency.dtype,
            )
        stats = {
            "support_ratio_offdiag": float(support_ratio.numpy()),
            "adjacency_entropy_mean": float(tf.reduce_mean(entropy).numpy()),
            "effective_edges_mean": float(tf.reduce_mean(tf.exp(entropy)).numpy()),
        }
        if self._raw_gate is not None:
            selected_values = tf.boolean_mask(self._selected_gate, self._selected_gate > 0.0)
            raw_values = tf.boolean_mask(self._raw_gate, self._raw_gate > 0.0)
            stats.update({
                "raw_gate_mean": float(tf.reduce_mean(raw_values).numpy()),
                "raw_gate_std": float(tf.math.reduce_std(raw_values).numpy()),
                "selected_gate_mean": float(tf.reduce_mean(selected_values).numpy()),
                "selected_gate_std": float(tf.math.reduce_std(selected_values).numpy()),
            })
        return stats
