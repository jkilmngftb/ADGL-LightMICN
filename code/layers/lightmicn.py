"""MICN temporal forecast adapter for GCN latent station histories (TensorFlow).

Source: wanghq21/MICN, commit 370c69b841d72246556ca05dd23163c560c22b5a,
models/local_global.py and models/model.py (ICLR 2023).

Preserves MIC multi-scale decomposition, downsample/isometric/transpose conv,
shared normalization, ReLU FFN, seasonal projection and trend regression.
Adaptations: each station's GCN channels are the input variates; output is a
64-dim latent one-step forecast for the unchanged station prediction head.
Uses the existing target-calendar fusion downstream, rather than introducing
official time-marker embedding. Uses the common configured dropout throughout.
This is an encoder adaptation, not a reproduction of the complete raw-data MICN.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf


class MICNMovingAverage(tf.keras.layers.Layer):
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("MICN decomposition kernel must be positive and odd")

    def call(self, inputs):
        radius = (self.kernel_size-1)//2
        padded = tf.concat([tf.repeat(inputs[:, :1], radius, axis=1), inputs,
            tf.repeat(inputs[:, -1:], radius, axis=1)], axis=1)
        return tf.nn.avg_pool1d(padded, self.kernel_size, strides=1, padding="VALID")

    def get_config(self):
        return {**super().get_config(), "kernel_size": self.kernel_size}


class MICNLocalGlobalBlock(tf.keras.layers.Layer):
    """Direct NWC port of the official MIC block; shared norm across branches."""
    def __init__(self, hidden_dim, sequence_length, conv_kernels=(2, 4), dropout=0.1, use_decomposition=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.sequence_length = int(sequence_length)
        self.conv_kernels = tuple(int(k) for k in conv_kernels)
        self.dropout_rate = float(dropout)
        self.use_decomposition = bool(use_decomposition)
        if not self.conv_kernels or any(k < 2 or k > self.sequence_length for k in self.conv_kernels):
            raise ValueError("MICN convolution scales must be within [2, sequence_length]")
        self.decompositions = [MICNMovingAverage(k if k % 2 else k+1, name=f"decomposition_{i}")
            for i, k in enumerate(self.conv_kernels)]
        self.downsample = [tf.keras.layers.Conv1D(self.hidden_dim, k, strides=k,
            padding="valid", name=f"downsample_{i}") for i, k in enumerate(self.conv_kernels)]
        self.down_lengths = [(self.sequence_length+2*(k//2)-k)//k+1 for k in self.conv_kernels]
        self.isometric = [tf.keras.layers.Conv1D(self.hidden_dim, length, padding="valid",
            name=f"isometric_{i}") for i, length in enumerate(self.down_lengths)]
        self.upsample = [tf.keras.layers.Conv1DTranspose(self.hidden_dim, k, strides=k,
            padding="valid", name=f"upsample_{i}") for i, k in enumerate(self.conv_kernels)]
        # Official MIC uses the SAME LayerNorm for every scale, both before
        # upsampling and after the seasonal residual; do not split its weights.
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="shared_mic_norm")
        self.merge = tf.keras.layers.Conv2D(self.hidden_dim, (len(self.conv_kernels), 1),
            padding="valid", name="multiscale_merge")
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(self.hidden_dim*4, activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate), tf.keras.layers.Dense(self.hidden_dim)], name="mic_ffn")
        self.ffn_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="mic_ffn_norm")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        tf.debugging.assert_equal(tf.shape(inputs)[1], self.sequence_length)
        branches = []
        for k, decomp, down, iso, up in zip(self.conv_kernels, self.decompositions,
                self.downsample, self.isometric, self.upsample):
            seasonal = inputs-decomp(inputs) if self.use_decomposition else inputs
            padded = tf.pad(seasonal, [[0, 0], [k//2, k//2], [0, 0]])
            local = self.dropout(tf.math.tanh(down(padded)), training=training)
            prefix = tf.zeros([tf.shape(local)[0], tf.shape(local)[1]-1, self.hidden_dim], dtype=local.dtype)
            global_hidden = self.dropout(tf.math.tanh(iso(tf.concat([prefix, local], axis=1))), training=training)
            global_hidden = self.norm(global_hidden+local)
            restored = self.dropout(tf.math.tanh(up(global_hidden)), training=training)
            branches.append(self.norm(restored[:, :self.sequence_length]+seasonal))
        merged = tf.squeeze(self.merge(tf.stack(branches, axis=1)), axis=1)
        return self.ffn_norm(merged+self.ffn(merged, training=training))

    def get_config(self):
        return {**super().get_config(), "hidden_dim": self.hidden_dim,
            "sequence_length": self.sequence_length, "conv_kernels": self.conv_kernels,
            "dropout": self.dropout_rate, "use_decomposition": self.use_decomposition}


class MICNTemporalEncoder(tf.keras.layers.Layer):
    """One-step MICN latent forecast per station, output [B,N,H]."""
    def __init__(self, seq_len, num_nodes, hidden_dim=64, conv_kernels=(2, 4),
            num_layers=1, dropout=0.1, use_decomposition=True, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.conv_kernels = tuple(int(k) for k in conv_kernels)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.use_decomposition = bool(use_decomposition)
        if self.num_layers < 1 or self.hidden_dim % 2:
            raise ValueError("MICN needs >=1 layer and an even latent dimension")
        self.decompositions = [MICNMovingAverage(k if k % 2 else k+1)
            for k in self.conv_kernels]
        # Official TokenEmbedding uses circular-padding Conv1D(kernel=3).
        self.value_embedding = tf.keras.layers.Conv1D(self.hidden_dim, 3, padding="valid",
            kernel_initializer=tf.keras.initializers.HeNormal(), name="micn_value_embedding")
        positions = np.arange(self.seq_len+1, dtype=np.float32)[:, None]
        frequencies = np.exp(np.arange(0, self.hidden_dim, 2, dtype=np.float32)*(-np.log(10000.0)/self.hidden_dim))
        pe = np.zeros((self.seq_len+1, self.hidden_dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(positions*frequencies)
        pe[:, 1::2] = np.cos(positions*frequencies)
        self.position_embedding = tf.constant(pe[None])
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.blocks = [MICNLocalGlobalBlock(self.hidden_dim, self.seq_len+1,
            self.conv_kernels, self.dropout_rate, use_decomposition=self.use_decomposition,
            name=f"micn_block_{i}") for i in range(self.num_layers)]
        self.seasonal_projection = tf.keras.layers.Dense(self.hidden_dim, name="micn_seasonal_projection")
        # The trainer overrides this kernel to 1/seq_len for the paper protocol;
        # the initializer stays at ones so that override remains explicit.
        self.trend_regression = tf.keras.layers.Dense(1, kernel_initializer="ones",
            name="micn_trend_regression") if self.use_decomposition else None

    def call(self, graph_hidden, training=None):
        tf.debugging.assert_rank(graph_hidden, 4)
        tf.debugging.assert_equal(tf.shape(graph_hidden)[1:], [self.seq_len, self.num_nodes, self.hidden_dim])
        batch = tf.shape(graph_hidden)[0]
        histories = tf.reshape(tf.transpose(graph_hidden, [0, 2, 1, 3]),
            [batch*self.num_nodes, self.seq_len, self.hidden_dim])
        trend = (tf.add_n([decomp(histories) for decomp in self.decompositions])/len(self.decompositions)
                 if self.use_decomposition else tf.zeros_like(histories))
        seasonal = histories-trend
        extended = tf.concat([seasonal, tf.zeros_like(seasonal[:, :1])], axis=1)
        circular_padded = tf.concat([extended[:, -1:], extended, extended[:, :1]], axis=1)
        embedded = self.value_embedding(circular_padded)+tf.cast(self.position_embedding, graph_hidden.dtype)
        hidden = self.embedding_dropout(embedded, training=training)
        for block in self.blocks:
            hidden = block(hidden, training=training)
        seasonal_forecast = self.seasonal_projection(hidden[:, -1])
        trend_forecast = (tf.squeeze(self.trend_regression(tf.transpose(trend, [0, 2, 1])), axis=-1)
                          if self.trend_regression is not None else tf.zeros_like(seasonal_forecast))
        return tf.reshape(seasonal_forecast+trend_forecast, [batch, self.num_nodes, self.hidden_dim])

    def get_config(self):
        return {**super().get_config(), "seq_len": self.seq_len, "num_nodes": self.num_nodes,
            "hidden_dim": self.hidden_dim, "conv_kernels": self.conv_kernels,
            "num_layers": self.num_layers, "dropout": self.dropout_rate,
            "use_decomposition": self.use_decomposition}
