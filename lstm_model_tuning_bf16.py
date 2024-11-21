#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
    from google.colab import drive, files
    IN_COLAB = True
except:
    IN_COLAB = False

if IN_COLAB:
    import sys
    drive.mount('/content/gdrive', force_remount=True)
    sys.path.append("./gdrive/My Drive/Colab Notebooks/")


# In[2]:


import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import scipy.io as io
import matplotlib.pyplot as plt
import os
import time
import datetime
import copy
import dataclasses
from dataclasses import dataclass, field
import itertools
import json

from smfret.tf_layers import Attention
from smfret.tf_layers import Conv
from smfret.tf_layers import Summary
from smfret.tf_layers import PrependTaskToken
from smfret.tf_layers import Reconstructor
from smfret.tf_layers import Embedding
from smfret.tf_layers import PositionEmbedding

from smfret.trace_simulator import Simulator
from smfret.trace_simulator import ParameterGenerator
from smfret.trace_simulator import SimulatedTraceSet

from smfret.multi_task_learning import FRETStateTraceSet
from smfret.multi_task_learning import SavedTraceSet

from smfret.dataset import MatlabTraceSet
from smfret.dataset import FRETTraceSet
from smfret.learning_task import SimpleTask

from smfret.tf_learning import LRSchedule

from tqdm.auto import tqdm
from IPython import display

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})
# In[3]:


@dataclass(eq=True, frozen=True)
class NetworkParams:
    """Class for storing hyper parameters."""
    conv_width: int
    inner_dim: int
    dense_width_array: int
    dense_activation: int
    attention_num_heads: int
    attention_key_dim: int
    attention_depth: int
    max_lr: float
    warmup_steps: int
    max_iteration: int
    batch_size: int
    evolve_rate_per_iteration: int
    use_layer_norm: bool = False
    use_checkpoint: bool = False
    checkpoint_path: str = ''
    save_model_path: str = field(default='', compare = False)
    tasks: tuple = ()
    grad_clip_norm: float = 1e-2
    dtype: str = 'bfloat16'
    dropout: float = 0.0


def build_conv(params):
    """Builds the conv layer of the NN."""
    conv = Conv(width=params.conv_width, inner_dim=params.inner_dim)
    return conv


def build_transformer(params):
    """Builds the transformer components of the NN."""
    base_seq_input = keras.Input(shape=(None, params.inner_dim), dtype=params.dtype)
    # positional_embedding = PositionEmbedding()
    # seq_input = base_seq_input + positional_embedding(base_seq_input)
    seq_input = base_seq_input
    seq_input += tf.expand_dims(seq_input[:, 0, :], axis=1)  # Uses task token effectively
    for _ in range(params.attention_depth):
        seq_input_reidule = seq_input
        if params.use_layer_norm:
            ln = tf.keras.layers.LayerNormalization()
            seq_input = ln(seq_input)
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=params.inner_dim, return_sequences=True),
            merge_mode='sum',
        )
        dropout = keras.layers.Dropout(params.dropout)
        seq_input = dropout(lstm(seq_input, ))

        seq_input += seq_input_reidule
        seq_input_reidule = seq_input

        if params.use_layer_norm:
            ln = tf.keras.layers.LayerNormalization()
            seq_input = ln(seq_input)

        for dense_unit in [*params.dense_width_array, params.inner_dim]:
            dense_ff = keras.layers.Dense(units=dense_unit, activation=params.dense_activation)
            dropout = keras.layers.Dropout(params.dropout)
            seq_input = dropout(dense_ff(seq_input))

        seq_input += seq_input_reidule

        if not params.use_layer_norm:
            seq_input = tf.math.l2_normalize(seq_input, axis=-1)

    seq_output = seq_input
    transformer = tf.keras.Model(inputs=base_seq_input, outputs=seq_output)
    return transformer

def build_reconstructor(params):
    """Builds the reconstructor that maps a stride window back to each frame."""
    reconstructor = Reconstructor(width=params.conv_width)
    return reconstructor


# In[4]:


refresh = True

if refresh:
    models = {}

types = ['FRETSTATE', 'FRETSTATECOUNT', 'FRETSTATE_1STATE', 'FRETSTATE_2STATE', 'FRETSTATE_3STATE', 'FRETSTATE_4STATE']
framewise_types = ['FRETSTATE', 'FRETSTATE_1STATE', 'FRETSTATE_2STATE', 'FRETSTATE_3STATE', 'FRETSTATE_4STATE']

n_categories = {
    'FRETSTATE': 22,
    'FRETSTATECOUNT': 6,
    'FRETSTATE_1STATE': 22,
    'FRETSTATE_2STATE': 22,
    'FRETSTATE_3STATE': 22,
    'FRETSTATE_4STATE': 22,
}

TYPES = types
FRAMEWISE_TYPES = framewise_types
N_CATEGORIES = n_categories

def prepare_models(params):
    """Prepares all models for every task."""

    if IN_COLAB and params.dtype == 'float16':
      keras.mixed_precision.set_global_policy('mixed_float16')
    elif IN_COLAB and params.dtype == 'bfloat16':
      keras.mixed_precision.set_global_policy('mixed_bfloat16')
    else:
      keras.mixed_precision.set_global_policy('float32')

    general_token = PrependTaskToken(params.inner_dim)
    task_token_assignment = {name: general_token for name in types}

    if params.use_checkpoint:
      saved_model = keras.models.load_model(f'{params.checkpoint_path}/{params.tasks[0]}.h5', compile=False)
      saved_embedding = saved_model.layers[-2]
      conv = saved_embedding.conv
      transformer = saved_embedding.transformer
      reconstructor = saved_embedding.reconstructor
    else:
      conv = build_conv(params)
      transformer = build_transformer(params)
      reconstructor = build_reconstructor(params)


    models = {}

    for type_name in params.tasks:

        if params.use_checkpoint:
          saved_model = keras.models.load_model(f'{params.checkpoint_path}/{type_name}.h5', compile=False)
          saved_embedding = saved_model.layers[-2]
          prepend_task_token = saved_embedding.prepend_task_token
          readout = saved_model.layers[-1]

        else:
          readout = keras.layers.Dense(units=N_CATEGORIES[type_name])
          prepend_task_token = task_token_assignment[type_name]

        embedding = Embedding(
            conv=conv,
            transformer=transformer,
            prepend_task_token=prepend_task_token,
            reconstructor=reconstructor,
            framewise=(type_name in FRAMEWISE_TYPES),
        )

        base_input = keras.Input(shape=(None, 4))
        embedding_vector = embedding(base_input)
        logits = readout(embedding_vector)

        model = keras.Model(inputs=base_input, outputs=logits)
        model.compile(
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        if type_name not in models:
            models[type_name] = model

    return models


# In[5]:


keras.losses.SparseCategoricalCrossentropy(from_logits=True)(tf.constant([1.0], dtype=tf.float32), tf.constant([[0., 0.]], dtype=tf.bfloat16), )


# In[6]:


def read_data(params):
    """Reads training and val datasets."""
    if IN_COLAB:
      path = './gdrive/My Drive/Data/'
    else:
      path = 'saved_dataset/'
    train_sets = {}

    for name, _ in tqdm(N_CATEGORIES.items(), desc='Training Dataset'):
        file = path + f'train/{name}.npz'
        train_sets[name] = SavedTraceSet(size=params.batch_size, file=file)

    val_sets = {}
    val_size = 1000
    for name, _ in tqdm(n_categories.items(), desc='Val Dataset'):
        file = path + f'eval/{name}.npz'
        val_sets[name] = SavedTraceSet(size=val_size, file=file)

    return train_sets, val_sets


# In[7]:


VAL_FREQUENCY = 100
PLOT_CLEAR_FREQUENCY = 100
VARS = None
@dataclass(eq=True, frozen=True)
class TrainResults:
    """Stores results from model training."""
    val_loss_history: np.ndarray
    train_loss_history: np.ndarray
    val_task_loss_history: dict
    train_task_loss_history: dict

    def min_val_loss(self):
        """Calculates minimum validation loss."""
        return np.min(val_loss_history)


def combine_grad(model_vars_list, grads_list):
    """Combines mulitple calculations of grad for a single update step."""
    combined_model_vars = []
    combined_grads = []
    positions = {}

    for model_vars, grads in zip(model_vars_list, grads_list):
        for var, grad in zip(model_vars, grads):
            ref = var.ref()
            if ref in positions:
                pos = positions[ref]
                if combined_grads[pos] is None:
                    combined_grads[pos] = grad
                elif grad is not None:
                    combined_grads[pos] += grad
            else:
                combined_grads.append(grad)
                combined_model_vars.append(var)
                positions[ref] = len(combined_model_vars) - 1

    return (combined_model_vars, combined_grads)


def clip_grad_by_norm(grad, clip_norm):
    """Clips the gradient by its norm."""
    if grad is None:
        return None
    else:
        return tf.clip_by_norm(grad, clip_norm)


def train(params, models, train_sets, val_sets, save_models=True):
    """Iterates through the data and trains the models."""
    plt.figure(dpi=120)
    start = time.time()
    lr = LRSchedule(
        max_learning_rate=params.max_lr,
        warmup_steps=params.warmup_steps,
        decay_steps=int(params.max_iteration * 1.1),
    )

    val_loss_history = []
    train_loss_history = []
    val_task_loss_history = {}
    train_task_loss_history = {}

    optimizer = tf.keras.optimizers.legacy.Adam(lr)

    if params.dtype == 'float16':
        optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer, dynamic=False, initial_scale=1024.0)
        scale_loss = optimizer.get_scaled_loss
        unscale_grads = optimizer.get_unscaled_gradients
    else:
        scale_loss = lambda x: x
        unscale_grads = lambda x: x

    def train_step(dataset, model, task_name, training):
        """Runs ne step of training and returns loss."""
        task = SimpleTask(dataset.to_tensor(), dataset.label, model, task_name)
        return train_step_core(task.model, task.feature_tensor, task.labels, training)

    @tf.function(reduce_retracing=True)
    def train_step_core(model, feature_tensor, labels, training):
        """Core of the training step."""
        y_pred = model(feature_tensor, training=training)
        y_true = tf.cast(labels, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        loss = model.compiled_loss(y_true, y_pred)
        return loss

    for i in tqdm(range(params.max_iteration), position=0, leave=True):
        total_loss = 0.0
        model_vars_list = []
        grads_list = []
        for task_name in params.tasks:
            with tf.GradientTape() as tape:
                loss = train_step(train_sets[task_name], models[task_name], task_name, training=True)
                float16_safe_loss = scale_loss(loss)

            if task_name not in train_task_loss_history:
                train_task_loss_history[task_name] = [loss.numpy()]
            else:
                train_task_loss_history[task_name].append(loss.numpy())

            total_loss += loss
            train_sets[task_name].evolve(params.evolve_rate_per_iteration)
            model_vars = tape.watched_variables()
            grads = unscale_grads(tape.gradient(float16_safe_loss, model_vars))
            grads = [clip_grad_by_norm(g, params.grad_clip_norm) for g in grads]
            model_vars_list.append(model_vars)
            grads_list.append(grads)

        model_vars, grads = combine_grad(model_vars_list, grads_list)
        train_loss_history.append(total_loss.numpy())

        optimizer.apply_gradients(zip(grads, model_vars))
        # print('train step:', i, 'total_loss:', total_loss.numpy())

        if i % VAL_FREQUENCY == 0:
            val_loss = 0.0
            for task_name in params.tasks:
                val_loss += train_step(val_sets[task_name], models[task_name], task_name, training=False)
                if task_name not in val_task_loss_history:
                    val_task_loss_history[task_name] = [val_loss.numpy()]
                else:
                    val_task_loss_history[task_name].append(val_loss.numpy())
            val_loss_history.append(val_loss.numpy())
            print('validation loss:', val_loss.numpy(), 'time elapsed:', np.round(time.time() - start, 1), 's')

        if save_models:
            for task_name in params.tasks:
                if i % VAL_FREQUENCY == 0 and i > 0:
                    if np.min(val_loss_history) == val_loss.numpy():
                        models[task_name].save(f'{params.save_model_path.rstrip("/")}/{task_name}.h5')

    train_results = TrainResults(
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_task_loss_history=val_task_loss_history,
        train_task_loss_history=train_task_loss_history,
    )

    return train_results


# In[8]:


params = NetworkParams(
    conv_width=100,
    inner_dim=32,
    dense_width_array=(64, 32, 16),
    dense_activation='relu',
    attention_num_heads=8,
    attention_key_dim=16,
    attention_depth=5,
    max_lr=1e-3,
    warmup_steps=50,
    max_iteration=100,
    batch_size=128,
    evolve_rate_per_iteration=32,
)

with tf.device('/CPU:0'):
  train_sets, val_sets = read_data(params)


# In[9]:


def load_log(file):
    """Loads the log of a training experiment."""
    with open(file) as f:
        r = json.load(f)
    return r


# In[10]:


lr = [1e-2]
attention_depth = [1, 2, 3, 4]
dense_activation = ['gelu', 'relu', 'softplus']
inner_dim = [32, 64, 96, 128]
batch_size = [64]
conv_width = [50]
dropout = [0.1]

tasks = [', '.join(types)]

tasks = [tuple(task.replace(' ', '').split(',')) for task in tasks]

options = itertools.product(batch_size, inner_dim, dense_activation, attention_depth, lr, dropout, conv_width, tasks)


# In[ ]:


USE_EXISTING_LOG = True

if USE_EXISTING_LOG:
    log_path = 'logs/hptuning_20240420-194234.json'
    if os.path.exists(log_path):
        hp_tuning_results = load_log(log_path)
    else:
        hp_tuning_results = []
else:
    hp_tuning_results = []

timestr = time.strftime("%Y%m%d-%H%M%S")
if IN_COLAB:
    if not USE_EXISTING_LOG:
        log_path = f'./gdrive/My Drive/Data/logs/lstm/hptuning_{timestr}.json'
        print(log_path)
    saved_model_path_format = '/content/gdrive/MyDrive/Data/saved_models/lstm-model-{}/'
else:
    if not USE_EXISTING_LOG:
        log_path = f'logs/hptuning_{timestr}.json'
        print(log_path)
    saved_model_path_format = 'saved_models/lstm-model-{}/'

with tf.device('/CPU:0'):
    for option in options:
        batch_size, inner_dim, dense_activation, attention_depth, lr, dropout, conv_width, tasks = option
        saved_model_path = saved_model_path_format.format(time.strftime('%Y%m%d-%H%M%S'))
        params = NetworkParams(
            conv_width=conv_width,
            inner_dim=inner_dim,
            dense_width_array=[4 * inner_dim, 4 * inner_dim],
            dense_activation=dense_activation,
            attention_num_heads=1,
            attention_key_dim=inner_dim // 4,
            attention_depth=attention_depth,
            max_lr=lr,
            warmup_steps=200,
            max_iteration=1002,
            batch_size=batch_size,
            evolve_rate_per_iteration=batch_size // 2,
            use_layer_norm=True,
            use_checkpoint=False,
            checkpoint_path='/content/gdrive/MyDrive/Data/saved_models/model-20230703-154340',
            save_model_path=saved_model_path,
            tasks=tasks,
            dtype='float32',
            dropout=dropout,
        )
        
        if os.path.exists(log_path):
            hp_tuning_results = load_log(log_path)
        existing_params = [NetworkParams(**eval(e)[0]) for e in hp_tuning_results]

        if params in existing_params:
            print('ha!')
            continue

        hp_tuning_results.append(
            str((dataclasses.asdict(params), TrainResults([], [], {}, {})))
        )

        with open(log_path, 'w') as f:
            json.dump(hp_tuning_results, f)

        for name in train_sets:
            train_sets[name].resize(params.batch_size)

        try:
            models = prepare_models(params)
            train_results = train(params, models, train_sets, val_sets, save_models=False)
        except:
            hp_tuning_results = load_log(log_path)
            existing_params = [NetworkParams(**eval(e)[0]) for e in hp_tuning_results]
            for i, p in enumerate(existing_params):
                if params == p:
                    del hp_tuning_results[i]
                    with open(log_path, 'w') as f:
                        json.dump(hp_tuning_results, f)
                    break
            raise

        hp_tuning_results = load_log(log_path)
        existing_params = [NetworkParams(**eval(e)[0]) for e in hp_tuning_results]

        for i, p in enumerate(existing_params):
            if params == p:
                hp_tuning_results[i] = (
                    str((dataclasses.asdict(params), dataclasses.asdict(train_results))))
                break

        with open(log_path, 'w') as f:
            json.dump(hp_tuning_results, f)

        del models


# In[ ]:




