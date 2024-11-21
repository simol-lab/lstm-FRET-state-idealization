"""Classes and functions for learning and HP tuning with TensorFlow."""

import tensorflow as tf
import numpy as np


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    COSINE_WAVE_PERIOD = 2000
    def __init__(self, max_learning_rate, warmup_steps, decay_steps):
        self.max_learning_rate = tf.cast(max_learning_rate, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.decay_steps = tf.cast(decay_steps, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        if step <= self.warmup_steps:
            return self.max_learning_rate / self.warmup_steps * step
        elif step <= self.warmup_steps + self.decay_steps:
            return (self.max_learning_rate / self.decay_steps 
                    * (self.decay_steps + self.warmup_steps - step) 
                    * 0.5 * (1 + np.cos(2 * np.pi * (step - self.warmup_steps) / self.COSINE_WAVE_PERIOD)))
        else:
            return 0.0
    
    def get_config(self):
        config = {
            "max_learning_rate": self.max_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
        }
        return config