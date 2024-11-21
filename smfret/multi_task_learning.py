"""Simulated smFRET traces dataset classes for multi-task learning."""

import numpy as np
from tqdm.auto import tqdm
import multiprocessing

import smfret.dataset
from smfret.dataset import MatlabTraceSet
from smfret.dataset import FRETTraceSet
from smfret.dataset import FRETTrace
from smfret.trace_simulator import Simulator
from smfret.trace_simulator import SingleChannelSimulator
from smfret.trace_simulator import ParameterGenerator
from smfret.trace_simulator import SimulatedTraceSet

import tensorflow as tf
from datetime import datetime as tm

rng = np.random.default_rng()

class EvolvingTraceSet(SimulatedTraceSet):
    """Class for trace sets capable of replacing X% traces with new traces."""
    def __init__(self, size, params_gen):
        super().__init__(None)
        self.size = 0
        self.params_gen = params_gen
        self.traces = []
        self.populate(size)
        self.vectorize_traces()
        self.is_labeled = True
    
    def populate(self, n: int):
        """Populates the dataset using the simulator."""
        new_traces = []
        try:
            pos = multiprocessing.current_process()._identity[0] - 1
            disable = False
        except:
            disable = True
            pos = 0
            
        desc_text = self.__class__.__name__
        for _ in tqdm(range(n), desc=desc_text, position=pos, disable=disable):
            new_traces.append(self.trace_gen())
        self.traces += new_traces
        self.size += n

    def delete(self, n: int):
        """Removes the traces on the top of the traces list."""
        self.traces = self.traces[n:]
        self.size -= n
        
    def evolve(self, n):
        """Adds new traces and removes old traces."""
        self.populate(n)
        self.delete(n)
        self.vectorize_traces()

    def vectorize_traces(self):
        """Stacks traces signals into np vectors."""
        self.time = self.traces[0].time
        self.donor = np.stack([trace.donor for trace in self.traces], axis=0)
        self.acceptor = np.stack([trace.acceptor for trace in self.traces], axis=0)
        self.label = np.stack([trace.label for trace in self.traces], axis=0)
    
    def trace_gen(self):
        """Generates a single labeled trace."""
        pass
    
    def save(self, file):
        """Saves the dataset to file."""
        self.vectorize_traces()
        data = {
            'time': self.time,
            'donor': self.donor,
            'acceptor': self.acceptor,
            'label': self.label,
            'size': self.size
        }
        np.savez(file, **data)
    

class SavedTraceSet(EvolvingTraceSet):
    """Class for trace sets saved on disk."""
    def __init__(self, size, file):
        self.file = file
        self.counter = 0
        self.epochs = 0
        self.cached_traces_data = None
        self.load_file_to_cache()
        super().__init__(size=size, params_gen=None)
        
    def resize(self, size):
        """Changes the size of the dataset."""
        self.counter = 0
        self.epochs = 0
        super().__init__(size, params_gen=None)
        
    def load_file_to_cache(self):
        """Reads the saved traces."""
        self.cached_traces_data = dict(np.load(self.file))
        total = self.cached_traces_data['donor'] + self.cached_traces_data['acceptor']
        max_total = np.expand_dims(total.max(axis=-1), axis=-1)
        self.cached_stacked = tf.cast(np.stack([
            self.cached_traces_data['donor'] / max_total,
            self.cached_traces_data['acceptor'] / max_total,
            total / max_total,
            np.repeat([self.cached_traces_data['time']], self.cached_traces_data['size'], axis=0) / smfret.dataset.TIME_DENOMINATOR
        ], axis=-1), tf.bfloat16)

        # reduce the memory footprint of cached_traces_data
        for key in self.cached_traces_data:
            if self.cached_traces_data[key].dtype == np.float64:
                self.cached_traces_data[key] = self.cached_traces_data[key].astype(np.float32, casting='same_kind')
        
    
    def trace_gen(self):
        trace = FRETTrace(
            donor=self.cached_traces_data['donor'][self.counter, :],
            acceptor=self.cached_traces_data['acceptor'][self.counter, :],
            time=self.cached_traces_data['time'],
            label=self.cached_traces_data['label'][self.counter, ...],
        )
        trace.counter = self.counter
        
        self.counter += 1
        if self.counter == self.cached_traces_data['size']:
            self.counter = 0
            self.epochs += 1
            
        return trace
    
    def vectorize_traces(self):
        """Overwrites the default behavior of vectorize_traces to gain performance."""
        with tf.device('/CPU:0'):
            self.time = self.traces[0].time
            counters = [trace.counter for trace in self.traces]
            self.donor = self.cached_traces_data['donor'][counters, :]
            self.acceptor = self.cached_traces_data['acceptor'][counters, :]
            self.label = self.cached_traces_data['label'][counters, ...]
            self.stacked = tf.gather(self.cached_stacked, counters)
        
    
    def to_tensor(self, size=None, normalize=True):
        """Converts the smFRET data into the form for training TensorFlow models."""
        with tf.device('/CPU:0'):
            return tf.cast(self.stacked, tf.float32)


class FRETStateTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with FRET states."""
    fret_states_resolution = 0.05
    lifetime_min = 500
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        trace.label = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        return trace
    

class FRETStateCountTraceSet(EvolvingTraceSet):
    """Class for trace sets labeled with the number of FRET states."""
    fret_states_resolution = 0.1
    lifetime_min = 500
    quanta = 1
    lower_limit = 1
    upper_limit = 4
    quantize_bins = np.arange(lower_limit, upper_limit, quanta)
    
    def trace_gen(self):
        trace = None
        while trace is None or trace.acceptor_lifetime <= self.lifetime_min:
            params = self.params_gen.generate()
            trace = Simulator(params).generate()

        states = np.digitize(trace.fret_ideal, np.arange(0, 1, self.fret_states_resolution), right=True)
        trace.label = np.digitize(np.unique(states).size, self.quantize_bins, right=True)
        return trace