import csv
import os
import random
import math
import pickle
import numpy as np
import argparse
import torch
import torch.nn.functional as F
import multiprocessing
import time
from tqdm import tqdm
import neat

from neat_mlp_trainer import NeatMLPTrainer
from data_gen import DataGenerator as DataGen

TASK_TEMPLATES = [
    {
        'template': 'divisible_by_k',
        'param_gen': lambda: {'k': random.randint(2, 100000)},
        'est_params': 2113,
        'file_stem_gen': lambda p: f"div_by_{p['k']}"
    },
    {
        'template': 'mod_equals',
        'param_gen': lambda: {'k': random.randint(2, 256), 'r': None},  # r set later
        'est_params': 2113,
        'file_stem_gen': lambda p: f"mod_{p['k']}_eq_{p['r']}"
    },
    {
        'template': 'trailing_zeros_ge',
        'param_gen': lambda: {'t': random.randint(1, 32)},
        'est_params': 65,
        'file_stem_gen': lambda p: f"tz_ge_{p['t']}"
    },
    {
        'template': 'popcount_mod_equals',
        'param_gen': lambda: {'m': random.randint(2, 16), 's': None},
        'est_params': 4225,
        'file_stem_gen': lambda p: f"popcount_mod_{p['m']}_eq_{p['s']}"
    },
    {
        'template': 'binary_palindrome',
        'param_gen': lambda: {},
        'est_params': 2113,
        'file_stem_gen': lambda p: "bin_palindrome"
    },
    {
        'template': 'highest_set_bit_eq',
        'param_gen': lambda: {'h': random.randint(0, 63)},
        'est_params': 65,
        'file_stem_gen': lambda p: f"msb_eq_{p['h']}"
    },
    {
        'template': 'lowest_set_bit_eq',
        'param_gen': lambda: {'l': random.randint(0, 63)},
        'est_params': 65,
        'file_stem_gen': lambda p: f"lsb_eq_{p['l']}"
    },
    {
        'template': 'perfect_kth_power',
        'param_gen': lambda: {'k': random.randint(2, 6)},
        'est_params': 16641,
        'file_stem_gen': lambda p: f"perfect_{p['k']}th_power"
    },
    {
        'template': 'mersenne_number',
        'param_gen': lambda: {},
        'est_params': 4225,
        'file_stem_gen': lambda p: "mersenne"
    },
    {
        'template': 'lsb_all_ones',
        'param_gen': lambda: {'length': random.randint(1, 64)},
        'est_params': 2113,
        'file_stem_gen': lambda p: f"lsb_all_ones_len_{p['length']}"
    },
    {
        'template': 'xor_of_all_bits_equals',
        'param_gen': lambda: {'v': random.choice([0, 1])},
        'est_params': 4225,
        'file_stem_gen': lambda p: f"xor_eq_{p['v']}"
    },
    {
        'template': 'leading_zeros_ge',
        'param_gen': lambda: {'lz': random.randint(1, 63)},
        'est_params': 65,
        'file_stem_gen': lambda p: f"lz_ge_{p['lz']}"
    },
    # Removed tasks: exact_popcount, perfect_kth_power, upper_half_popcount_eq, consecutive_ones_ge, is_prime, is_fibonacci
]

def sample_task():
    t = random.choice(TASK_TEMPLATES)
    params = t['param_gen']()
    if 'r' in params and params['r'] is None:
        params['r'] = random.randint(0, params['k'] - 1)
    if 's' in params and params['s'] is None:
        params['s'] = random.randint(0, params['m'] - 1)
    file_stem = t['file_stem_gen'](params)
    est_params = t['est_params']
    complexity = 'easy' if est_params < 100 else 'medium' if est_params < 10000 else 'hard'
    return t['template'], params, file_stem, est_params, complexity

def load_csv_to_xy(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # header
        X = []
        y = []
        for row in reader:
            n = int(row[0])
            label = int(row[1])
            bin_vec = [(n >> i) & 1 for i in range(64)]  # LSB first
            X.append(bin_vec)
            y.append([label])
    return np.array(X), np.array(y)

class TqdmReporter(neat.reporting.BaseReporter):
    def __init__(self, generations):
        self.pbar = tqdm(total=generations, desc="Generations progress", position=1, leave=False)

    def end_generation(self, config, population, species_set):
        self.pbar.update(1)

    def complete_extinction(self):
        self.pbar.close()

def train_task(task_args):
    task_id, is_test, config_path, models_dir, device = task_args
    random.seed(time.time() + os.getpid())  # Seed per process

    template, params, file_stem, est_params, complexity = sample_task()
    print(f"Task {task_id+1}: {template} {params}")

    # Data size and hyperparams
    if is_test:
        num_samples = 1000
        generations = 10
        hybrid = False
        backprop_epochs = 0
    else:
        if complexity == 'easy':
            num_samples = 5000
            generations = 20
        elif complexity == 'medium':
            num_samples = 50000
            generations = 50
        else:
            num_samples = 500000
            generations = 100
        hybrid = True
        backprop_epochs = 10

    # Generate data
    getattr(DataGen, template)(**params, num_samples=num_samples)

    # Load
    X_train, y_train = load_csv_to_xy(f"data/train/{file_stem}.csv")
    X_val, y_val = load_csv_to_xy(f"data/val/{file_stem}.csv")

    # Train
    trainer = NeatMLPTrainer(config_path, X_train, y_train, X_val, y_val, device=device)
    trainer.p.add_reporter(TqdmReporter(generations))  # Add per-task progress bar

    winner, tuned_net = trainer.train(generations=generations, hybrid_backprop=hybrid, backprop_epochs=backprop_epochs)

    # Save
    task_dir = os.path.join(models_dir, file_stem)
    os.makedirs(task_dir, exist_ok=True)
    pickle.dump(winner, open(os.path.join(task_dir, 'winner.pkl'), 'wb'))
    pickle.dump(trainer.config, open(os.path.join(task_dir, 'config.pkl'), 'wb'))
    if tuned_net:
        torch.save(tuned_net, os.path.join(task_dir, 'tuned_net.pt'))  # or state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['test', 'full'], required=True)
    parser.add_argument('--num_processes', type=int, default=8, help="Number of parallel processes")
    args = parser.parse_args()

    is_test = args.mode == 'test'
    num_tasks = 5 if is_test else 2000
    models_dir = 'models/'
    os.makedirs(models_dir, exist_ok=True)
    config_path = 'neat_config.txt'  # Assume exists

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    task_inputs = [(i, is_test, config_path, models_dir, device) for i in range(num_tasks)]

    with multiprocessing.Pool(processes=args.num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(train_task, task_inputs), total=num_tasks, desc="Overall models progress", position=0):
            pass