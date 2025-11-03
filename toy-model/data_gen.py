import random
import os
import csv
import math

TRAIN_PATH = "data/train/"
VAL_PATH = "data/val/"

class DataGenerator:
    def __init__(self):
        os.makedirs(TRAIN_PATH, exist_ok=True)
        os.makedirs(VAL_PATH, exist_ok=True)

    @staticmethod
    def divisible_by_k(k, num_samples, train_split=0.8):
        if k <= 0:
            raise ValueError("k must be positive integer > 0")
        
        train_file_path = os.path.join(TRAIN_PATH, f"div_by_{k}.csv")
        val_file_path = os.path.join(VAL_PATH, f"div_by_{k}.csv")
        
        # Generate equal positives and negatives
        max_n = (1 << 64) - 1  # 2^64 - 1
        positives = []
        negatives = []
        
        # Positives: random multiples within range
        max_multiple = max_n // k
        for _ in range(math.ceil(num_samples / 2)):
            multiple = random.randint(0, max_multiple)
            positives.append((multiple * k, 1))
        
        # Negatives: random non-multiples
        for _ in range(math.ceil(num_samples / 2)):
            n = random.randint(0, max_n)
            while n % k == 0:
                n = random.randint(0, max_n)
            negatives.append((n, 0))
        
        # Combine, shuffle, split
        all_samples = positives + negatives
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_split)
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        
        # Write with headers
        with open(train_file_path, 'w', newline='') as train_f:
            train_writer = csv.writer(train_f)
            train_writer.writerow(['number', 'label'])
            train_writer.writerows(train_samples)
        
        with open(val_file_path, 'w', newline='') as val_f:
            val_writer = csv.writer(val_f)
            val_writer.writerow(['number', 'label'])
            val_writer.writerows(val_samples)