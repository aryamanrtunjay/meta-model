import neat
import torch
import torch.nn.functional as F
from pytorch_neat.recurrent_net import RecurrentNet  # From PyTorch-NEAT
import numpy as np
import os
from tqdm import tqdm

class CustomTqdmReporter(neat.reporting.BaseReporter):
    def __init__(self, generations):
        self.generation = 0
        self.generations = generations
        self.pbar = tqdm(total=generations, desc="Generations progress", position=1, leave=False, ncols=100)
        self.current_best_fitness = None
        self.avg_fitness = None
        self.stdev_fitness = None
        self.species_count = None
        self.mean_distance = None

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        self.current_best_fitness = best_genome.fitness if best_genome.fitness is not None else "N/A"
        fitnesses = [g.fitness for g in population.values() if g.fitness is not None]
        self.avg_fitness = np.mean(fitnesses) if fitnesses else "N/A"
        self.stdev_fitness = np.std(fitnesses) if fitnesses else "N/A"
        self.species_count = len(species.species)
        
        # Calculate mean genetic distance (from StatisticsReporter logic)
        distances = []
        for s in species.species.values():
            for ind1 in s.members.values():
                for ind2 in s.members.values():
                    if id(ind1) < id(ind2):
                        distances.append(ind1.distance(ind2, config.genome_config, config.genome_config))
        self.mean_distance = np.mean(distances) if distances else "N/A"
        
        self.pbar.set_postfix({
            'Avg Fit': f"{self.avg_fitness:.2f}",
            'Stdev': f"{self.stdev_fitness:.2f}",
            'Best Fit': f"{self.current_best_fitness:.2f}",
            'Species': self.species_count,
            'Dist': f"{self.mean_distance:.2f}"
        })
        self.pbar.update(1)

    def complete_extinction(self):
        self.pbar.close()

    def end(self):
        self.pbar.close()

class NeatMLPTrainer:
    def __init__(self, config_path, X_train, y_train, X_val=None, y_val=None, device='cpu', batch_size=512):
        """
        Initialize the trainer.
        - config_path: Path to NEAT config file (example provided below).
        - X_train, y_train: Training data (np.array or torch.Tensor).
        - X_val, y_val: Optional validation data for early stopping or monitoring.
        - device: 'cpu' or 'cuda'.
        - batch_size: For batched evaluation in fitness function.
        """
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        # Expose a NEAT Population so callers can attach reporters (e.g., tqdm) before training.
        self.p = neat.Population(self.config)
        self.device = device
        # Prefer bfloat16 on CUDA for speed/throughput; fall back to float32 on CPU
        self.dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
        self.batch_size = batch_size
        
        # Convert to Torch tensors if not already
        self.X_train = torch.as_tensor(X_train, dtype=self.dtype, device=device)
        self.y_train = torch.as_tensor(y_train, dtype=self.dtype, device=device)
        
        if X_val is not None and y_val is not None:
            self.X_val = torch.as_tensor(X_val, dtype=self.dtype, device=device)
            self.y_val = torch.as_tensor(y_val, dtype=self.dtype, device=device)
        else:
            self.X_val = None
            self.y_val = None
        
        self.input_size = self.X_train.shape[1]
        self.output_size = self.y_train.shape[1] if len(self.y_train.shape) > 1 else 1
        self.config.genome_config.input_keys = [-i-1 for i in range(self.input_size)]  # Adjust inputs
        self.config.genome_config.output_keys = [i for i in range(self.output_size)]  # Adjust outputs

    def eval_genome(self, genome, config):
        """
        Evaluate a genome's fitness using a PyTorch network.
        Fitness = negative loss (e.g., BCE for binary classification) on training data.
        """
        # Create PyTorch recurrent net from genome (can handle feedforward too)
        net = RecurrentNet.create(genome, config, batch_size=self.batch_size, device=self.device, dtype=self.dtype)
        
        # Forward pass in batches to handle large datasets
        losses = []
        with torch.no_grad():
            for i in range(0, len(self.X_train), self.batch_size):
                batch_X = self.X_train[i:i+self.batch_size]
                batch_y = self.y_train[i:i+self.batch_size]
                outputs = net.activate(batch_X)  # Returns torch tensor
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y)  # For binary; change for multi-class
                losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        
        # Optional: Add validation penalty for generalization
        if self.X_val is not None:
            val_losses = []
            for i in range(0, len(self.X_val), self.batch_size):
                batch_X = self.X_val[i:i+self.batch_size]
                batch_y = self.y_val[i:i+self.batch_size]
                outputs = net.activate(batch_X)
                loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
                val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
            fitness = - (avg_loss + 0.5 * avg_val_loss)  # Penalize overfitting
        else:
            fitness = -avg_loss
        
        return fitness

    def train(self, generations=100, hybrid_backprop=False, lr=0.001, backprop_epochs=10):
        """
        Run NEAT evolution.
        - generations: Number of generations to evolve.
        - hybrid_backprop: If True, after evolution, fine-tune winner with backprop.
        - lr: Learning rate for backprop.
        - backprop_epochs: Epochs for fine-tuning.
        Returns the best genome and (if hybrid) the fine-tuned net.
        """
        # Attach default reporters here (stats); callers may have already added their own via self.p
        self.p.add_reporter(neat.StatisticsReporter())

        # Run evolution
        winner = self.p.run(self._eval_genomes, generations)

        if hybrid_backprop:
            # Attempt fine-tuning only if the created network exposes trainable parameters (nn.Module-like).
            net = RecurrentNet.create(winner, self.config, batch_size=self.batch_size, device=self.device, dtype=self.dtype)
            can_train = hasattr(net, 'parameters')
            try:
                can_train = can_train and len(list(net.parameters())) > 0  # type: ignore[attr-defined]
            except Exception:
                # If parameters() is not iterable or not implemented
                can_train = False

            if not can_train:
                # PyTorch-NEAT RecurrentNet isn't an nn.Module; skip hybrid backprop gracefully.
                print("[NeatMLPTrainer] Skipping hybrid backprop: RecurrentNet has no trainable parameters.")
                return winner, None

            optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # type: ignore[attr-defined]

            for epoch in range(backprop_epochs):
                for i in range(0, len(self.X_train), self.batch_size):
                    batch_X = self.X_train[i:i+self.batch_size]
                    batch_y = self.y_train[i:i+self.batch_size]
                    outputs = net.activate(batch_X)
                    loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            return winner, net
        else:
            return winner, None

    def _eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = self.eval_genome(genome, config)