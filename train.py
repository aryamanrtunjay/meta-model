import os
import gc
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from autoencoder import VQVAE # Imports the new VQVAE
import argparse
from tqdm import tqdm
import wandb
import numpy as np

# --- Configuration ---
H_DRIVE_BASE = "H:\\meta-model-data"
TRACE_DIR = os.path.join(H_DRIVE_BASE, "traces")
TOKENIZED_DIR = os.path.join(H_DRIVE_BASE, "tokenized")
CODEBOOK_PATH = os.path.join(H_DRIVE_BASE, "codebook.pt")
VOCAB_PATH = os.path.join(H_DRIVE_BASE, "target_vocab.json")
NORM_STATS_PATH = os.path.join(H_DRIVE_BASE, "norm_stats.pt")

MAX_SEQ_LEN = 4096 # fx graphs are much longer. You may need to tune this.

FEATURE_DIM = 19   # We define this new 19-dim vector below

os.makedirs(TOKENIZED_DIR, exist_ok=True)

TARGET_VOCAB = None
TARGET_VOCAB_SIZE = 0

def load_target_vocab():
    """Loads the target vocabulary mapping from disk."""
    global TARGET_VOCAB, TARGET_VOCAB_SIZE
    if TARGET_VOCAB is not None:
        return TARGET_VOCAB
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError("Run preprocess.py to create target_vocab.json first!")
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure keys remain strings and values integers
    TARGET_VOCAB = {k: int(v) for k, v in data.items()}
    TARGET_VOCAB_SIZE = max(len(TARGET_VOCAB), 1)
    return TARGET_VOCAB

def get_shape_vec(shape_list, max_dims=4):
    """Pads a shape list to a fixed length."""
    shape_vec = shape_list + [0] * (max_dims - len(shape_list))
    return shape_vec[:max_dims]

def build_sequence_from_graph(trace, target_vocab=None, vocab_size=None):
    """Build fx graph feature sequence."""
    sequence = []

    if 'graph' not in trace or 'params' not in trace:
        return [([0] * FEATURE_DIM)]

    param_stats = trace['params'].get('parameters', {})
    vocab = target_vocab if target_vocab is not None else load_target_vocab()
    size = vocab_size if vocab_size is not None else max(TARGET_VOCAB_SIZE, len(vocab))
    denom = max(size - 1, 1)

    for node in trace['graph']:
        node_vec = [0] * FEATURE_DIM

        op = node.get('op')
        if op == 'placeholder':
            node_vec[0] = 1
        elif op == 'call_module':
            node_vec[1] = 1
        elif op == 'call_function':
            node_vec[2] = 1
        elif op == 'get_attr':
            node_vec[3] = 1
        elif op == 'output':
            node_vec[4] = 1

        target_str = str(node.get('target', ''))
        idx = vocab.get(target_str, 0)
        node_vec[5] = idx / denom

        if op == 'get_attr' and target_str in param_stats:
            p = param_stats[target_str]
            stats = p.get('statistics', {})
            shape = p.get('shape', [])
            shape_vec = get_shape_vec(shape)

            node_vec[6] = np.log10(p.get('numel', 1) + 1)
            node_vec[7] = stats.get('mean', 0.0)
            node_vec[8] = stats.get('std', 0.0)
            node_vec[9] = stats.get('min', 0.0)
            node_vec[10] = stats.get('max', 0.0)
            node_vec[11] = shape_vec[0]
            node_vec[12] = shape_vec[1]
            node_vec[13] = shape_vec[2]
            node_vec[14] = shape_vec[3]

        tensor_meta = node.get('meta', {}).get('tensor', {})
        out_shape = tensor_meta.get('shape', [])
        out_shape_vec = get_shape_vec(out_shape)

        node_vec[15] = out_shape_vec[0]
        node_vec[16] = out_shape_vec[1]
        node_vec[17] = out_shape_vec[2]
        node_vec[18] = out_shape_vec[3]

        sequence.append(node_vec)

    if not sequence:
        return [([0] * FEATURE_DIM)]

    return sequence

class ArchitectureDataset(Dataset):
    def __init__(self, trace_dir):
        self.trace_dir = trace_dir
        self.traces = [f for f in os.listdir(trace_dir) if f.endswith('.json')]
        load_target_vocab()

        if os.path.exists(NORM_STATS_PATH):
            stats = torch.load(NORM_STATS_PATH, map_location='cpu')
            self.min_vals = stats["min"].float().unsqueeze(0)
            self.max_vals = stats["max"].float().unsqueeze(0)
            self.range = (self.max_vals - self.min_vals) + 1e-6
        else:
            raise FileNotFoundError("Run preprocess.py to create norm_stats.pt first!")
    
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace_path = os.path.join(self.trace_dir, self.traces[idx])
        try:
            with open(trace_path, 'r') as f:
                trace = json.load(f)
        except Exception as e:
            # On corrupt file, just get the next one
            return self.__getitem__((idx + 1) % len(self))
        
        # 1. Get the sequence of feature vectors (e.g., [4096, 19])
        sequence_vectors = build_sequence_from_graph(trace)
        
        # 2. Convert to tensor [L, F]
        tensor = torch.tensor(sequence_vectors, dtype=torch.float)
        tensor = (tensor - self.min_vals) / self.range
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        # 3. Pad or Truncate sequence length (L)
        current_len = tensor.shape[0]
        if current_len > MAX_SEQ_LEN:
            # Truncate
            tensor = tensor[:MAX_SEQ_LEN, :]
        elif current_len < MAX_SEQ_LEN:
            # Pad
            padding = torch.zeros(MAX_SEQ_LEN - current_len, FEATURE_DIM, dtype=torch.float)
            tensor = torch.cat((tensor, padding), dim=0)
        
        # 4. Transpose to [F, L] for the VQ-VAE's Conv1d layers
        # Final shape: [19, 4096]
        tensor = tensor.T
        
        return tensor

def train_vqvae(dataset, epochs=20, batch_size=32, lr=1e-4, device='cuda', num_data_workers=4):
    
    # --- Pass our new FEATURE_DIM to the VQVAE ---
    model = VQVAE(feature_dim=FEATURE_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_workers, pin_memory=True)
    
    wandb.watch(model, log="all", log_freq=100)
    
    for epoch in range(epochs):
        epoch_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        total_loss, total_recon_loss, total_commit_loss = 0, 0, 0
        
        for batch in epoch_bar:
            if batch is None:
                continue
            
            # batch shape is [B, F, L] (e.g., [32, 19, 4096])
            batch = batch.to(device)
            
            recon, z, z_q, indices = model(batch)
            loss, recon_loss, commit_loss = model.loss(batch, recon, z, z_q)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            z_flat = z.permute(0, 2, 1).reshape(-1, model.code_dim)
            model.update_codebook(z_flat.detach(), indices)
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_commit_loss += commit_loss.item()
            
            # Log batch loss to W&B
            wandb.log({"batch_loss": loss.item()})
            
            # Update TQDM postfix
            epoch_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "recon": f"{recon_loss.item():.4f}",
                "commit": f"{commit_loss.item():.4f}"
            })
            
            del batch, recon, z, z_q, indices, z_flat, loss
            gc.collect()
            torch.cuda.empty_cache()
        
        # Log average epoch loss to W&B
        avg_loss = total_loss / len(loader)
        avg_recon = total_recon_loss / len(loader)
        avg_commit = total_commit_loss / len(loader)
        
        wandb.log({
            "epoch": epoch, 
            "avg_loss": avg_loss,
            "avg_recon_loss": avg_recon,
            "avg_commit_loss": avg_commit
        })
        print(f"Epoch {epoch+1} Complete: Avg Loss {avg_loss:.4f} (Recon: {avg_recon:.4f}, Commit: {avg_commit:.4f})")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Training Script")
    parser.add_argument('--epochs', type=int, default=50, 
                        help="Number of epochs to train VQ-VAE.")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument('--data_workers', type=int, default=8,
                        help="Number of DataLoader workers for VQ-VAE training.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate for VQ-VAE training.")
    args = parser.parse_args()
    
    # --- Hyperparameters ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- W&B Initialization ---
    wandb.init(
        project="meta-model-vqvae-v2", # New project name
        config={
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "data_workers": args.data_workers,
            "max_seq_len": MAX_SEQ_LEN,
            "feature_dim": FEATURE_DIM,
            "total_traces": len(os.listdir(TRACE_DIR))
        }
    )

    # --- Train VQ-VAE ---
    dataset = ArchitectureDataset(TRACE_DIR)
    if len(dataset) == 0:
        print(f"No valid traces found in {TRACE_DIR}. Exiting.")
        wandb.finish()
        return
        
    print(f"Starting VQ-VAE training on {device.upper()} with {len(dataset)} traces...")
    vqvae = train_vqvae(dataset, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        lr=args.lr, 
                        device=device, 
                        num_data_workers=args.data_workers)
    
    torch.save(vqvae.state_dict(), CODEBOOK_PATH.replace(".pt", "_model.pt"))
    torch.save(vqvae.codebook, CODEBOOK_PATH)
    print(f"VQ-VAE training complete. Model and codebook saved to H: drive.")
    
    # --- Tokenize and save (Now reads/writes from H: drive) ---
    print("Tokenizing all traces with the new model...")
    vqvae.eval()
    
    # Create a simple dataloader for tokenization
    # We use a batch_size > 1 for GPU efficiency
    token_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.data_workers)
    
    trace_files = dataset.traces # Get the list of files in the correct order
    token_idx = 0

    with torch.no_grad():
        for batch in tqdm(token_loader, desc="Tokenizing"):
            if batch is None:
                continue
            
            batch = batch.to(device)
            # Get the indices from the VQ-VAE
            # indices shape will be [B, L_down] (e.g., [64, 64])
            _, _, _, indices = vqvae(batch)
            
            # Move to CPU and convert to list
            indices_list = indices.cpu().tolist()
            
            # Save each file in the batch
            for token_sequence in indices_list:
                if token_idx >= len(trace_files):
                    break # Should not happen, but safeguard
                
                trace_file = trace_files[token_idx]
                tok_path = os.path.join(TOKENIZED_DIR, trace_file.replace('trace_', 'tok_'))
                
                # We don't need 'gaps' anymore since all sequences are 64 tokens
                tokenized = {'tokens': token_sequence}
                
                with open(tok_path, 'w') as f:
                    json.dump(tokenized, f)
                
                token_idx += 1

    wandb.finish()
    print("All steps complete. Data is saved on H: drive.")

if __name__ == "__main__":
    main()