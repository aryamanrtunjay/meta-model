import os
import logging

# === SILENCE ALL EXTERNAL LIBRARIES ===
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)

import gc
import json
import torch
from compiler import ModelTracer
import shutil
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial
from huggingface_hub.utils import HfHubHTTPError

# --- Globals for Workers ---
tracer = None

def init_worker(models_dir, params_dir, hf_cache_base):
    """Initializer for each worker process."""
    global tracer
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "critical"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    pid = os.getpid()
    worker_cache_dir = os.path.join(hf_cache_base, f"worker_{pid}")
    os.environ["HF_HOME"] = worker_cache_dir
    import logging
    logging.getLogger("transformers").setLevel(logging.CRITICAL)
    logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
    shutil.rmtree(worker_cache_dir, ignore_errors=True)
    tracer = ModelTracer(
        models_dir=models_dir, 
        params_dir=params_dir,
        hf_cache_dir=worker_cache_dir
    )

def process_model(model_name: str, trace_dir: str):
    """Traces a single model in a separate process."""
    global tracer
    if tracer is None:
        return (model_name, "error (worker not init)")

    safe_name = model_name.replace('/', '_').replace(':', '_')
    trace_path = os.path.join(trace_dir, f'trace_{safe_name}.json')
    local_model_path = os.path.join(tracer.models_dir, safe_name)
    local_params_path = os.path.join(tracer.params_dir, safe_name)
        
    try:
        from transformers import AutoModel
        model = tracer.load_model(model_name, AutoModel) 
        trace_data = tracer.trace(model, model_name)
        
        with open(trace_path, 'w') as f:
            json.dump(trace_data, f, indent=4)
        
        del model
        del trace_data
        
        return (model_name, "success")

    except HfHubHTTPError as e:
        # Handle 401/404 errors (gated or missing models)
        with open("trace_errors.log", "a") as log_f:
            log_f.write(f"Access error (401/404) tracing {model_name}: {e}\n")
        return (model_name, "access_error")
    except Exception as e:
        with open("trace_errors.log", "a") as log_f:
            log_f.write(f"Error tracing {model_name}: {e}\n")
        return (model_name, "error")

    finally:
        # Clean up all temp files for this model
        shutil.rmtree(local_model_path, ignore_errors=True)
        shutil.rmtree(local_params_path, ignore_errors=True)
        shutil.rmtree(tracer.hf_cache_dir, ignore_errors=True)
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Parallel Model Tracing Script")
    parser.add_argument('--mode', type=str, default='full', choices=['test', 'full'], 
                        help="Run mode: 'test' for a quick check, 'full' for the complete run.")
    parser.add_argument('--trace_workers', type=int, default=8,
                        help="Number of parallel processes for tracing.")
    args = parser.parse_args()

    # === H: DRIVE PATHS ===
    H_DRIVE_BASE = "H:\\meta-model-data"
    MODELS_DIR = os.path.join(H_DRIVE_BASE, "models")
    PARAMS_DIR = os.path.join(H_DRIVE_BASE, "trace_params")
    HF_CACHE_BASE = os.path.join(H_DRIVE_BASE, "hf_cache")
    trace_dir = os.path.join(H_DRIVE_BASE, "traces")
    
    for path in [MODELS_DIR, PARAMS_DIR, HF_CACHE_BASE, trace_dir]:
        os.makedirs(path, exist_ok=True)
    
    # --- Load Model List from File (from G: drive) ---
    model_list_file = "model_list.txt"
    if not os.path.exists(model_list_file):
        print(f"Error: {model_list_file} not found.")
        return

    with open(model_list_file, "r") as f:
        all_model_names = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(all_model_names)} models from {model_list_file}.")
    
    # --- PRE-FILTERING STEP ---
    print("Scanning for existing traces...")
    try:
        existing_traces = set(os.listdir(trace_dir))
    except FileNotFoundError:
        existing_traces = set()

    models_to_trace = []
    completed_count = 0

    for model_name in all_model_names:
        safe_name = model_name.replace('/', '_').replace(':', '_')
        trace_filename = f"trace_{safe_name}.json"
        
        if trace_filename in existing_traces:
            completed_count += 1
        else:
            models_to_trace.append(model_name)
    
    print(f"Found {completed_count} existing traces. {len(models_to_trace)} models remaining.")
    
    # Handle test mode
    if args.mode == 'test':
        models_to_trace = models_to_trace[:3]
        num_trace_workers = 1
    else:
        num_trace_workers = args.trace_workers
    
    if not models_to_trace:
        print("All models are already traced. Proceeding to training.")
    else:
        print(f"Starting parallel tracing with {num_trace_workers} worker(s)...")
        print(f"All data will be written to {H_DRIVE_BASE}")
        print("Tracing workers will use CPU-only to avoid VRAM conflicts.")

        ctx = multiprocessing.get_context('spawn') 
        
        worker_init_func = partial(init_worker, 
                                   models_dir=MODELS_DIR, 
                                   params_dir=PARAMS_DIR, 
                                   hf_cache_base=HF_CACHE_BASE)
        
        task_func = partial(process_model, trace_dir=trace_dir)

        with ctx.Pool(processes=num_trace_workers, initializer=worker_init_func) as pool:
            pbar = tqdm(total=len(all_model_names), desc="Tracing Models", initial=completed_count)
            
            for model_name, status in pool.imap_unordered(task_func, models_to_trace):
                if status not in ["success", "skipped"]:
                    pbar.set_postfix_str(f"Error on {model_name[:30]}... ({status})")
                pbar.update(1)
                
            pbar.close()

    print("Model tracing complete. You can now run train_vqvae.py")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()