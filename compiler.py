import os
import sys
import json
import hashlib
import types
import importlib.util
import importlib.machinery as _machinery
from enum import Enum
from collections.abc import Mapping, Sequence
from typing import List, Tuple, Any, Dict, Optional, Type

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION_IMPORT", "1")

def _ensure_torchvision_stub() -> None:
    """Provide minimal torchvision stub minimal when native package is unavailable."""
    if "torchvision" in sys.modules:
        return
    try:
        __import__("torchvision")
        return
    except Exception:
        sys.modules.pop("torchvision", None)
        sys.modules.pop("torchvision.transforms", None)
        sys.modules.pop("torchvision.io", None)
        sys.modules.pop("torchvision.ops", None)
        sys.modules.pop("torchvision.datasets", None)
        sys.modules.pop("torchvision.models", None)
        sys.modules.pop("torchvision.utils", None)
        vision_stub = types.ModuleType("torchvision")
        transforms_stub = types.ModuleType("torchvision.transforms")
        io_stub = types.ModuleType("torchvision.io")
        ops_stub = types.ModuleType("torchvision.ops")
        datasets_stub = types.ModuleType("torchvision.datasets")
        models_stub = types.ModuleType("torchvision.models")
        utils_stub = types.ModuleType("torchvision.utils")

        class InterpolationMode(Enum):
            NEAREST = "nearest"
            NEAREST_EXACT = "nearest-exact"
            BILINEAR = "bilinear"
            BICUBIC = "bicubic"
            BOX = "box"
            HAMMING = "hamming"
            LANCZOS = "lanczos"

        def _not_available(*_args, **_kwargs):  # pragma: no cover - stub safeguard
            raise RuntimeError("torchvision operations are unavailable in this runtime")

        transforms_stub.InterpolationMode = InterpolationMode
        io_stub.read_video = _not_available
        io_stub.write_video = _not_available
        ops_stub.nms = _not_available
        utils_stub.save_image = _not_available

        vision_stub.__spec__ = _machinery.ModuleSpec("torchvision", loader=None)
        transforms_stub.__spec__ = _machinery.ModuleSpec("torchvision.transforms", loader=None)
        io_stub.__spec__ = _machinery.ModuleSpec("torchvision.io", loader=None)
        ops_stub.__spec__ = _machinery.ModuleSpec("torchvision.ops", loader=None)
        datasets_stub.__spec__ = _machinery.ModuleSpec("torchvision.datasets", loader=None)
        models_stub.__spec__ = _machinery.ModuleSpec("torchvision.models", loader=None)
        utils_stub.__spec__ = _machinery.ModuleSpec("torchvision.utils", loader=None)
        vision_stub.__path__ = []
        vision_stub.transforms = transforms_stub
        vision_stub.io = io_stub
        vision_stub.ops = ops_stub
        vision_stub.datasets = datasets_stub
        vision_stub.models = models_stub
        vision_stub.utils = utils_stub

        sys.modules["torchvision"] = vision_stub
        sys.modules["torchvision.transforms"] = transforms_stub
        sys.modules["torchvision.io"] = io_stub
        sys.modules["torchvision.ops"] = ops_stub
        sys.modules["torchvision.datasets"] = datasets_stub
        sys.modules["torchvision.models"] = models_stub
        sys.modules["torchvision.utils"] = utils_stub

_ensure_torchvision_stub()

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import ShapeProp
from transformers.modeling_outputs import ModelOutput
from transformers import AutoModel, AutoModelForCausalLM

class ModelTracer:
    def __init__(self, models_dir: str, params_dir: str, hf_cache_dir: str):
        self.base_dir = "." # Set a default, though it's less used now
        self.models_dir = models_dir
        self.params_dir = params_dir
        self.hf_cache_dir = hf_cache_dir

        # Ensure all H: drive paths exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.hf_cache_dir, exist_ok=True)

    def load_model(self, model_name: str, model_class: Type[torch.nn.Module], save: bool = True) -> torch.nn.Module:
        safe_name = model_name.replace('/', '_').replace(':', '_')
        save_path = os.path.join(self.models_dir, safe_name)

        load_kwargs: Dict[str, Any] = {}
        load_kwargs["cache_dir"] = self.hf_cache_dir

        load_kwargs: Dict[str, Any] = {}
        model_lower = model_name.lower()
        if "llama" in model_lower:
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            dtype = torch.float16 if (torch.cuda.is_available() or has_mps) else torch.float32
            load_kwargs["torch_dtype"] = dtype
            load_kwargs["trust_remote_code"] = True
            if importlib.util.find_spec("accelerate") is not None:
                load_kwargs["low_cpu_mem_usage"] = True

        def _attempt_load(self, target_class: Type[torch.nn.Module], source: str, extra_kwargs: Dict[str, Any], *, allow_token: bool) -> torch.nn.Module:
            # --- Import is REMOVED from here ---
            if allow_token:
                return target_class.from_pretrained(source, token=os.environ.get('HF_TOKEN'), **extra_kwargs)
            return target_class.from_pretrained(source, **extra_kwargs)

        if os.path.exists(save_path):
            try:
                return _attempt_load(self, model_class, save_path, load_kwargs, allow_token=False)
            except ValueError as e:
                if "Unrecognized model" in str(e):
                    # This line will now work correctly
                    return _attempt_load(self, AutoModelForCausalLM, save_path, load_kwargs, allow_token=False)
                raise

        try:
            model = _attempt_load(self, model_class, model_name, load_kwargs, allow_token=True)
        except ValueError as e:
            if "Unrecognized model" in str(e):
                # This line will also work correctly
                model = _attempt_load(self, AutoModelForCausalLM, model_name, load_kwargs, allow_token=True)
            else:
                raise e
        if save:
            model.save_pretrained(save_path)
        return model

    def get_dummy_input(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """Generate minimal dummy inputs based on model config."""
        if not hasattr(model, 'config'):
            raise ValueError("Model has no config attribute")
        
        config = model.config
        model_type = getattr(config, 'model_type', '').lower()
        
        # --- NEW LOGIC ---
        # We must check for specific model types *first* before
        # checking for generic attributes like 'vocab_size',
        # as some vision models (like BEiT) also have a vocab_size.

        # Priority 1: Multimodal models (e.g., CLIP)
        if model_type == 'clip':
            return {
                "input_ids": torch.randint(0, getattr(config, 'vocab_size', 49408), (1, 77)),
                "attention_mask": torch.ones(1, 77, dtype=torch.long),
                "pixel_values": torch.randn(1, 3, getattr(config, 'image_size', 224), getattr(config, 'image_size', 224))
            }

        # Priority 2: Pure Vision models
        # === UPDATED THIS LIST ===
        vision_model_types = [
            'resnet', 'vit', 'convnext', 'detr', 'yolos', 'beit',
            'swin', 'segformer', 'maskformer', 'mask2former', 'yolos',
            'dpt', 'glpn', 'upernet'
        ]
        if model_type in vision_model_types or hasattr(config, 'image_size'):
            img_size = getattr(config, 'image_size', 224)
            # Handle cases where image_size might be a tuple
            if isinstance(img_size, (tuple, list)):
                img_size = img_size[0]
            num_channels = getattr(config, 'num_channels', 3)
            return {"pixel_values": torch.randn(1, num_channels, img_size, img_size)}

        # Priority 3: Pure Text/NLP models
        if hasattr(config, 'vocab_size'):
            vocab_size = config.vocab_size
            max_seq_len = getattr(config, 'max_position_embeddings', 128)
            offset = 2 if getattr(config, 'model_type', '') == 'roberta' else 0
            seq_len = min(128, max_seq_len - offset)
            dummy_inputs = {
                "input_ids": torch.randint(0, vocab_size, (1, seq_len)),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long)
            }
            if getattr(config, 'type_vocab_size', 0) > 0:
                dummy_inputs["token_type_ids"] = torch.zeros(1, seq_len, dtype=torch.long)
            return dummy_inputs
            
        # Priority 4: Diffusion-like (e.g., UNet)
        elif hasattr(config, 'sample_size') and hasattr(config, 'in_channels'):
            sample_size = getattr(config, 'sample_size', 64)
            in_channels = getattr(config, 'in_channels', 4)
            cross_attention_dim = getattr(config, 'cross_attention_dim', 768)
            return {
                "sample": torch.randn(1, in_channels, sample_size, sample_size),
                "timestep": torch.tensor([1], dtype=torch.long),
                "encoder_hidden_states": torch.randn(1, 77, cross_attention_dim)
            }
            
        else:
            raise ValueError(f"Unsupported model config for dummy input generation: {config}")

    def build_hierarchy(self, model: torch.nn.Module, name: str = "root") -> Dict[str, Any]:
        path = name
        hierarchy = {
            "name": name,
            "path": path,
            "class": model.__class__.__name__,
            "repr": repr(model),  # Includes hyperparameters like kernel_size, num_heads, etc.
            "children": []
        }
        for child_name, submodule in model.named_children():
            child_path = f"{path}.{child_name}" if path != "root" else child_name
            child_hier = self.build_hierarchy(submodule, child_path)
            hierarchy["children"].append(child_hier)
        return hierarchy

    def serialize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        tensor_cpu = tensor.detach().cpu()
        meta: Dict[str, Any] = {
            "type": "tensor",
            "dtype": str(tensor_cpu.dtype),
            "shape": list(tensor_cpu.shape),
            "device": str(tensor.device),
            "requires_grad": bool(tensor.requires_grad),
            "numel": tensor_cpu.numel()
        }
        if tensor_cpu.numel() > 0:
            try:
                sample_tensor = tensor_cpu.flatten()[:16]
                meta["sample_values"] = sample_tensor.tolist()
            except Exception:
                meta["sample_values"] = "unavailable"

            try:
                stats_source = tensor_cpu.dequantize() if tensor_cpu.is_quantized else tensor_cpu.float()
                stats = {
                    "mean": float(stats_source.mean().item()),
                    "std": float(stats_source.std(unbiased=False).item()) if tensor_cpu.numel() > 1 else 0.0,
                    "min": float(stats_source.min().item()),
                    "max": float(stats_source.max().item())
                }
                meta["statistics"] = stats
            except Exception:
                meta["statistics"] = "unavailable"
        return meta

    def serialize_item(self, item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            return self.serialize_tensor(item)
        if isinstance(item, ModelOutput):
            return {key: self.serialize_item(value) for key, value in item.items()}
        if isinstance(item, Mapping):
            return {str(key): self.serialize_item(value) for key, value in item.items()}
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [self.serialize_item(value) for value in item]
        if item is None:
            return None
        if isinstance(item, torch.dtype):
            return str(item)
        if isinstance(item, torch.Size):
            return list(item)
        return {
            "type": type(item).__name__,
            "repr": repr(item)
        }

    def extract_module_hyperparams(self, module: torch.nn.Module) -> Dict[str, Any]:
        hyperparams: Dict[str, Any] = {}
        common_attrs = [
            "kernel_size",
            "stride",
            "padding",
            "dilation",
            "groups",
            "bias",
            "in_features",
            "out_features",
            "eps",
            "momentum",
            "num_heads",
            "head_dim",
            "hidden_size",
            "intermediate_size",
            "dropout",
            "activation",
            "layer_norm_eps",
            "max_position_embeddings"
        ]
        for attr in common_attrs:
            if hasattr(module, attr):
                value = getattr(module, attr)
                if isinstance(value, torch.nn.Parameter):
                    hyperparams[attr] = {
                        "type": "parameter",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype)
                    }
                elif isinstance(value, torch.Tensor):
                    hyperparams[attr] = self.serialize_tensor(value)
                elif isinstance(value, (int, float, bool, str)):
                    hyperparams[attr] = value
                elif isinstance(value, (tuple, list)) and all(isinstance(v, (int, float, bool)) for v in value):
                    hyperparams[attr] = list(value)
                else:
                    hyperparams[attr] = repr(value)
        return hyperparams

    def serialize_fx_value(self, value: Any) -> Any:
        if isinstance(value, torch.fx.Node):
            return {
                "type": "node_ref",
                "name": value.name,
                "op": value.op
            }
        if isinstance(value, (list, tuple)):
            return [self.serialize_fx_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self.serialize_fx_value(val) for key, val in value.items()}
        if isinstance(value, torch.dtype):
            return str(value)
        if isinstance(value, torch.Size):
            return list(value)
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        return repr(value)

    def collect_runtime_trace(self, model: torch.nn.Module, dummy_inputs: Dict[str, torch.Tensor]) -> Tuple[List[Dict[str, Any]], Any]:
        trace_entries: List[Dict[str, Any]] = []
        handles = []

        module_paths: Dict[torch.nn.Module, str] = {}
        for module_path, module in model.named_modules():
            module_paths[module] = module_path if module_path else "root"

        def hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any):
            entry = {
                "module_path": module_paths.get(module, module.__class__.__name__),
                "module_class": module.__class__.__name__,
                "module_repr": repr(module),
                "inputs": self.serialize_item(inputs),
                "outputs": self.serialize_item(output),
                "hyperparams": self.extract_module_hyperparams(module)
            }
            entry["call_index"] = len(trace_entries)
            trace_entries.append(entry)

        for _, module in model.named_modules():
            handles.append(module.register_forward_hook(hook))

        model.eval()
        with torch.inference_mode():
            runtime_output = self.invoke_model_with_dummy(model, dummy_inputs)

        for handle in handles:
            handle.remove()

        trace_entries.sort(key=lambda item: item["call_index"])
        return trace_entries, self.serialize_item(runtime_output)

    def invoke_model_with_dummy(self, model: torch.nn.Module, dummy_inputs: Dict[str, torch.Tensor]):
        try:
            return model(**dummy_inputs)
        except TypeError:
            # Some modules expect positional args; fall back to positional calling
            ordered_values = tuple(dummy_inputs[key] for key in dummy_inputs)
            return model(*ordered_values)

    def save_state_dict(self, model_name: str, model: torch.nn.Module) -> Tuple[str, Dict[str, Any]]:
        safe_name = model_name.replace('/', '_').replace(':', '_')
        target_dir = os.path.join(self.params_dir, safe_name)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, "state_dict.pt")
        state_dict = model.state_dict()
        torch.save(state_dict, target_path)

        metadata: Dict[str, Any] = {}
        parameter_names = {name for name, _ in model.named_parameters()}
        with torch.no_grad():
            for param_name, tensor in state_dict.items():
                tensor_cpu = tensor.detach().cpu()
                metadata[param_name] = {
                    "shape": list(tensor_cpu.shape),
                    "dtype": str(tensor_cpu.dtype),
                    "numel": tensor_cpu.numel(),
                    "requires_grad": param_name in parameter_names,
                }
                if tensor_cpu.numel() > 0:
                    stats_source = tensor_cpu.dequantize() if getattr(tensor_cpu, "is_quantized", False) else tensor_cpu.float()
                    metadata[param_name]["statistics"] = {
                        "mean": float(stats_source.mean().item()),
                        "std": float(stats_source.std(unbiased=False).item()) if tensor_cpu.numel() > 1 else 0.0,
                        "min": float(stats_source.min().item()),
                        "max": float(stats_source.max().item())
                    }
                    metadata[param_name]["sample_values"] = tensor_cpu.flatten()[:16].tolist()

        sha256_hash = hashlib.sha256()
        with open(target_path, "rb") as state_file:
            for chunk in iter(lambda: state_file.read(1024 * 1024), b""):
                sha256_hash.update(chunk)

        state_info = {
            "path": os.path.relpath(target_path, self.base_dir),
            "sha256": sha256_hash.hexdigest()
        }

        return target_path, {"state_dict": state_info, "parameters": metadata}

    def trace(self, model: torch.nn.Module, model_name: Optional[str] = None, dummy_inputs: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        if model_name is None:
            model_name = model.__class__.__name__
        safe_name = model_name.replace('/', '_').replace(':', '_')

        trace_data: Dict[str, Any] = {
            "format_version": "1.0",
            "model_name": model_name,
            "safe_model_name": safe_name,
            "status": "pending"
        }

        if not isinstance(model, torch.nn.Module):
            trace_data["status"] = "error"
            trace_data["error"] = "Provided object is not a torch.nn.Module."
            return trace_data

        try:
            if dummy_inputs is None:
                dummy_inputs = self.get_dummy_input(model)

            model.eval()
            graph_json = None
            graph_error = None

            try:
                input_items = list(dummy_inputs.items())

                class Wrapper(torch.nn.Module):
                    def __init__(self, core_model: torch.nn.Module, key_order: List[str]):
                        super().__init__()
                        self.core_model = core_model
                        self.key_order = key_order

                    def forward(self, *args):
                        kwargs = {key: arg for key, arg in zip(self.key_order, args)}
                        return self.core_model(**kwargs)

                key_order = [key for key, _ in input_items]
                wrapper = Wrapper(model, key_order)
                example_inputs = tuple(value for _, value in input_items)

                traced = make_fx(wrapper, tracing_mode="symbolic", _allow_non_fake_inputs=True)(*example_inputs)
                ShapeProp(traced).propagate(*example_inputs)

                graph_json = []
                for node in traced.graph.nodes:
                    node_meta: Dict[str, Any] = {}
                    if 'tensor_meta' in node.meta:
                        tm = node.meta['tensor_meta']
                        node_meta['tensor'] = {
                            "shape": list(tm.shape) if hasattr(tm, 'shape') else None,
                            "dtype": str(tm.dtype) if hasattr(tm, 'dtype') else None,
                            "stride": list(tm.stride) if hasattr(tm, 'stride') and tm.stride is not None else None,
                            "requires_grad": bool(getattr(tm, 'requires_grad', False)),
                            "device": str(getattr(tm, 'device', 'unknown'))
                        }
                    if 'val' in node.meta:
                        node_meta['value'] = self.serialize_fx_value(node.meta['val'])

                    graph_json.append({
                        "name": node.name,
                        "op": node.op,
                        "target": str(node.target),
                        "args": self.serialize_fx_value(node.args),
                        "kwargs": {key: self.serialize_fx_value(value) for key, value in node.kwargs.items()},
                        "users": sorted(user.name for user in node.users),
                        "meta": node_meta
                    })

                trace_data["graph"] = graph_json
                trace_data["graph_ir"] = str(traced.graph)
                trace_data["status"] = "ok"
            except Exception as trace_exc:
                graph_error = str(trace_exc)
                trace_data["status"] = "partial"
                trace_data["graph_error"] = graph_error

            hierarchy = self.build_hierarchy(model)
            trace_data["hierarchy"] = hierarchy

            runtime_trace, runtime_output = self.collect_runtime_trace(model, dummy_inputs)
            trace_data["runtime_trace"] = runtime_trace
            trace_data["model_output"] = runtime_output
            trace_data["input_signature"] = {key: self.serialize_item(value) for key, value in dummy_inputs.items()}

            saved_path, params_info = self.save_state_dict(model_name, model)
            trace_data["params"] = params_info

            if hasattr(model, "config") and hasattr(model.config, "to_dict"):
                trace_data["model_config"] = model.config.to_dict()

            if graph_json is None:
                trace_data.setdefault("warnings", []).append("Graph trace unavailable; see graph_error for details.")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return trace_data
        except Exception as e:
            trace_data["status"] = "error"
            trace_data["error"] = str(e)
            return trace_data

    def save_trace(self, trace_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        model_name = trace_data["model_name"]
        safe_name = trace_data["safe_model_name"]
        if filename is None:
            filename = f"trace_{safe_name}.json"
        trace_path = os.path.join(self.base_dir, filename)
        with open(trace_path, "w") as trace_file:
            json.dump(trace_data, trace_file, indent=4)
        return trace_path

# Example usage
if __name__ == "__main__":
    tracer = ModelTracer()

    image_model_specs: List[Tuple[str, Type[torch.nn.Module]]] = [
        ("microsoft/resnet-50", AutoModel),
        ("facebook/convnext-base-224", AutoModel),
        ("google/vit-base-patch16-224", AutoModel),
    ]

    for model_name, model_class in image_model_specs:
        model = tracer.load_model(model_name, model_class)
        trace_data = tracer.trace(model, model_name)
        tracer.save_trace(trace_data)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()