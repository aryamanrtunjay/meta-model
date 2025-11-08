from huggingface_hub import list_models
from tqdm import tqdm

# We will search for models that have ANY of these vision tags
VISION_TAGS = [
    "image-classification",
    "object-detection",
    "image-segmentation",
    "image-to-image",
    "unconditional-image-generation",
    "text-to-image",
    "visual-question-answering",
    "zero-shot-image-classification",
]

# This will hold all unique model IDs
model_ids = set()

print("Fetching models for all vision tags...")
# Use tqdm to show a progress bar
for tag in tqdm(VISION_TAGS, desc="Fetching tags"):
    # We fetch models that have the vision tag AND are PyTorch models
    # 'full=True' is needed to get the tags, 'limit' can be increased
    models = list_models(filter=[tag, "pytorch"], full=True, limit=5000) 
    
    for model in models:
        # We only want models from the main 'transformers' library
        # This avoids diffusers, adapters, etc.
        if "transformers" in model.tags:
            model_ids.add(model.modelId)

# Save the list to a file
output_filename = "model_list.txt"
with open(output_filename, "w") as f:
    for model_id in sorted(list(model_ids)):
        f.write(model_id + "\n")

print(f"\nDone. Found {len(model_ids)} unique vision models.")
print(f"Model list saved to {output_filename}")