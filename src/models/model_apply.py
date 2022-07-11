import os
import yaml

from src.models.model import SeqMoverScore

# =====================================
# Import Training Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))

model_name = params["model_params"]["model"]
tokenizer_name = params["model_params"]["tokenizer"]
reg = params["model_params"]["reg"]
nit = params["model_params"]["nit"]

# =====================================
# Define the Model
# =====================================

model = SeqMoverScore(model=model_name, tokenizer=tokenizer_name, reg=reg, nit=nit)

# =====================================
# Apply the Model on an Example
# =====================================

system_text = "He was making coffee"

references = [
    "He was making black coffee",
    "She drinks coffee",
    "Mark was going skiing",
    "Coffee is the source of life",
]


image_path = "./results/example_explain.png"
model.visualize(system_text, references, image_path=image_path)
