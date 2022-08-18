import os
import yaml

from src.models.model import Seq_LM_EMD

# =====================================
# Import Training Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))

model_name = params["model_params"]["model"]
tokenizer_name = params["model_params"]["tokenizer"]
dist_type = params["model_params"]["dist_type"]
reg1 = params["model_params"]["reg1"]
reg2 = params["model_params"]["reg2"]
nit = params["model_params"]["nit"]

# =====================================
# Define the Model
# =====================================

model = Seq_LM_EMD(
    model=model_name,
    tokenizer=tokenizer_name,
    dist_type=dist_type,
    reg1=reg1,
    reg2=reg2,
    nit=nit,
)

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
