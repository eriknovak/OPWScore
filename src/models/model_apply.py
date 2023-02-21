import os
import yaml

from src.models.model import OPWScore

# =====================================
# Import Training Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))

distance = params["model_params"]["distance"]
weight_dist = params["model_params"]["weight_dist"]
temporal_type = params["model_params"]["temporal_type"]
reg1 = params["model_params"]["reg1"]
reg2 = params["model_params"]["reg2"]
nit = params["model_params"]["nit"]

# =====================================
# Define the Model
# =====================================

model = OPWScore(
    distance=distance,
    weight_dist=weight_dist,
    temporal_type=temporal_type,
    lang="en",
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


image_path = "results/example_explain.png"
model.visualize(system_text, references, image_path=image_path)
