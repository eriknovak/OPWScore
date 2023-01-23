import sys
import yaml
from src.models.model import Seq_LM_EMD
from src.data.preprocess import calculate_fluency

# =====================================
# Import Inputs
# =====================================

languages = sys.argv[1].split(",") if len(sys.argv) >= 2 else None
datasets = sys.argv[2].split(",") if len(sys.argv) >= 3 else None


# =====================================
# Import Model Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))

distance = params["model"]["distance"]
weight_dist = params["model"]["weight_dist"]
temporal_type = params["model"]["temporal_type"]
reg1 = params["model"]["reg1"]
reg2 = params["model"]["reg2"]
nit = params["model"]["nit"]

# =====================================
# Define the Model
# =====================================

for language in languages:
    # prepare the model
    model = Seq_LM_EMD(
        distance=distance,
        weight_dist=weight_dist,
        temporal_type=temporal_type,
        lang=language,
        reg1=reg1,
        reg2=reg2,
        nit=nit,
    )

    # =====================================
    # Calculate the Scores on the Datasets
    # =====================================

    if datasets == None or "wmt18" in datasets:
        print("WMT18 dataset: Start evaluation")
        calculate_fluency(model, "wmt18")
