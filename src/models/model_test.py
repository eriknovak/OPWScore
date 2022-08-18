import sys
import yaml

from src.models.model import Seq_LM_EMD

from src.data.preprocess import calculate_scores_sts, calculate_scores_wmt18


# =====================================
# Import Inputs
# =====================================

datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

# =====================================
# Import Model Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))

model_name = params["model_params"]["model"]
tokenizer_name = params["model_params"]["tokenizer"]
dist_type = params["model_params"]["dist_type"]
reg1 = float(params["model_params"]["reg1"])
reg2 = float(params["model_params"]["reg2"])
nit = int(params["model_params"]["nit"])

# =====================================
# Define the Model
# =====================================

# prepare the model
model = Seq_LM_EMD(
    model=model_name,
    tokenizer=tokenizer_name,
    dist_type=dist_type,
    reg1=reg1,
    reg2=reg2,
    nit=nit,
)

# =====================================
# Calculate the Scores on the Datasets
# =====================================


if datasets == None or "sts" in datasets:
    # TODO: implement the evaluation
    print("STS Benchmarks dataset: Start evaluation")
    calculate_scores_sts(model)

if datasets == None or "wmt18" in datasets:
    print("WMT18 dataset: Start evaluation")
    calculate_scores_wmt18(model)
