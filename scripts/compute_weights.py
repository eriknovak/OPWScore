import os
import sys
import yaml

from tqdm import tqdm

from src.data.Datasets import WMT17
from transformers import AutoTokenizer
from src.utils.weight_store import WeightStore

# =====================================
# DEFAULT Values
# =====================================

# create the directory path to store the IDF weights
dir_path = os.path.join("results", "weight_stores")
os.makedirs(dir_path, exist_ok=True)

# =====================================
# Import Inputs
# =====================================

languages = sys.argv[1].split(",") if len(sys.argv) >= 2 else None

# =====================================
# Import Model Parameters
# =====================================

params = yaml.safe_load(open("params.yaml"))
tokenizer_name = params["model_params"]["tokenizer"]
# initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# =====================================
# Calculate the Weights for the Languages
# =====================================


for language in tqdm(languages, desc="languages"):
    # prepare the targetted language pairs
    language_pairs = list(
        filter(lambda x: x.split("-")[1] == language, WMT17.supported_languages)
    )

    # initialize the weight store
    ws = WeightStore()
    # iterate through the language pairs
    for lang_pair in tqdm(language_pairs, desc="language pairs"):
        dataset = WMT17(lang_pair, batch_size=32)
        dataloader = dataset.setup().test_dataloader()
        # use the references to calculate the tokens
        for data in tqdm(dataloader, desc="datasets"):
            ws.add_corpus(data["reference"], tokenizer=tokenizer)
    # save the weight store
    file_path = (
        f"weight_store.{language}.wmt17.{tokenizer_name.replace('/', '_')}.pickle"
    )
    ws.save(os.path.join(dir_path, file_path))
