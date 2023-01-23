import sys
from src.data.Datasets import WMT18
from src.utils.checklist.translation import TranslationTemplates

from src.utils.fs import save_scores_to_file
from tqdm import tqdm

# =====================================
# Import Inputs
# =====================================

datasets = sys.argv[1].split(",") if len(sys.argv) == 2 else None

TT = TranslationTemplates()

# =====================================
# Prepare the Dataset Helper Functions
# =====================================


def scramble_refs_wmt18():
    for lang_pair in tqdm(WMT18.supported_languages, desc="language pairs"):
        # load the datasets
        dataset = WMT18(lang_pair, batch_size=1)
        dataloader = dataset.setup().test_dataloader()
        # jumble the sentences
        jumbles = []
        for data in tqdm(dataloader, desc="datasets"):
            source = data["source"][0]
            system = data["system"][0]
            reference = data["reference"][0]
            jumbles.append(
                {
                    "o_source": source,
                    "o_system": system,
                    "o_reference": reference,
                    "j_system": [TT.jumble(system) for _ in range(10)],
                    "j_reference": [TT.jumble(reference) for _ in range(10)],
                }
            )
        # save the metric scores
        rel_path = "data/processed/wmt18"
        file_path = f"permutations.{lang_pair}.json"
        save_scores_to_file(rel_path, file_path, jumbles)


# =====================================
# Calculate the Scores on the Datasets
# =====================================

if __name__ == "__main__":
    if datasets == None or "wmt18" in datasets:
        print("WMT18 dataset: Start evaluation")
        scramble_refs_wmt18()
