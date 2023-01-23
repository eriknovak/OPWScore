import os
import json


def read_json_file(dirpath, filename):
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "r", encoding="utf8") as f:
        return json.load(f)


def save_scores_to_file(dirpath, filename, scores):
    """Saves the scores into the file
    Args:
        dirpath (str): The relative path from the src/utils folder.
        filename (str): The name of the file.
        scores (dict[]): The array of dictionaries containing the model ID and score.
    Returns:
        None

    """
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, filename), "w", encoding="utf8") as f:
        json.dump(scores, f, ensure_ascii=False)
