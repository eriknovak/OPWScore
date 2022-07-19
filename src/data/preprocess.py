import os

DIRNAME = os.path.dirname(__file__)


def construct_source_path(lang_pair):
    source_lang = lang_pair.split("-")[0]
    lang_pair_id = lang_pair.replace("-", "")
    rel_path = f"../../data/raw/sources/newstest2018-{lang_pair_id}-src.{source_lang}"
    source_path = os.path.join(DIRNAME, rel_path)
    return source_path


def construct_reference_path(lang_pair):
    source_lang = lang_pair.split("-")[1]
    lang_pair_id = lang_pair.replace("-", "")
    rel_path = (
        f"../../data/raw/references/newstest2018-{lang_pair_id}-ref.{source_lang}"
    )
    reference_path = os.path.join(DIRNAME, rel_path)
    return reference_path


def get_system_paths(lang_pair, only_hybrid=True):
    [source_lang, target_lang] = lang_pair.split("-")
    rel_path = f"../../data/raw/system-outputs/newstest2018/{source_lang}-{target_lang}"
    system_folder_path = os.path.join(DIRNAME, rel_path)

    files = []
    for _, _, files in os.walk(system_folder_path):
        files = [
            {
                "path": os.path.join(DIRNAME, rel_path, file),
                "model": f"{file.split('.')[1]}.{file.split('.')[2]}",
            }
            for file in files
            if not only_hybrid or "hybrid" in file
        ]
    # return all files in the system path
    return files


def get_source_sentences(lang_pair):
    with open(construct_source_path(lang_pair), "r", encoding="utf8") as f:
        return [line.strip() for line in f.readlines()]


def get_reference_sentences(lang_pair):
    with open(construct_reference_path(lang_pair), "r", encoding="utf8") as f:
        return [line.strip() for line in f.readlines()]


def get_system_sentences(lang_pair, hybrid=True):
    system_file_names = get_system_paths(lang_pair, hybrid)

    system_values = {}
    for file in system_file_names:
        with open(file["path"], "r", encoding="utf8") as f:
            system_values[file["model"]] = [line.strip() for line in f.readlines()]

    return system_values


def get_humaneval_values(lang_pair, only_hybrid=True):
    human_eval = {}

    rel_path = "../../data/raw/DA/sys/DA-syslevelhybrids.csv"
    file_path = os.path.join(DIRNAME, rel_path)
    with open(file_path, "r", encoding="utf8") as f:
        for line in f.readlines():
            if lang_pair in line:
                [_, _, HYBRID, HUMAN] = line.strip().split(" ")
                human_eval[f"hybrid.{HYBRID}"] = float(HUMAN)

    if not only_hybrid:
        rel_path = "../../data/raw/DA/sys/DA-syslevel.csv"
        file_path = os.path.join(DIRNAME, rel_path)
        with open(file_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                if lang_pair in line:
                    [_, HUMAN, SYSTEM] = line.split(" ")
                    human_eval[SYSTEM] = float(HUMAN)

    return human_eval
