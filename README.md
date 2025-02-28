# OPWScore

This project contains the code for running the OPWScore experiments.

## 🔃 Citation

If you find this code useful, feel free to reference the following paper:

```bib
@article{Novak2024-et,
  title={Evaluating Text Generation Model Performance by Combining Semantic Meaning and Word Order}, 
  author={Novak, Erik and Bizjak, Luka and Mladenić, Dunja and Grobelnik, Marko},
  journal={IEEE Access},
  year={2024},
  volume={12},
  number={},
  pages={95265-95277},
  doi={10.1109/ACCESS.2024.3426082}
}
```

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [conda][conda]. For setting up your research environment and python dependencies.
- [dvc][dvc]. For versioning your data.
- [git][git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First create the virtual environment where all the modules will be stored.

#### Using venv

Using the `venv` command, run the following commands:

```bash
# create a new virtual environment
python -m venv .venv

# activate the environment (UNIX)
source ./.venv/bin/activate

# activate the environment (WINDOWS)
./.venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

#### Using conda

Install [conda][conda], a program for creating python virtual environments. Then run the following commands:

```bash
# create a new virtual environment
conda create --name opwscore python=3.8 pip

# activate the environment
conda activate opwscore

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

The data used in the experiments are examples from the WMT17, WMT18 and WMT20
metric evaluation data sets.

The data sets are taken from the COMET metric page. Download the files and store them as stated in the table.

| Data set | Folder Save Path | Link                                                                                                 |
| -------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| WMT17    | `data/raw/wmt17` | [Download](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2017-da.csv.tar.gz) |
| WMT18    | `data/raw/wmt18` | [Download](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2018-da.csv.tar.gz) |
| WMT20    | `data/raw/wmt20` | [Download](https://unbabel-experimental-data-sets.s3.eu-west-1.amazonaws.com/wmt/2020-da.csv.tar.gz) |

## ⚗️ Experiments

To run the experiments, run the folowing commands:

```bash
# calculate the IDF weights
python scripts/models/compute_weights.py en,cs,de,fi,ru,tr,zh

# run the adequacy experiments on the selected languages and data sets
python scripts/models/performance_test.py en,cs,de,fi,ru,tr,zh wmt18,wmt20

# calculate the model's adequacy performance scores on the provided data sets
python scripts/models/performance_eval.py wmt18,wmt20

# run the fluency experiments on the selected data sets
python scripts/models/fluency_test.py wmt18

# calculate the model's fluency performance scores on the provided data sets
python scripts/models/fluency_eval.py wmt18
```

### 🦉 Using DVC

An alternative way of running the whole experiment is by using [DVC][dvc]. To do this,
simply run:

```bash
dvc exp run
```

This command will read the `dvc.yaml` file and execute the stages accordingly, taking
any dependencies into consideration. **NOTE: This will only run the experiments on the WMT18 data sets.**

Afterwards, we can compare the performance of the models by running:

```bash
dvc exp show
```

To save the best performance parameters run:

```bash
# [exp-id] is the ID of the experiment that yielded the best performance
dvc exp apply [exp-id]
```

### Results

The `results` folder contain the experimental results.

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work was supported by the Slovenian Research Agency, and the European Union's Horizon 2020 project Humane AI Net [H2020-ICT-952026] and the Horizon Europe project enRichMyData [HE-101070284].

[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
