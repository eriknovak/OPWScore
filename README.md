# Seq-LM-EMD

This project contains the code for running the Seq-LM-EMD experiments.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [conda][conda]. For setting up your research environment and python dependencies.
- [dvc][dvc]. For versioning your data.
- [git][git]. For versioning your code.

## 🛠️ Setup

### Create a python environment

First create the virtual environment where all the modules will be stored.

#### Using virtualenv

Using the `virtualenv` command, run the following commands:

```bash
# install the virtual env command
pip install virtualenv

# create a new virtual environment
virtualenv -p python ./.venv

# activate the environment (UNIX)
./.venv/bin/activate

# activate the environment (WINDOWS)
./.venv/Scripts/activate

# deactivate the environment (UNIX & WINDOWS)
deactivate
```

#### Using conda

Install [conda][conda], a program for creating python virtual environments. Then run the following commands:

```bash
# create a new virtual environment
conda create --name seq-lm-emd python=3.8 pip

# activate the environment
conda activate seq-lm-emd

# deactivate the environment
deactivate
```

### Install

To install the requirements run:

```bash
pip install -e .
```

## 🗃️ Data

TODO: Provide information about the data used in the experiments

- Where is the data found
- How is the data structured

## ⚗️ Experiments

To run the experiments, run the folowing commands:

```bash
# generate the test model outputs
python src/models/model_test.py sts,wmt18

# evaluate the model outputs
python src/models/model_eval.py sts,wmt18
```

### 🦉 Using DVC

An alternative way of running the whole experiment is by using [DVC][dvc]. To do this,
simply run:

```bash
dvc exp run
```

This command will read the `dvc.yaml` file and execute the stages accordingly, taking
any dependencies into consideration.

To run multiple experiments we execute the following command:

```bash
# prepare the queue of experiments
dvc exp run --queue -S model_params.dist_type=seq  -S model_params.reg1=0.2  -S model_params.reg2=0.2  -S model_params.nit=100
dvc exp run --queue -S model_params.dist_type=emd  -S model_params.reg1=0.2  -S model_params.reg2=None -S model_params.nit=100
dvc exp run --queue -S model_params.dist_type=cls  -S model_params.reg1=None -S model_params.reg2=None -S model_params.nit=None
dvc exp run --queue -S model_params.dist_type=max  -S model_params.reg1=None -S model_params.reg2=None -S model_params.nit=None
dvc exp run --queue -S model_params.dist_type=mean -S model_params.reg1=None -S model_params.reg2=None -S model_params.nit=None

# execute all queued experiments (run 3 jobs in parallel)
dvc exp run --run-all --jobs 3
```

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

The results folder contain the experiment

TODO: Provide a list/table of experiment results

## 📦️ Available models

This project producted the following models:

- TODO: Name and the link to the model

## 🚀 Using the trained model

When the model is trained, the following script shows how one can use the model:

```python
TODO: Provide example on how to use the model
```

## 📚 Papers

In case you use any of the components for your research, please refer to
(and cite) the papers:

TODO: Paper

### 📓 Related work

T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore:
Evaluating Text Generation with BERT,” Apr. 2020. Accessed: Jun. 10, 2022.
[Online]. Available: http://www.openreview.net/pdf?id=SkeHuCVFDr

W. Zhao, M. Peyrard, F. Liu, Y. Gao, C. M. Meyer, and S. Eger, “MoverScore: Text
generation evaluating with contextualized embeddings and earth mover distance,”
presented at the Proceedings of the 2019 Conference on Empirical Methods in Natural
Language Processing and the 9th International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), Hong Kong, China, 2019. doi: 10.18653/v1/d19-1053.

B. Su and G. Hua, “Order-preserving Wasserstein distance for sequence matching,”
presented at the 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
Honolulu, HI, Jul. 2017. doi: 10.1109/cvpr.2017.310.

## 🚧 Work In Progress

- [ ] Setup script
- [ ] Code for data prep
- [ ] Code for model training
- [ ] Code for model validation
- [ ] Code for model evaluation
- [ ] Modify `params.yaml` and modify the scripts to read the params from the file
- [ ] Modify DVC pipelines for model training and evaluation

## 📣 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

This work is supported by the Slovenian Research Agency and the TODO.

[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
