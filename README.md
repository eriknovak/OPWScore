# Deneir Script

This is a template repository for creating an experiment environment in Python.
Its intention is to speed up the research process - reducing the repository
structure design - and to have it clean and concise throught multiple
experiments.

Inspired by the [cookiecutter][cookiecutter] folder structure.

**Instructions:**

- Search for all TODOs in the project and add the appropriate values
- Rename this READMEs title and description

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
conda create --name [TODO] python=3.8 pip

# activate the environment
conda activate [TODO]

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
TODO:Provide scripts for the experiments
```

### 🦉 Using DVC

An alternative way of running the whole experiment is by using [DVC][dvc]. To do this,
simply run:

```bash
dvc exp run
```

This command will read the `dvc.yaml` file and execute the stages accordingly, taking
any dependencies into consideration.

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

TODO: Related paper

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

[cookiecutter]: https://drivendata.github.io/cookiecutter-data-science/
[python]: https://www.python.org/
[conda]: https://www.anaconda.com/
[git]: https://git-scm.com/
[dvc]: https://dvc.org/
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
