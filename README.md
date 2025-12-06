# ITU BDS MLOPS'25 - Project

## 📝 About the Project
This project is a submission for the **ITU BDS MLOPS'25** final exam. It demonstrates the transformation of a raw, experimental Jupyter notebook into a production-ready MLOps pipeline.

**The Machine Learning Objective** The goal of the underlying model is to identify website users who are potential new customers. By analyzing user behavior data, the model performs a **binary classification** to predict whether a specific user will convert (turn into a customer).

**The Engineering Objective** The original codebase consisted of a monolithic notebook containing data processing, training, and "fluff" (unused code/comments). This project refactors that source into a clean, modularized structure following PEP 8 standards. It implements a full automation pipeline using **Dagger**, **DVC** for data versioning, and **MLFlow** for experiment tracking and model selection.

## 🗄️ Project Organization

```
.
├── .github
│   └── workflows
│       └── test_action.yml
├── .gitignore
├── .dagger
│   ├── dagger.gen.go
│   ├── go.mod
│   ├── go.sum
│   ├── internal
│   └── main.go
├── LICENSE
├── Makefile
├── README.md
├── dagger.json
├── pyproject.toml
├── requirements.txt
├── data
│   ├── interim
│   │   └── make_dataset.py
│   ├── processed
│   └── raw
├── models
├── notebooks
│   └── main.ipynb
├── tests
│   ├── data
│   │   ├── training_data.csv
│   │   ├── X.csv
│   │   └── y.csv
│   └── test_training_data.py
└── itu_sdse_project
    ├── config.py
    ├── features.py
    ├── helpers.py
    └── modeling
        ├── predict.py
        ├── selection.py
        └── train.py
```

## 🚀 Installation
> Make sure you have [Go](https://go.dev/doc/install), [Python](https://www.python.org/downloads/), and [dvc](https://doc.dvc.org/install) already installed on your machine.

To start out using the project, clone the repository and follow these steps:
```bash
# 1. Create virtual environment
make create_environment

# 2. Activate virtual environment
workon itu-sdse-project

# 3. Install dependencies
make requirements

# 4. Pull raw data (Optional for dagger workflow)
dvc get https://github.com/Jeppe-T-K/itu-sdse-project-data \
raw_data.csv -o data/raw
```
> [!warning]
> - If the `make` command doesn't work, install [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/)
> - If you have any problem with running the `workon` command, please refer to this [post](https://stackoverflow.com/questions/21928555/virtualenv-workon-command-not-found).

## 📚 Documentation
In this project, we split the Jupyter notebook located in `notebooks/main.ipynb` into manageable, clean, and easy-to-read code throughout the codebase.

First we tackled the data cleaning part of the notebook, where we placed data generation into `make_dataset.py` files inside the `data/` directory.

Training code in the Jupyter notebook file was structured inconsistently, so we decided to refactor it significantly. Starting off, the `xgboost` model was included in our MLFlow setup. The Logistic Regression model was already utilizing MLFlow, so we followed the code in the Logistic Regression notebook cell and recreated it in the `xgboost` portion of the code.

After the training code, we implemented the functionality of selecting the best-performing model after training runs have finished. For this process, we decided to refactor the code in the notebook to use strictly MLFlow for the performance data of our trained models.

Finally, we leveraged our organized codebase to write Dagger functions. The biggest benefits were code consistency and reliability, which enabled us to build the Dagger pipeline quickly and correctly.

To conclude, we consolidated our work into a GitHub Action. Its sole purpose was to train the best-performing model and test it using the inference action specified in the project description.

## ⚙️ Commands & Options
### `train.py`
Trains a classification model using the dataset found in `data/processed/`.

```bash
python itu_sdse_project/modeling/train.py <log-reg|xgboost> 
```

| Argument | Required | Description                         |
| -------- | -------- | ----------------------------------- |
| log-reg  | true     | Trains a Logistic Regression model. |
| xgboost  | true     | Trains an XGBoost Classifier.       |

### `selection.py`
Selects the best performing model from training runs and registers it as staging in MLFlow.

```bash
python itu_sdse_project/modeling/selection.py
```

### `features.py`
Creates datasets for model training.

```bash
python itu_sdse_project/features.py
```

## 🤖 Dagger Automation
### `BuildEnv`
Builds the environment using `python:3.12.2-bookworm` Docker image, installs python dependencies, dvc, and pulls raw data.

### `PrepareData`
Cleans the raw data and splits the cleaned data into processed `features.csv` and `labels.csv` files.

### `Train`
Trains the models `log-reg` and `xgboost`.

### `Select`
Selects the model with the highest f1-score from all training runs, and registers it into staging phase.

### `Download`
Downloads the best performing model as a `model.pkl` artifact.
> [!info]
> To run any of the commands type `dagger call <command>` from the project root directory

## ✅ Project requirements checklist
- [x] Remove unused code and misleading comments
- [x] Usage of Git
- [x] Usage of PRs
- [x] Decomposition of the notebook
- [x] Adherence to the MLOps project structure
- [x] Presence of tests
- [x] Management of data
- [x] Presence of a workflow that trains the model
- [x] Presence of a workflow that tests the model
- [x] Orchestration of dagger workflow through gh actions

## Contributions
**Jan Nahalka**
- Jupyter Notebook Decomposition
- Dagger workflow
- Github actions workflow

**Oliver Souc**
- Cookiecutter template setup
- README Documentation
- Testing, logging

## Exam Notes
- We have not implemented deployment to the production since it was not required in the exam document.