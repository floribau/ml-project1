# ML Project 1

This project contains the data-processing and binary-classification workflow in
[`run.ipynb`](./run.ipynb). The notebook works with the supplied BRFSS dataset
and includes regularized logistic regression and a linear SVM trained with
stochastic gradient descent.

## Requirements

- Python 3
- A Linux terminal with `bash` or a similar POSIX-compatible shell

The required Python packages are pinned in
[`requirements.txt`](./requirements.txt).

## Set up the virtual environment

Open a terminal in the project root (the folder containing
`requirements.txt`) and create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

With the environment activated, install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Select the interpreter from `.venv` if it is not chosen
automatically.

## Prepare the data

The notebook expects the extracted dataset files in `data/dataset`, because
the loading cell uses the path `data/dataset`. If the dataset is still
compressed, extract [`data/dataset_to_release.zip`](./data/dataset_to_release.zip)
so that the resulting files are located under:

```text
data/
└── dataset/
    ├── x_train.csv
    ├── x_test.csv
    └── y_train.csv
```

## Run the notebook

From the project root, with the virtual environment activated, start Jupyter:

```bash
python -m jupyter notebook
```

Open [`run.ipynb`](./run.ipynb) in the Jupyter browser window.

Run the notebook from top to bottom through the following stages:

1. **Setup and imports**: import the project modules and set the random seed.
2. **Loading the data**: load the training and test data from `data/dataset`.
3. **Preprocessing the data**: clean the survey codes, select matching
   training/test features, split off a validation set, impute missing values,
   standardize the features, add the bias term, and balance the training
   classes.
4. **Examine the data**: optionally inspect correlations, covariance,
   variances, and class distributions.
5. **Training the model**: optionally run the regularized logistic-regression
   and SVM training cells.
6. **Prediction**: load the trained model weights and generate predictions for
   the test set.

## Skipping the training cells

Training is not required to reproduce the prediction output. The logistic
regression training and the SVM comparison can take a significant amount of
time, and the repository already contains the selected logistic-regression
weights in [`best_model_weights.npy`](./best_model_weights.npy).

You must still run the setup/imports, data-loading, and preprocessing sections,
because the prediction section needs the preprocessed test matrix
(`tx_test`) and the test IDs. After those sections have completed, you can
skip the training cells and continue at **Prediction**. That section:

1. loads `best_model_weights.npy`;
2. applies the sigmoid classifier to the preprocessed test data; and
3. writes the predictions to `test.csv`.

If you want to reproduce the experiments, run the **Training the model**
section as well. It contains the regularized logistic-regression experiment
used for the final model and the linear SVM comparison.

## Project files

- [`run.ipynb`](./run.ipynb): end-to-end loading, preprocessing, training, and
  prediction workflow.
- [`requirements.txt`](./requirements.txt): pinned Python dependencies.
- [`helpers.py`](./helpers.py): data loading and submission-file helpers.
- [`preprocessing.py`](./preprocessing.py): feature cleaning and preprocessing
  functions.
- [`implementations.py`](./implementations.py): model implementations,
  including regularized logistic regression and SVM training.
- [`best_model_weights.npy`](./best_model_weights.npy): precomputed selected
  logistic-regression weights used by the prediction shortcut.
- [`data/`](./data/): dataset archive and extracted dataset location.
