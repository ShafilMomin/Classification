# Classification: Social Network Ads

Compare **seven classification models** that predict whether a user purchased a product from their age and estimated salary.

**Python · pandas · scikit-learn · Matplotlib**

## Dataset and workflow

[Social_Network_Ads.csv](Social_Network_Ads.csv) contains **400 rows** and three columns: `Age`, `EstimatedSalary`, and the target `Purchased` (0 or 1).

Every notebook uses the same split: **300 training rows / 100 test rows**, `random_state=0`. StandardScaler is fitted on training data only. Each model makes predictions, prints a confusion matrix and plots its decision boundaries.

## Measured comparison

All seven notebooks were executed during the September 2026 review, including their plot cells.

| Notebook / model | Test accuracy |
| :--- | :---: |
| [Logistic Regression](logistic_regression.ipynb) | 89% |
| [K-Nearest Neighbors](k_nearest_neighbors.ipynb) | 93% |
| [Linear SVM](support_vector_machine.ipynb) | 90% |
| [RBF SVM](kernel_svm.ipynb) | 93% |
| [Gaussian Naive Bayes](naive_bayes.ipynb) | 90% |
| [Decision Tree](decision_tree_classification.ipynb) | 91% |
| [Random Forest](random_forest_classification.ipynb) | 91% |

KNN and RBF SVM tie at 93% on this one split. This does not establish that they are best on every dataset. There is no cross-validation or separate model-selection set here.

KNN uses 5 neighbors. The random forest uses 10 trees. Both tree examples use the entropy criterion.

## Review improvement

Decision-boundary plots now use a bounded **200 × 200 grid**. The previous salary-axis step generated millions of points and could use excessive memory. Model training and the dataset were preserved.

## Run it

Open the notebook in Jupyter or Google Colab. Keep its CSV/TSV in the notebook's working folder; opening a notebook from GitHub in Colab does not automatically upload the data.

For a local setup, clone this repository, create and activate a virtual environment, then run:

```bash
python -m pip install numpy pandas matplotlib scikit-learn notebook
python -m notebook
```

Run cells from top to bottom.

[Verification notes](VALIDATION.md) · Tested versions: `requirements.txt`.
