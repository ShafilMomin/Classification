# Verification notes

Review date: 17 September 2026.

Checks ran locally in Python 3.10. Notebook cells were executed in order with plots rendered using a non-interactive backend. Results are from this dataset and setup, not guaranteed real-world performance.

## decision_tree_classification.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.91,
  "confusion_matrix": [
    [
      62,
      6
    ],
    [
      3,
      29
    ]
  ]
}
```

## kernel_svm.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.93,
  "confusion_matrix": [
    [
      64,
      4
    ],
    [
      3,
      29
    ]
  ]
}
```

## k_nearest_neighbors.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.93,
  "confusion_matrix": [
    [
      64,
      4
    ],
    [
      3,
      29
    ]
  ]
}
```

## logistic_regression.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.89,
  "confusion_matrix": [
    [
      65,
      3
    ],
    [
      8,
      24
    ]
  ]
}
```

## naive_bayes.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.9,
  "confusion_matrix": [
    [
      65,
      3
    ],
    [
      7,
      25
    ]
  ]
}
```

## random_forest_classification.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.91,
  "confusion_matrix": [
    [
      63,
      5
    ],
    [
      4,
      28
    ]
  ]
}
```

## support_vector_machine.ipynb

```json
{
  "check": "all code cells executed",
  "accuracy": 0.9,
  "confusion_matrix": [
    [
      66,
      2
    ],
    [
      8,
      24
    ]
  ]
}
```

## Packages

```json
{
  "numpy": "2.2.6",
  "pandas": "2.3.3",
  "matplotlib": "3.10.9",
  "scikit-learn": "1.7.2",
  "scipy": "1.15.3",
  "nltk": "3.10.3",
  "xgboost": "3.2.0"
}
```

TensorFlow was not installed in the review environment. ANN checks cover preprocessing only; CNN checks cover syntax only. Their historical training outputs are described separately in their READMEs.
