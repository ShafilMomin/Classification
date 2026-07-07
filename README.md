# 🎯 Social Network Ads — Classification Algorithms Comparison

A comprehensive machine learning project that implements and compares **7 classification algorithms** on the same dataset to predict whether a user will purchase a product based on Age and Estimated Salary.

---

## 📌 Problem Statement

A company ran Social Network Ads targeting users. The goal is to predict — based on a user's **Age** and **Estimated Salary** — whether they will **purchase the product or not**.

This project benchmarks 7 different classification algorithms on the same dataset, comparing their accuracy and decision boundaries.

---

## 📂 Repository Structure

```
├── logistic_regression.ipynb              # Logistic Regression
├── k_nearest_neighbors.ipynb              # K-Nearest Neighbors
├── support_vector_machine.ipynb           # SVM (Linear Kernel)
├── kernel_svm.ipynb                       # SVM (RBF Kernel)
├── naive_bayes.ipynb                      # Naive Bayes
├── decision_tree_classification.ipynb     # Decision Tree
├── random_forest_classification.ipynb     # Random Forest
├── Social_Network_Ads.csv                 # Shared dataset
└── README.md
```

---

## 📊 Dataset

**File:** `Social_Network_Ads.csv`
**Rows:** 400 users
**Train / Test Split:** 300 train / 100 test (75% / 25%)

| Column | Description |
|---|---|
| **Age** | User's age ← Feature used |
| **EstimatedSalary** | Estimated annual salary ← Feature used |
| **Purchased** | Target → 1 = Bought, 0 = Didn't buy |

---

## ⚙️ Common Pipeline (All 7 Notebooks)

Every notebook follows the same preprocessing pipeline:

```python
# 1. Load dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values   # Age + EstimatedSalary
y = dataset.iloc[:, -1].values    # Purchased (0 or 1)

# 2. Train-Test Split (75/25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 3. Feature Scaling (important — Age and Salary are on very different scales!)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

# 4. Train classifier → Predict → Evaluate
# (Changes per algorithm — see below)

# 5. Single user prediction example
classifier.predict(sc.transform([[30, 87000]]))  # Age=30, Salary=87k → [0] (Won't buy)

# 6. Confusion Matrix + Accuracy
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)

# 7. Decision boundary visualization (train set + test set)
```

---

## 🤖 Algorithm Details

### 1️⃣ Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
```
- Linear decision boundary
- Outputs probability via Sigmoid function
- Fast and interpretable baseline

---

### 2️⃣ K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
```
- Classifies based on 5 nearest neighbors
- `p=2` = Euclidean distance
- Non-parametric — no assumptions about data distribution

---

### 3️⃣ Support Vector Machine (Linear Kernel)
```python
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
```
- Finds optimal hyperplane separating the two classes
- Maximizes margin between classes
- Best for linearly separable data

---

### 4️⃣ Kernel SVM (RBF Kernel)
```python
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
```
- Uses Radial Basis Function (RBF) kernel
- Maps data into higher dimensions — handles non-linear boundaries
- More flexible than linear SVM

---

### 5️⃣ Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
```
- Probabilistic classifier using Bayes theorem
- Assumes features are independent (naive assumption)
- Fast, works well with small datasets

---

### 6️⃣ Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
```
- Splits data using Information Gain (entropy criterion)
- Creates interpretable if-else rules
- Can overfit — but Random Forest fixes this

---

### 7️⃣ Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
```
- Ensemble of 10 Decision Trees
- Each tree votes → majority wins
- Reduces overfitting of single Decision Tree

---

## 📈 Results — All 7 Algorithms Compared

| Algorithm | Confusion Matrix | Accuracy |
|---|---|---|
| Logistic Regression | [[65, 3], [8, 24]] | **89%** |
| KNN | [[64, 4], [3, 29]] | **93%** |
| SVM (Linear) | [[66, 2], [8, 24]] | **90%** |
| Kernel SVM (RBF) | [[64, 4], [3, 29]] | **93%** |
| Naive Bayes | [[65, 3], [7, 25]] | **90%** |
| Decision Tree | [[62, 6], [3, 29]] | **91%** |
| Random Forest | [[63, 5], [4, 28]] | **91%** |

> 🏆 **Best Performers:** KNN and Kernel SVM — both at **93% accuracy**

---

## 🥇 Algorithm Ranking

```
🥇 93% → KNN (k=5, Euclidean)
🥇 93% → Kernel SVM (RBF)
🥉 91% → Decision Tree (entropy)
🥉 91% → Random Forest (10 trees)
   90% → SVM (Linear)
   90% → Naive Bayes
   89% → Logistic Regression
```

---

## ⚔️ Algorithm Comparison

| Algorithm | Boundary Type | Speed | Interpretable | Scales Well |
|---|---|---|---|---|
| **Logistic Regression** | Linear | ⚡ Fast | ✅ Yes | ✅ Yes |
| **KNN** | Non-linear | 🐢 Slow (predict) | ❌ No | ❌ No |
| **SVM Linear** | Linear | ⚡ Fast | ❌ No | ✅ Yes |
| **Kernel SVM** | Non-linear | Medium | ❌ No | ❌ No |
| **Naive Bayes** | Non-linear | ⚡ Fast | ✅ Partial | ✅ Yes |
| **Decision Tree** | Non-linear | ⚡ Fast | ✅ Yes | ✅ Yes |
| **Random Forest** | Non-linear | Medium | ❌ No | ✅ Yes |

---

## 💡 Key Insights

```
✅ Non-linear algorithms won (KNN, Kernel SVM at 93%)
   → The purchase decision boundary is NOT a straight line
   → Age + Salary together create curved patterns

✅ Kernel SVM vs Linear SVM: 93% vs 90%
   → RBF kernel handles the non-linearity better

✅ Random Forest vs Decision Tree: same 91%
   → 10 trees didn't add much gain on this small dataset

✅ Logistic Regression lowest at 89%
   → Confirms data is not linearly separable
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Loading dataset |
| `numpy` | Array operations |
| `matplotlib` | Decision boundary visualization |
| `scikit-learn` | All 7 classifiers + preprocessing + evaluation |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib scikit-learn
```

**3. Run any notebook**
Open any `.ipynb` file in Jupyter Notebook or Google Colab and run all cells.

> 💡 All notebooks use the same `Social_Network_Ads.csv` — upload once in Colab!

---

## 🧠 Key Concepts Used

- **Binary Classification** — Predict 1 of 2 outcomes (Buy / Not Buy)
- **Feature Scaling** — StandardScaler to normalize Age and Salary
- **Train-Test Split** — 75/25 split for unbiased evaluation
- **Confusion Matrix** — TP, TN, FP, FN breakdown
- **Decision Boundary** — Visual separation of classes in 2D
- **Ensemble Learning** — Random Forest combining multiple trees
- **Kernel Trick** — SVM mapping to higher dimensions

---

## 👤 Author

**Shafil** — AI & ML Engineering Student
B.Tech Final Year | Specialization: Computer Vision & Deep Learning

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
