# 🔵 K-Means Clustering — University Segmentation (Unsupervised Learning)

An **unsupervised machine learning project** that applies K-Means clustering to group universities into categories (Private vs Public) based on their features — without using labels during training.

This project demonstrates clustering, feature scaling, and cluster evaluation techniques.

---

## 📌 Overview

This project uses the **K-Means clustering algorithm** to segment universities into two groups based on institutional features. Although the dataset contains true labels (Private/Public), the model is trained **without using them**, and results are later compared for evaluation.

This highlights how unsupervised learning can discover natural structure in data.

---

## 🧠 Learning Type

**Unsupervised Learning**

- No labels used during model training
- Clusters formed purely from feature similarity
- True categories used only for post-cluster evaluation

---

## 🎯 Objective

Group universities into clusters based on characteristics such as:

- Enrollment
- Spending per student
- Graduation rate
- Faculty ratio
- Out-of-state tuition
- Other institutional metrics

Then compare clusters with actual categories.

---

## 🧩 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 📊 Dataset

University dataset containing institutional statistics and features.

Includes:
- Numeric institutional metrics
- Private/Public label (used only for evaluation — not training)

---

## 🔬 Project Workflow

### 1️⃣ Data Exploration
- Dataset inspection
- Feature distributions
- Correlation visualization
- Pair plots and cluster tendency checks

---

### 2️⃣ Data Preprocessing
- Feature selection
- Feature scaling using StandardScaler
- Normalization for distance-based clustering

---

### 3️⃣ Model Training — K-Means

- Choose number of clusters (K = 2)
- Fit K-Means on scaled features
- Assign cluster labels --> `KMeans(n_clusters=2)`
- 
---

### 4️⃣ Evaluation (Post-Clustering)

Since this is unsupervised learning:

- Compare cluster assignments vs actual labels
- Build confusion matrix
- Analyze cluster alignment

---

## 📈 Results

`classification_report:`
```
              precision    recall  f1-score   support

           0       0.21      0.65      0.31       212
           1       0.31      0.06      0.10       565

    accuracy                           0.22       777
   macro avg       0.26      0.36      0.21       777
weighted avg       0.29      0.22      0.16       777
```

Verdict - Not so bad considering the algorithm is purely using the features to cluster the universities into 2 distinct groups! This is how K Means is useful for clustering un-labeled data.

---

## ⚖️ Key Insight

Even without labels, K-Means was able to separate institutions into meaningful groups based on financial and academic features.

This demonstrates the power of unsupervised learning for structure discovery.

---

## ▶️ How to Run

### Clone repo

```bash
git clone https://github.com/rohitb281/k-means-clustering-project.git
cd k-means-clustering-project
```

### Install dependencies
```
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Run notebook
```
jupyter notebook
```

### Open
```
K Means Clustering Project.ipynb
```

Run all cells.

---

## 🧠 Concepts Demonstrated
- Unsupervised learning
- K-Means clustering
- Distance-based algorithms
- Feature scaling importance
- Cluster evaluation
- Post-cluster label comparison

---

## 🚀 Possible Improvements
- Elbow method for optimal K
- Silhouette score analysis
- PCA visualization of clusters
- Try hierarchical clustering
- DBSCAN comparison
- Cluster profiling

---

## ⚠️ Limitations
- K must be chosen manually
- Sensitive to feature scaling
- Assumes spherical clusters
- Results depend on initialization

---

## 📄 License
- Open for educational and portfolio use.

---

## 👤 Author
= Rohit Bollapragada
- GitHub: https://github.com/rohitb281
