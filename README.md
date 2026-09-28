# Credit Card Customer Segmentation — Unsupervised Learning

## 📌 Project Overview

This project focuses on **Credit Card Customer Segmentation** using unsupervised machine learning techniques.

The objective is to identify groups of customers with similar financial and behavioural patterns. These customer segments can help a credit card business understand how different customers use their cards and support more targeted business strategies.

The project uses the **Credit Card Dataset for Clustering** from Kaggle and applies:

- Exploratory Data Analysis (EDA)
- Behavioural Feature Engineering
- Missing Value Treatment
- Outlier Handling
- Log Transformation
- Feature Scaling
- K-Means Clustering
- Agglomerative Hierarchical Clustering
- Cluster Profiling
- Clustering Evaluation

---

## 🎯 Business Problem

Credit card customers have different spending and payment behaviours.

Some customers primarily use their cards for purchases, while others rely more heavily on cash advances. Customers can also differ in their balances, credit limits, payment behaviour, and overall card activity.

The purpose of this project is to group customers with similar behavioural characteristics into meaningful segments.

These segments can help a business:

- Understand customer behaviour
- Identify different usage patterns
- Develop customer-specific strategies
- Improve customer engagement
- Support data-driven marketing decisions

---

## 📊 Dataset

The project uses the **Credit Card Dataset for Clustering**.

### Dataset Source

**Kaggle:**  
https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

The dataset contains approximately **8,950 credit card customers** and **18 features**.

The `CUST_ID` column is removed because it is an identifier and does not represent customer behaviour.

---

## 🧾 Dataset Features

| Feature | Description |
|---|---|
| `BALANCE` | Account balance |
| `BALANCE_FREQUENCY` | Frequency of balance updates |
| `PURCHASES` | Total purchases |
| `ONEOFF_PURCHASES` | One-time purchase amount |
| `INSTALLMENTS_PURCHASES` | Installment purchase amount |
| `CASH_ADVANCE` | Cash advance amount |
| `PURCHASES_FREQUENCY` | Frequency of purchases |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-time purchases |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash advances |
| `CASH_ADVANCE_TRX` | Number of cash advance transactions |
| `PURCHASES_TRX` | Number of purchase transactions |
| `CREDIT_LIMIT` | Credit limit |
| `PAYMENTS` | Total payments |
| `MINIMUM_PAYMENTS` | Minimum payment amount |
| `PRC_FULL_PAYMENT` | Percentage of full payments |
| `TENURE` | Customer tenure in months |

---

# 🔄 Project Workflow

The project follows the following workflow:

**Data Loading**  
↓  
**Data Cleaning**  
↓  
**Missing Value Treatment**  
↓  
**Duplicate Check**  
↓  
**Exploratory Data Analysis**  
↓  
**Behavioural Feature Engineering**  
↓  
**Outlier Handling**  
↓  
**Log Transformation**  
↓  
**Feature Scaling**  
↓  
**K-Means Clustering**  
↓  
**Agglomerative Hierarchical Clustering**  
↓  
**Cluster Visualisation**  
↓  
**Cluster Profiling**  
↓  
**Model Evaluation**

---

# 🧹 1. Data Preprocessing

## Data Cleaning

The dataset is loaded and the customer identifier is removed.

The `CUST_ID` column is not used for clustering because it is only an identifier and does not contain behavioural information.

### Missing Values

Missing values are checked across all columns.

The missing values in:

- `CREDIT_LIMIT`
- `MINIMUM_PAYMENTS`

are handled using **median imputation**.

Median imputation is suitable for these financial variables because it is less influenced by extreme values than mean imputation.

### Duplicate Records

Duplicate rows are checked before continuing with the analysis.

---

# 📈 2. Exploratory Data Analysis

Exploratory analysis is performed to understand the distribution and relationships between customer features.

### Distribution Analysis

Histograms are created for:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`

These plots help identify skewness and extreme values.

### TENURE Analysis

A boxplot of `TENURE` is used to understand its distribution and identify possible unusual values.

### Correlation Analysis

A correlation heatmap is created for the numerical features.

This helps identify relationships between different customer behaviours and financial variables.

---

# 👥 3. Customer Behaviour Analysis

Additional behavioural analysis is performed before clustering.

### Purchase Utilisation

Purchase utilisation is calculated as:

**PURCHASES / CREDIT_LIMIT**

This provides an indication of the proportion of the available credit limit associated with purchases.

### Top Cash-Advance Customers

The top 10 customers with the highest `CASH_ADVANCE` values are identified and compared with their `PURCHASES` values.

This helps understand whether customers with high cash advances also have high purchase activity.

### Zero-Purchase Customers

The percentage of customers with:

**PURCHASES = 0**

is calculated to understand how many customers have no recorded purchase activity.

---

# ⚙️ 4. Behavioural Feature Engineering

Four additional behavioural features are created.

### Monthly Average Purchase

**Monthly_Avg_Purchase = PURCHASES / TENURE**

This estimates the customer's average monthly purchase activity.

### Monthly Average Cash Advance

**Monthly_Avg_Cash_Advance = CASH_ADVANCE / TENURE**

This estimates the customer's average monthly cash-advance activity.

### Limit Usage

**Limit_Usage = BALANCE / CREDIT_LIMIT**

This represents the customer's balance relative to their available credit limit.

### Payment-to-Minimum-Payment Ratio

**Payment_to_Minpayment_Ratio = PAYMENTS / (MINIMUM_PAYMENTS + 1)**

This compares the customer's total payments with their minimum payment requirement.

---

# 📦 5. Outlier Handling

Financial variables can contain extreme values that may strongly influence distance-based clustering algorithms.

Boxplots are created for:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`

Outliers are handled using the **IQR method**.

The upper threshold is defined as:

**Q3 + 3 × IQR**

Values above this threshold are capped rather than removed.

This approach is known as **winsorization** and allows the original customer records to remain in the dataset.

---

# 📉 6. Log Transformation

Several monetary features are right-skewed because a relatively small number of customers can have very large financial values.

The following features are transformed using `log1p`:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`

Histograms are plotted before and after transformation to observe the effect of the transformation and whether the distributions become more symmetrical.

---

# 📏 7. Feature Scaling

The complete set of engineered and transformed features is standardized using **StandardScaler**.

Scaling is important because clustering algorithms are distance-based and features with larger numerical ranges can otherwise have a greater influence on the clustering process.

After scaling, the feature means are approximately **0** and the standard deviations are approximately **1**.

The resulting scaled dataset is stored as:

**`cc_scaled`**

---

# 🔵 8. K-Means Clustering

## Elbow Method

K-Means is evaluated for values of **k from 2 to 10**.

The elbow method uses inertia to observe how the within-cluster variation changes as the number of clusters increases.

The K-Means model uses:

- `k-means++` initialization
- Multiple initialisations
- Fixed random state for reproducibility

## Silhouette Analysis

Silhouette scores are calculated for values of **k from 2 to 10**.

The obtained scores are:

| Number of Clusters | Silhouette Score |
|---:|---:|
| 2 | 0.2032 |
| 3 | 0.2030 |
| 4 | 0.1905 |
| 5 | 0.1854 |
| 6 | 0.1747 |
| 7 | 0.1642 |
| 8 | 0.1659 |
| 9 | 0.1703 |
| 10 | 0.1751 |

Based on the silhouette results and the elbow analysis, **2 clusters** are used for the final K-Means model.

### Final K-Means Configuration

- Number of clusters: **2**
- Initialization: **k-means++**
- `n_init`: **20**
- `max_iter`: **500**
- `random_state`: **42**

The resulting cluster labels are stored in:

**`KMeans_Cluster`**

---

# 📊 9. K-Means Visualisation

The K-Means results are visualised using multiple plots.

### 2D Visualisation 1

**BALANCE vs PURCHASES**

This visualisation shows how customers are distributed according to their balance and purchase behaviour.

### 2D Visualisation 2

**CREDIT_LIMIT vs CASH_ADVANCE**

This visualisation helps identify differences in credit limits and cash-advance behaviour between clusters.

### 3D Visualisation

A 3D visualisation is created using:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`

The cluster labels are represented using different colours.

---

# 👤 10. K-Means Cluster Profiling

The clusters are profiled using the mean values of:

- `BALANCE`
- `PURCHASES`
- `CASH_ADVANCE`
- `CREDIT_LIMIT`
- `PAYMENTS`
- `TENURE`

### Cluster Profile

| Cluster | BALANCE | PURCHASES | CASH_ADVANCE | CREDIT_LIMIT | PAYMENTS | TENURE |
|---|---:|---:|---:|---:|---:|---:|
| Cluster 0 | 1.7962 | 2.0002 | 0.3169 | 2.2046 | 1429.36 | 11.64 |
| Cluster 1 | 2.0877 | 0.7904 | 2.0240 | 2.2019 | 1593.89 | 11.33 |

> **Note:** `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, and `CREDIT_LIMIT` in this table are log-transformed values. They should therefore be interpreted as transformed feature values rather than original currency amounts.

---

# 🧑‍💼 11. Customer Personas

Based on the behavioural characteristics of the K-Means clusters, two broad customer personas are identified.

## Cluster 0 — Transactor

Cluster 0 has relatively higher purchase activity and substantially lower cash-advance activity.

This suggests a customer group that primarily uses the credit card for purchases rather than relying heavily on cash advances.

### Business Insight

This segment can be viewed as purchase-oriented customers whose primary card behaviour is transaction-based rather than cash-advance-based.

---

## Cluster 1 — Cash-Advance Reliant

Cluster 1 has significantly higher cash-advance activity and lower purchase activity compared with Cluster 0.

This indicates a group that relies more heavily on cash advances than regular purchases.

### Business Insight

This segment can be viewed as customers with stronger cash-advance usage behaviour and a different financial usage pattern from the purchase-oriented segment.

---

# 🌳 12. Agglomerative Hierarchical Clustering

Agglomerative Hierarchical Clustering is also applied to the scaled customer data.

A hierarchical clustering dendrogram is created using **Ward linkage**.

The dendrogram is used to examine potential cluster cuts for:

- 3 clusters
- 4 clusters
- 5 clusters

A configuration of **4 clusters** is selected for the Agglomerative model.

---

# 🔗 13. Linkage Method Comparison

Three linkage methods are compared using four clusters:

- Ward
- Complete
- Average

The resulting silhouette scores are:

| Linkage Method | Silhouette Score |
|---|---:|
| Ward | 0.1275 |
| Complete | 0.7936 |
| Average | 0.8267 |

Among the linkage methods tested in the notebook, **Average linkage produced the highest silhouette score**.

Therefore, the final Agglomerative configuration uses:

- Number of clusters: **4**
- Linkage: **Average**

The resulting labels are stored in:

**`Agg_Cluster`**

---

# 📊 14. Agglomerative Visualisation

The Agglomerative clustering results are visualised using:

### 2D Visualisation

**BALANCE vs PURCHASES**

### 3D Visualisation

**BALANCE vs PURCHASES vs CASH_ADVANCE**

These visualisations help examine how the hierarchical clustering groups customers in feature space.

---

# 📏 15. Clustering Evaluation

The clustering models are evaluated using internal clustering metrics.

### Silhouette Score

Measures how similar a customer is to its own cluster compared with other clusters.

**Higher values indicate better-defined separation and cohesion.**

### Davies-Bouldin Index

Measures the similarity between clusters based on their compactness and separation.

**Lower values are generally better.**

### Calinski-Harabasz Index

Measures the relationship between between-cluster dispersion and within-cluster dispersion.

**Higher values are generally better.**

The notebook calculates these metrics for:

- K-Means
- Agglomerative Hierarchical Clustering

A comparison table is generated as part of the model evaluation.

---

# 🔄 16. K-Means Stability Analysis

The stability of the K-Means solution is tested using multiple random states:

- 0
- 7
- 21
- 42
- 99

Silhouette scores are calculated for each random state.

The mean and standard deviation of the scores are then calculated to understand how sensitive the K-Means result is to initialization.

A smaller variation across random states indicates a more consistent clustering result.

---

# 💡 17. Key Business Insights

The analysis identifies two major behavioural patterns through K-Means:

### Purchase-Oriented Customers

These customers show higher purchase activity and relatively low cash-advance usage.

### Cash-Advance-Oriented Customers

These customers show significantly higher cash-advance activity and lower purchase activity.

These behavioural differences can help a credit card business distinguish between customers who primarily use their cards for purchases and customers who rely more heavily on cash advances.

---

# 🛠️ Technologies Used

The project uses the following technologies and Python libraries:

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---
