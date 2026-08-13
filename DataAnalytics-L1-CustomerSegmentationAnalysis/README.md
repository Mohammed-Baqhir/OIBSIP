# Customer Segmentation Analysis

## Oasis Infobyte — Data Analytics Task 2

An interactive **RFM + K-Means customer segmentation dashboard** for an e-commerce customer base.

## Folder Structure

```text
Customer-Segmentation-Analysis/
│
├── data/
│   └── marketing_campaign.csv
│
├── customer_segmentation_analysis.ipynb
├── app.py
├── README.md
└── requirements.txt
```

## Task Requirements Covered

- Dataset loading and structure inspection
- Missing and inconsistent-data handling
- Average purchase value
- Purchase frequency
- CLV proxy with an explicit limitation note
- RFM feature selection: Recency, Frequency, Monetary
- Log transformation for skewed purchase measures
- StandardScaler
- K-Means clustering
- Elbow Method
- Silhouette Score
- Two cluster scatter plots
- Cluster profiling
- Customers-per-cluster bar chart
- Marketing recommendations
- Interactive Run Clustering control
- Interactive customer-segment prediction
- Jupyter Notebook analytical workflow

## Dataset

The project uses the Marketing Campaign / Customer Personality Analysis dataset.

The CSV file is tab-separated, so the project loads it using:

```python
pd.read_csv("data/marketing_campaign.csv", sep="\t")
```

## RFM Features

### Recency
Existing `Recency` value: days since the customer's last purchase.

### Frequency

```text
NumWebPurchases
+ NumCatalogPurchases
+ NumStorePurchases
```

### Monetary

```text
MntWines
+ MntFruits
+ MntMeatProducts
+ MntFishProducts
+ MntSweetProducts
+ MntGoldProds
```

### Average Purchase Value

```text
Monetary / Frequency
```

### CLV Limitation

An exact Customer Lifetime Value requires transaction history, customer lifespan and usually margin/profit assumptions. Those fields are not available in this customer-level dataset. Therefore, the dashboard uses **historical Monetary spend as a CLV proxy** and explicitly labels it as such.

## Clustering Process

1. Clean the customer data.
2. Build RFM behavioural features.
3. Log-transform Frequency and Monetary to reduce skew.
4. Standardise Recency, Frequency and Monetary using `StandardScaler`.
5. Determine a practical K using the Elbow Method.
6. Run K-Means with `random_state=42` and `n_init=10`.
7. Evaluate the selected K using the Silhouette Score.
8. Profile the resulting clusters.
9. Convert clusters into business-friendly segment labels.
10. Recommend marketing actions.

## Dashboard Features

### Dataset & RFM
Shows:

- row and feature counts
- duplicate rows removed
- missing Income values handled
- invalid values removed
- dataset preview
- descriptive statistics

### Elbow Method
Shows:

- K values 2–8
- inertia
- silhouette scores
- currently selected K

### Cluster Analysis
Shows:

- Frequency vs Monetary
- Recency vs Monetary
- customers per segment

### Cluster Profiles
Shows:

- cluster number
- segment name
- customer count
- mean Recency
- mean Frequency
- mean Monetary
- average purchase value
- CLV proxy
- marketing action

### Predict Segment
Accepts:

- Recency
- Frequency
- Monetary

and assigns the nearest learned K-Means cluster.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Recommended GitHub Structure

```text
OIBSIP/
└── DataAnalytics-L2-CustomerSegmentation/
    ├── data/
    │   └── marketing_campaign.csv
    ├── customer_segmentation_analysis.ipynb
    ├── app.py
    ├── README.md
    └── requirements.txt
```
