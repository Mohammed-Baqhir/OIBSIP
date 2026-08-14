# 🧹 Oasis Infobyte – Data Analytics: Data Cleaning Task

This project performs data cleaning, validation, and exploratory analysis on the **Retail Store Sales: Dirty for Data Cleaning** dataset from Kaggle.

This project was completed as part of the **Oasis Infobyte Data Analytics Internship**.

---

## 📌 Project Overview

The original retail sales dataset contains **12,575 rows and 11 columns** with missing values, inconsistent categorical information, and incomplete transaction records.

The project cleans and validates the dataset by recovering missing values where possible, correcting inconsistent transaction values, and validating the relationship between quantity, price, and total spending.

An interactive **Streamlit dashboard** is also included to visualize the cleaning results and explore the cleaned dataset.

---

## 📊 Dataset

**Dataset:** Retail Store Sales: Dirty for Data Cleaning

**Source:** Kaggle

https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning

### Original Dataset

- Rows: **12,575**
- Columns: **11**
- Missing cells: **7,229**
- Duplicate rows: **0**

The dataset follows the transaction relationship:

```text
Total Spent = Quantity × Price Per Unit