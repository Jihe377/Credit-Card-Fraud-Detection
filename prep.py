import pandas as pd

data = pd.read_csv("creditcard.csv")
# print(data.head()) print first five rows of the data
#print(data.info()) # Display dataset structure and metadata
#show rows and columns, data type of each, null or not and the usage of memory 
#print(data.describe()) # Generate descriptive statistics for numerical columnsreturn 
# Return count of not null value, average value, standard deviation, Minimum and maximum values and Quartiles (percentiles)  
#print(data.isnull().sum()) # Check for missing values in each column

# Why These Three Steps Matter:
# Data Quality Assessment: Ensures data integrity and completeness
# Understanding Data Distribution: Helps identify patterns and potential issues
# Outlier Detection: Statistical summaries reveal extreme values
# Processing Strategy: Determines if you need to handle missing values, outliers, or data type conversions

# For the credit card fraud dataset specifically, you'll typically find that the data is quite "clean" (no missing values), 
# but there's a significant class imbalance problem where fraudulent transactions represent less than 1% of all transactions.

# print(data['Class'].value_counts()) # Check the distribution of the target variable (fraud vs. normal transactions)

# Severe Class Imbalance
# Normal: 99.83% (~284K transactions)
# Fraud: 0.17% (~500 transactions)
# Ratio: ~578:1

# 1. Evaluation Metrics
# Accuracy is misleading - a model predicting "all normal" gets 99.83% accuracy but is useless
# Focus on: Precision, Recall, F1-Score, AUC-ROC

# 2. Required Data Techniques
# Resampling: SMOTE, undersampling, or oversampling
# Class weights: Adjust algorithm to penalize misclassifying fraud more heavily
# Ensemble methods: Train on balanced subsets

# 3. Model Strategy
# Standard ML algorithms will likely ignore the minority class
# Need specialized techniques for imbalanced classification

# Bottom Line: This step reveals a classic imbalanced classification problem that requires 
# special handling - you can't just apply standard ML methods and expect good fraud detection performance.

import matplotlib.pyplot as plt
import seaborn as sns

# sns.histplot(data['Amount'], bins = 50)
# plt.show()

print(data['Class'].dtype)
# sns.countplot(data=data, x='Class')
plt.show()

print(data.shape)
# 只显示99%的数据范围
# amount_99 = data['Amount'].quantile(0.99)
# plt.figure(figsize=(10, 6))
# sns.histplot(data[data['Amount'] <= amount_99]['Amount'], bins=50) #只保留金额 ≤ 99分位数的交易; 排除极端值：去掉最高的1%交易
# plt.xlabel('Transaction Amount ($)')
# plt.title('Distribution of Transaction Amounts (99th Percentile)')
# plt.show()

# corr = data.corr()
# sns.heatmap(corr,annot=True)
# plt.show()
corr = data.corr()

plt.figure(figsize=(25, 15))
sns.heatmap(corr, annot=True, cmap='RdYlGn')
# plt.show()