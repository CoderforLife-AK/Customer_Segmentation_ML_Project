📊 **Customer Churn & Segmentation Analysis**

📝 **Project Overview**

This project focuses on identifying patterns in customer churn within a fictional telecommunications company using the industry-standard IBM Telco Customer Churn Dataset. Operating in a highly competitive telecom environment, retaining existing customers is significantly more cost-effective than acquiring new ones.

By conducting deep Exploratory Data Analysis (EDA) and profiling target segments, this repository serves as a foundational data science pipeline to pinpoint exactly why and when customers are likely to discontinue service.

🗂️ **Dataset Architecture**

The workflow parses an enterprise-level dataset containing 7,043 unique customer entries mapped across 33 distinct features. The dataset covers four critical customer dimensions:

Demographics: Gender, Senior Citizen status, Partner, and Dependents.

Geographics: Latitudinal/Longitudinal boundaries, City, and Zip Code mapping.

Account/Financial Profiles: Tenure (in months), Contract types, Monthly Charges, Total Charges, and Customer Lifetime Value (CLTV).

Services Enrolled: Phone lines, Streaming preferences, and Internet packages.

Class Balance (Target Variable: Churn Label)
Active Customers (No): 5,174 subscribers

Churned Customers (Yes): 1,869 subscribers (~26.5% churn rate)

🔬 **Core Workflow & Implementation**

1. Data Ingestion & Integrity Audits
Implemented automated error handling using pandas to interface with multi-column spreadsheet architectures (.xlsx).

Generated programmatic integrity checks (.shape, .info(), and .value_counts()) to trace data types and surface categorical layouts.

2. Analytical Visualizations
Leveraged seaborn and matplotlib to isolate core customer metrics.

Constructed a localized Kernel Density Estimate (KDE) over customer retention distribution matrices to evaluate the direct correlation between user lifetime duration (Tenure Months) and account cancellation probability.

📈 **Key Discovery Highlight**

The Early-Stage Churn Risk: Initial analytical histograms indicate a distinct, highly concentrated volume of customer attrition during the first 1–5 months of tenure. As customer loyalty passes the initial threshold, retention curves stabilize significantly, indicating that onboarding experiences or early-stage contract configurations heavily influence initial customer retention.

⚙️ **Tech Stack & Requirements**

To execute this analytical notebook or the converted deployment scripts locally, install the following foundational data engineering frameworks:

Bash
pip install pandas numpy matplotlib seaborn openpyxl

🚀 **Future Roadmap**

Feature Engineering: Encode high-cardinality categorical metrics (e.g., Contract types, Payment Methods) using One-Hot encoding.

Imbalance Handling: Apply SMOTE (Synthetic Minority Over-sampling Technique) to counteract the dataset's unequal class distributions.

Predictive Modeling: Benchmark Scikit-Learn ensemble models (Random Forests, Gradient Boosting) to compute structural risk coefficients (Churn Score).
