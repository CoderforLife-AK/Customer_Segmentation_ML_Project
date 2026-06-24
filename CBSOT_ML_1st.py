import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the excel file
df = pd.read_excel("Customer_churn.xlsx")

#Reading info from excel file
print(df.shape)
print(df.info())

df['Churn Label'].value_counts()
plt.figure(figsize=(8,6))
sns.histplot(df['Tenure Months'], bins = 30, kde = True)
plt.xlabel('Tenure Months')
plt.ylabel('Customer Count')
plt.title('Distribution of Tenure Months vs no of customer')
plt.show()

print(df['Tenure Months'].max())
print(df['Tenure Months'].min())

plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn Label', y='Tenure Months', data=df)
plt.title('Churn vs Tenure Months')
plt.show()

df['Churn Label'].unique()

print(df['Monthly Charges'].max())

df[df['Churn Label'] == 'Yes'] ['Monthly Charges'].quantile([0.25, 0.50, 0.75])

df[df['Churn Label'] == 'No'] ['Monthly Charges'].quantile([0.25, 0.50, 0.75])

df['Monthly Charges'].quantile([0.25, 0.50, 0.75])

df['Contract'].unique()

df['Contract'].value_counts()

plt.figure(figsize=(8,6))
sns.countplot(x = 'Contract', hue = 'Churn Label', data=df)
plt.xlabel('Contract')
plt.ylabel('count')
plt.title('Contract vs count')

df['Payment Method'].unique()

plt.figure(figsize=(8,6))
sns.countplot(x = 'Tech Support', hue = 'Churn Label', data=df)
plt.xlabel('Tech Support')
plt.ylabel('count')
plt.title('Tech Support vs count')
plt.show()

avg_tenure = df.groupby('Churn Label')['Tenure Months'].mean()
print(avg_tenure)

df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

df['Total Charges'].isnull().sum()

df[df['Total Charges'].isnull()]['Tenure Months']

df[df['Total Charges'].isnull()]['Tenure Months'].shape

columns_to_drop = [
    'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long',
    'Latitude', 'Longitude', 'Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'
]
df = df.drop(columns=columns_to_drop)
df.shape
df_encoded = pd.get_dummies(df, drop_first=True)
df_encoded.head()
x = df_encoded.drop('Churn Value', axis = 1)
y = df_encoded['Churn Value']
x.shape
y.shape
print(x)
print(y)

#Machine learning implementation
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
x_train.shape
x_test.shape
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

#Handle class imbalance
rf_balanced = RandomForestClassifier(n_estimators = 100, random_state = 42, class_weight = 'balanced')
rf_balanced.fit(x_train, y_train)
y_pred_balanced = rf_balanced.predict(x_test)
accuracy_balanced = accuracy_score(y_test, y_pred_balanced)
cm_balanced = confusion_matrix(y_test, y_pred_balanced)

print(accuracy_balanced, cm_balanced)
print(classification_report(y_test, y_pred_balanced))

#Hyperparameter tuning
rf_tuned = RandomForestClassifier(n_estimators = 200, random_state = 42, class_weight = 'balanced', max_depth = 10)
rf_tuned.fit(x_train, y_train)
y_pred_tuned = rf_tuned.predict(x_test)

print(classification_report(y_test, y_pred_tuned))

#Feature importance analysis
import pandas as pd
feature_importance = pd.DataFrame({
    'features': x.columns,
    'importance': rf_tuned.feature_importances_
})
feature_importance = feature_importance.sort_values(by = 'importance', ascending = False)
print(feature_importance.tail(15))

x_selected = x.drop(['Phone Service_Yes', 'Multiple Lines_No phone service'], axis = 1)

x_train_sel, x_test_sel, y_train_sel, y_test_sel = train_test_split(x_selected, y, test_size = 0.2, random_state = 42)

rf_selected=RandomForestClassifier(n_estimators=300,max_depth=10,random_state=42,class_weight='balanced')
rf_selected.fit(x_train_sel,y_train_sel)
y_pred_selected=rf_selected.predict(x_test_sel)

print (classification_report(y_test_sel, y_pred_selected))

#Combination of trees and depth
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
n_estimators_list = [100, 200, 300, 400, 500]
max_depth_list = [5, 10, 15]
result = []

for n_trees in n_estimators_list:
    for depth in max_depth_list:
        rf = RandomForestClassifier(n_estimators=n_trees, max_depth=depth, random_state=42, class_weight= 'balanced')
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        result.append(
            {
                'Trees': n_trees,
                'Depth': depth,
                'Accuracy': accuracy,
                'Recall': recall,
                'Precision': precision,
                'F1': f1
            }
        )
    result_df = pd.DataFrame(result)
    result_df = result_df.sort_values(by=['Recall', 'Accuracy'], ascending=False)
    print(result_df)

from sklearn.model_selection import cross_val_score
final_rf = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42, class_weight='balanced')

cv_accuracy = cross_val_score(final_rf, x, y, cv=5, scoring='accuracy')

print(cv_accuracy.mean())
cv_recall = cross_val_score(final_rf, x, y, cv=5, scoring='recall')
print(cv_recall.mean())

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
y_prob = rf_tuned.predict_proba(x_test)

churn_prob = y_prob[:, 1]
fpr, tpr, threshold = roc_curve(y_test, churn_prob)
auc_score = roc_auc_score(y_test, churn_prob)
print(auc_score)

#Customer Segmentation
segmentation_data = pd.DataFrame({
    'Tenure Months': x_test['Tenure Months'],
    'Monthly Charges': x_test['Monthly Charges'],
    'Total Charges': x_test['Total Charges'],
    'churn_prob': churn_prob
})

#Implement K Means
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(segmentation_data)
print(scaled_data[:5])

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i,random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    plt.figure(figsize = (8,6))
plt.plot(range(1, 16), wcss, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i,random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)
    plt.figure(figsize = (8,6))
plt.plot(range(1, 16), wcss, marker = 'o')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
segmentation_data['Cluster'] = clusters

cluster_summary = segmentation_data.groupby('Cluster').mean()
cluster_names = {
    0: 'Budget Loyal Customers',
    1: 'High Risk New Customers',
    2: 'Loyal Premium Customers'
}

segmentation_data['Cluster Segment'] = segmentation_data['Cluster'].map(cluster_names)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Monthly Charges', y='churn_prob', hue='Cluster Segment', data=segmentation_data, palette='Spectral')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Tenure Months', y='churn_prob', hue='Cluster Segment', data=segmentation_data, palette='Spectral')

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Total Charges', y='churn_prob', hue='Cluster Segment', data=segmentation_data, palette='Spectral')
