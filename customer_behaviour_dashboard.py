import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LinearRegression
import datetime as dt

st.set_page_config(page_title="Customer Behaviour Dashboard", layout="wide")

st.title("E-commerce Customer Behaviour Dashboard")

@st.cache_data
def load_data():
    # You can change these to file uploader for more flexibility
    df1 = pd.read_excel('Online_Retail_Cleaned.xlsx')
    df2 = pd.read_csv('rfm_segmented.csv')
    return df1, df2

df1, df2 = load_data()

# Data Cleaning
df1.dropna(inplace=True)
df1['Quantity'] = pd.to_numeric(df1['Quantity'], errors='coerce')
df1['UnitPrice'] = pd.to_numeric(df1['UnitPrice'], errors='coerce')
df1 = df1[(df1['Quantity'] > 0) & (df1['UnitPrice'] > 0)]
df1.dropna(subset=['CustomerID'], inplace=True)
df1['CustomerID'] = df1['CustomerID'].astype(int)
df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']
df1 = df1.drop_duplicates()

# Outlier removal (IQR)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
    if col in df1.columns:
        df1 = remove_outliers_iqr(df1, col)

for col in ['Recency', 'Frequency', 'Monetary']:
    if col in df2.columns:
        df2 = remove_outliers_iqr(df2, col)

# RFM Calculation
def calculate_rfm(df):
    last_purchase = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()
    ref_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    last_purchase['Recency'] = (ref_date - last_purchase['InvoiceDate']).dt.days
    freq = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index().rename(columns={'InvoiceNo': 'Frequency'})
    monetary = df.groupby('CustomerID')['TotalPrice'].sum().reset_index().rename(columns={'TotalPrice': 'Monetary'})
    rfm = last_purchase[['CustomerID', 'Recency']].merge(freq, on='CustomerID').merge(monetary, on='CustomerID')
    return rfm

rfm_df = calculate_rfm(df1)

# Score RFM
def score_rfm(df, column, is_recency=False):
    scored_data, bins = pd.qcut(df[column], 5, labels=False, retbins=True, duplicates='drop')
    num_bins = len(bins) - 1
    if is_recency:
        labels = list(range(num_bins, 0, -1))
    else:
        labels = list(range(1, num_bins + 1))
    score_map = {i: labels[i] for i in range(num_bins)}
    final_scores = scored_data.map(score_map)
    return final_scores

rfm_df['R_score'] = score_rfm(rfm_df, 'Recency', is_recency=True)
rfm_df['F_score'] = score_rfm(rfm_df, 'Frequency')
rfm_df['M_score'] = score_rfm(rfm_df, 'Monetary')
rfm_df['RFM_Score'] = rfm_df['R_score'].astype(str) + rfm_df['F_score'].astype(str) + rfm_df['M_score'].astype(str)
rfm_df['RFM_Score'] = rfm_df['RFM_Score'].astype(int)

def segment_customer(rfm_score):
    if rfm_score >= 555:
        return 'Best Customers'
    elif rfm_score >= 454:
        return 'Loyal Customers'
    elif rfm_score >= 414:
        return 'Recent but Infrequent'
    elif rfm_score >= 333:
        return 'Promising'
    elif rfm_score >= 222:
        return 'At Risk'
    elif rfm_score >= 111:
        return 'Lost Customers'
    else:
        return 'Others'

rfm_df['Segment'] = rfm_df['RFM_Score'].apply(segment_customer)

# Churn detection
obs_end = df1['InvoiceDate'].max()
churn_threshold = obs_end - dt.timedelta(days=90)
last_tx = df1.groupby('CustomerID')['InvoiceDate'].max().reset_index()
churned_customers = last_tx[last_tx['InvoiceDate'] < churn_threshold]['CustomerID'].values
rfm_df['Churn'] = rfm_df['CustomerID'].apply(lambda x: 1 if x in churned_customers else 0)

# Customer Lifetime Value (CLV)
rfm_df['CLV'] = rfm_df['Monetary']

# Add country info
customer_country = df1.groupby('CustomerID')['Country'].agg(lambda x: x.mode()[0]).reset_index()
rfm_df = rfm_df.merge(customer_country, on='CustomerID', how='left')

# Product Analysis
product_quantity = df1.groupby('StockCode')['Quantity'].sum().reset_index().rename(columns={'Quantity': 'TotalQuantitySold'})
product_revenue = df1.groupby('StockCode')['TotalPrice'].sum().reset_index().rename(columns={'TotalPrice': 'TotalRevenue'})
product_analysis_df = product_quantity.merge(product_revenue, on='StockCode')

# Streamlit Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Customer Segments", 
    "Product Analysis", 
    "Churn Prediction", 
    "CLV Prediction"
])

with tab1:
    st.header("Customer Segmentation")
    seg_counts = rfm_df['Segment'].value_counts()
    st.bar_chart(seg_counts)
    seg = st.selectbox("Select Segment", seg_counts.index)
    seg_df = rfm_df[rfm_df['Segment'] == seg]
    st.write(f"Descriptive Statistics for {seg} Segment")
    st.dataframe(seg_df[['Recency', 'Frequency', 'Monetary']].describe())

    # Top 10 products for this segment
    merged_df = pd.merge(df1, rfm_df, on='CustomerID', how='inner')
    top_products = merged_df[merged_df['Segment'] == seg]['Description'].value_counts().head(10)
    st.write(f"Top 10 Products Purchased by {seg} Segment")
    st.dataframe(top_products)

    # Country distribution
    st.write(f"Country Distribution for {seg} Segment")
    country_counts = merged_df[merged_df['Segment'] == seg]['Country'].value_counts()
    st.bar_chart(country_counts)

with tab2:
    st.header("Product Analysis")
    n = st.slider("Show top N products", 5, 20, 10)
    top_qty = product_analysis_df.sort_values(by='TotalQuantitySold', ascending=False).head(n)
    top_rev = product_analysis_df.sort_values(by='TotalRevenue', ascending=False).head(n)
    st.subheader("Top Products by Quantity Sold")
    st.dataframe(top_qty)
    fig, ax = plt.subplots()
    sns.barplot(x='StockCode', y='TotalQuantitySold', data=top_qty, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Top Products by Revenue")
    st.dataframe(top_rev)
    fig2, ax2 = plt.subplots()
    sns.barplot(x='StockCode', y='TotalRevenue', data=top_rev, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Common products
    common_codes = set(top_qty['StockCode']) & set(top_rev['StockCode'])
    st.write("Products in both Top Quantity and Top Revenue:")
    st.dataframe(product_analysis_df[product_analysis_df['StockCode'].isin(common_codes)])

with tab3:
    st.header("Churn Prediction")
    features = ['Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_Score']
    X = rfm_df[features]
    y = rfm_df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("Model Performance")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    st.write(f"Precision: {precision_score(y_test, y_pred):.4f}")
    st.write(f"Recall: {recall_score(y_test, y_pred):.4f}")
    st.write(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    st.write(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")

with tab4:
    st.header("CLV Prediction")
    features_clv = ['Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score', 'RFM_Score']
    X_clv = rfm_df[features_clv]
    y_clv = rfm_df['CLV']
    X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(X_clv, y_clv, test_size=0.2, random_state=42)
    model_clv = LinearRegression()
    model_clv.fit(X_train_clv, y_train_clv)
    y_pred_clv = model_clv.predict(X_test_clv)
    st.write("Regression Metrics")
    st.write(f"MAE: {mean_absolute_error(y_test_clv, y_pred_clv):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test_clv, y_pred_clv):.2f}")
    st.write(f"R2: {r2_score(y_test_clv, y_pred_clv):.2f}")

st.info("Dashboard by Streamlit. Customize the code for more features and upload your own data to get started!")