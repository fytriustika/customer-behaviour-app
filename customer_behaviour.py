import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, mean_absolute_error, r2_score, confusion_matrix
)
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df1 = pd.read_excel("Online_Retail_Cleaned.xlsx")
    df2 = pd.read_csv("rfm_segmented.csv")
    return df1, df2

df1, df2 = load_data()

# --- DATA CLEANING ---
df1 = df1.dropna()
df1['Quantity'] = pd.to_numeric(df1['Quantity'], errors='coerce')
df1['UnitPrice'] = pd.to_numeric(df1['UnitPrice'], errors='coerce')
df1 = df1[df1['Quantity'] > 0]
df1 = df1[df1['UnitPrice'] > 0]
df1['CustomerID'] = df1['CustomerID'].astype(int)
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'])
df1['TotalPrice'] = df1['Quantity'] * df1['UnitPrice']
df1 = df1.drop_duplicates()

# --- EXECUTIVE KPIs ---
kpi_total_customers = df1['CustomerID'].nunique()
kpi_total_sales = df1['TotalPrice'].sum()
kpi_avg_order_value = df1['TotalPrice'].mean()
kpi_total_orders = df1['InvoiceNo'].nunique()
kpi_top_country = df1['Country'].value_counts().idxmax()

st.title("E-commerce Analytics Dashboard")
st.markdown("#### Executive Summary")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Customers", kpi_total_customers)
col2.metric("Total Sales", f"${kpi_total_sales:,.0f}")
col3.metric("Avg Order Value", f"${kpi_avg_order_value:,.2f}")
col4.metric("Total Orders", kpi_total_orders)
col5.metric("Top Country", kpi_top_country)

# --- RFM ANALYSIS ---
last_date = df1['InvoiceDate'].max()
rfm = df1.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (last_date - x.max()).days,
    'InvoiceNo': pd.Series.nunique,
    'TotalPrice': np.sum
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_score'] = pd.qcut(-rfm['Recency'], 5, labels=False) + 1
rfm['F_score'] = pd.qcut(rfm['Frequency'], 5, labels=False, duplicates='drop') + 1
rfm['M_score'] = pd.qcut(rfm['Monetary'], 5, labels=False, duplicates='drop') + 1
rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

def segment_customer(row):
    score = int(row['RFM_Score'])
    if score >= 555:
        return 'Best Customers'
    elif score >= 454:
        return 'Loyal Customers'
    elif score >= 414:
        return 'Recent but Infrequent'
    elif score >= 333:
        return 'Promising'
    elif score >= 222:
        return 'At Risk'
    elif score >= 111:
        return 'Lost Customers'
    else:
        return 'Others'
rfm['Segment'] = rfm.apply(segment_customer, axis=1)

# --- RFM SEGMENTATION VISUALIZATION ---
st.markdown("#### Customer Segmentation (RFM)")
segment_counts = rfm['Segment'].value_counts().reset_index()
segment_counts.columns = ['Segment', 'Count']
fig_rfm = px.bar(segment_counts, x='Segment', y='Count', color='Segment', title="Customer Segments (RFM)")
st.plotly_chart(fig_rfm, use_container_width=True)

with st.expander("Show RFM segment statistics"):
    st.write(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].describe())

# --- PRODUCT ANALYSIS ---
st.markdown("#### Product Analysis")
product_stats = df1.groupby(['StockCode', 'Description']).agg(
    TotalQuantitySold=('Quantity', 'sum'),
    TotalRevenue=('TotalPrice', 'sum')
).reset_index()
top_selling = product_stats.sort_values(by='TotalQuantitySold', ascending=False).head(10)
most_profitable = product_stats.sort_values(by='TotalRevenue', ascending=False).head(10)

col6, col7 = st.columns(2)
with col6:
    st.markdown("**Top 10 Products by Quantity Sold**")
    fig_prod_qty = px.bar(top_selling, x='Description', y='TotalQuantitySold', color='TotalQuantitySold', title="Top Products by Quantity Sold")
    fig_prod_qty.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_prod_qty, use_container_width=True)
with col7:
    st.markdown("**Top 10 Products by Revenue**")
    fig_prod_rev = px.bar(most_profitable, x='Description', y='TotalRevenue', color='TotalRevenue', title="Top Products by Revenue")
    fig_prod_rev.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_prod_rev, use_container_width=True)

# --- TIME SERIES ANALYSIS ---
st.markdown("#### Time Series Analysis")
df1['InvoiceMonth'] = df1['InvoiceDate'].dt.to_period('M').dt.to_timestamp()
monthly_sales = df1.groupby('InvoiceMonth').agg(
    Orders=('InvoiceNo', pd.Series.nunique),
    Revenue=('TotalPrice', np.sum)
).reset_index()
fig_time = px.line(monthly_sales, x='InvoiceMonth', y='Revenue', title="Monthly Revenue Trend", markers=True)
st.plotly_chart(fig_time, use_container_width=True)

# --- COHORT/RETENTION ANALYSIS ---
st.markdown("#### Cohort Analysis (Customer Retention)")
df1['FirstPurchase'] = df1.groupby('CustomerID')['InvoiceDate'].transform('min')
df1['CohortMonth'] = df1['FirstPurchase'].dt.to_period('M').dt.to_timestamp()
df1['CohortIndex'] = ((df1['InvoiceMonth'].dt.year - df1['CohortMonth'].dt.year) * 12 + 
                      (df1['InvoiceMonth'].dt.month - df1['CohortMonth'].dt.month) + 1)
cohort_counts = df1.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
cohort_pivot = cohort_counts.pivot(index='CohortMonth', columns='CohortIndex', values='CustomerID')
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(cohort_pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
ax.set_title('Customer Retention Heatmap')
st.pyplot(fig)

# --- CHURN PREDICTION ---
st.markdown("#### Churn Prediction (Random Forest)")
churn_threshold = last_date - pd.Timedelta(days=90)
last_purchase = df1.groupby('CustomerID')['InvoiceDate'].max().reset_index()
churned_customers = last_purchase[last_purchase['InvoiceDate'] < churn_threshold]['CustomerID'].tolist()
rfm['Churn'] = rfm['CustomerID'].apply(lambda x: 1 if x in churned_customers else 0)

features = ['Recency', 'Frequency', 'Monetary', 'R_score', 'F_score', 'M_score']
X = rfm[features]
y = rfm['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
y_pred = rf_model.predict(X_test)
auc = roc_auc_score(y_test, y_pred_proba)
st.write(f"Random Forest Churn Model AUC: {auc:.3f}")
fig_roc = go.Figure()
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve'))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(title="ROC Curve - Churn Prediction", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
st.plotly_chart(fig_roc, use_container_width=True)
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
fig_rfimp = px.bar(feat_imp_df, x='feature', y='importance', title="Feature Importance - Churn Model")
st.plotly_chart(fig_rfimp, use_container_width=True)

# --- CLV PREDICTION ---
st.markdown("#### CLV Prediction (Linear Regression)")
rfm['CLV'] = rfm['Monetary']
X_clv = rfm[features]
y_clv = rfm['CLV']
X_train_clv, X_test_clv, y_train_clv, y_test_clv = train_test_split(X_clv, y_clv, test_size=0.2, random_state=42)
lm_clv = LinearRegression()
lm_clv.fit(X_train_clv, y_train_clv)
y_pred_clv = lm_clv.predict(X_test_clv)
mae = mean_absolute_error(y_test_clv, y_pred_clv)
r2 = r2_score(y_test_clv, y_pred_clv)
st.write(f"CLV Model - MAE: {mae:.2f}, R2: {r2:.3f}")
fig_clv_scatter = px.scatter(x=y_test_clv, y=y_pred_clv, labels={'x':'Actual CLV', 'y':'Predicted CLV'}, title="Actual vs Predicted CLV")
st.plotly_chart(fig_clv_scatter, use_container_width=True)

# --- CORRELATION MATRIX ---
st.markdown("#### Correlation Matrix of RFM Features")
corr = rfm[features].corr()
fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Correlation Matrix")
st.plotly_chart(fig_corr, use_container_width=True)

# --- GEO ANALYSIS ---
st.markdown("#### Geo Analysis (Top Countries by Revenue)")
geo_stats = df1.groupby('Country').agg(
    Customers=('CustomerID', 'nunique'),
    Revenue=('TotalPrice', 'sum')
).reset_index().sort_values('Revenue', ascending=False).head(10)
fig_geo = px.bar(geo_stats, x='Country', y='Revenue', color='Revenue', title="Top 10 Countries by Revenue")
st.plotly_chart(fig_geo, use_container_width=True)

# --- INTERACTIVE FILTERING EXAMPLE ---
st.markdown("#### Interactive Customer Segment Filter")
segments = rfm['Segment'].unique().tolist()
selected_segment = st.selectbox("Select Segment", segments)
filtered_rfm = rfm[rfm['Segment'] == selected_segment]
st.dataframe(filtered_rfm.head(20))

st.markdown(
    """
    <hr>
    <small>
    Dashboard by [Your Name] | Powered by Streamlit, scikit-learn, pandas, plotly, seaborn, and matplotlib.
    </small>
    """,
    unsafe_allow_html=True
)