Project Title: E-commerce Customer Behavior and Product Performance Analysis

Executive Summary: This project conducts a comprehensive analysis of e-commerce transaction data to understand customer behavior and identify high-performing products. By applying data cleaning techniques, RFM (Recency, Frequency, Monetary) analysis, customer segmentation, and product analysis, we successfully identified key customer segments, assessed the health of the customer base, and discovered products contributing the most to sales volume and revenue. These findings provide valuable insights for targeted marketing strategies, customer retention efforts, and product offering optimization.


Project Overview
    
Background: E-commerce businesses generate vast amounts of transaction and customer data. Analyzing this data is crucial for understanding customer behavior, identifying valuable customers, evaluating product performance, and ultimately driving business growth and profitability. This project utilizes a real-world e-commerce dataset to perform a detailed analysis of customer transactions and product sales.

Problem: The business needs to gain deeper insights into its customer base and product performance to make data-driven decisions. Specifically, there is a need to:

- Understand the different types of customers based on their purchasing patterns.
- Identify customers who are likely to stop purchasing (churn).
- Assess which products are most successful in terms of sales volume and revenue.
- Uncover actionable insights to improve marketing strategies, customer retention efforts, and product management.


Goal: The primary goal of this project is to analyze e-commerce customer behavior and product performance to extract meaningful insights that can inform business strategies and improve overall business outcomes.

Objectives: To achieve the project goal, the following objectives were pursued:

- Load, clean, and validate the e-commerce transaction and customer data.
- Perform RFM analysis to quantify customer value based on their purchasing activity.
- Segment customers into distinct groups based on their RFM scores.
- Analyze product performance by calculating total sales quantity and revenue for each product.
- Identify characteristics of key customer segments and high-performing products.
- Summarize the key findings and insights in a clear and concise manner.

Dataset Information:

The dataset consists of two files:
1. Online_Retail_Cleaned.xlsx (loaded into df1):
- Initial data count: (409371 rows, 11 columns).
- Main columns include: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, TotalPrice, and CheckBlank.
- Data types vary, including float64 (for IDs and numerical values), object (for codes and descriptions), and datetime64[ns] (for invoice dates).
- There are missing values in several columns such as InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, Unnamed: 8, and TotalPrice.
- Initial descriptive statistics show the range of values for numerical columns, including potential outliers in Quantity, UnitPrice, and TotalPrice (maximum values are significantly higher than the 3rd quartile).
2. rfm_segmented.csv (loaded into df2):
- Data count: (4334 rows, 9 columns).
- Columns include: CustomerID, Recency, Frequency, Monetary, R_score, F_score, M_score, RFM_Score, and Segment.
- Data types are predominantly integer (for IDs and RFM scores) and float64 (for Monetary).
- There are no missing values in this dataset.
- Descriptive statistics show the distribution of RFM metrics and scores. The Frequency and Monetary columns also indicate potential outliers based on their maximum values.
- Overall, df1 contains detailed transaction data, while df2 contains pre-calculated aggregated RFM and customer segmentation data. df1 requires handling of missing values and outliers, while df2 is relatively clean but still shows outliers in the RFM and Monetary metrics.


Analysis Process:

A. Data Loading & Cleaning:

 1. E-commerce transaction data and RFM segmentation data were loaded from .xlsx and .csv files.
 2. Thorough data cleaning steps were performed, including handling missing values, correcting data types, and identifying and removing outliers using the Interquartile Range (IQR) method. This ensured high data quality for subsequent analysis.
 3. Duplicate rows found in the transaction data were also removed.
 4. RFM Analysis & Customer Segmentation:
 5. RFM metrics were calculated for each customer to measure how recently (Recency), how frequently (Frequency), and how much money (Monetary) they spent.
 6. Based on the RFM scores, customers were segmented into meaningful categories such as 'Best Customers', 'Loyal Customers', 'At Risk', and 'Lost Customers'. This segmentation allowed for a better understanding of the customer base composition.
 7. Product Analysis:
 8. Product analysis was conducted to evaluate the performance of each product based on the total quantity sold and the total revenue generated.
 9. The top products by both metrics were identified to highlight the most popular and profitable items.

B. Key Findings (Data Insights):
  1. Customer Segment Composition: The RFM segmentation analysis revealed the distribution of customers across different segments. The 'Lost Customers' segment was the largest segment in this dataset, indicating a key challenge area for re-engagement strategies.
  2. Characteristics of the 'Lost Customers' Segment: Further investigation into the 'Lost Customers' segment showed that a significant portion of customers in this segment are located in the United Kingdom and tend to purchase certain types of general household and gift items. This insight is crucial for designing targeted campaigns to bring them back.
  3. Key Products with Dual Performance: Five products were consistently found in the top 10 lists for both quantity sold and total revenue. These products ('23209', '84879', '20725', '85123A', and '85099B') are likely the backbone of sales volume and profitability.
  4. Performance Comparison of Key Products: Among these dual-performing products, '84879' stood out for its high sales quantity, while '85123A' generated the highest total revenue, indicating a higher monetary value per unit or significant sales volume combined with good pricing.

C. Conclusion & Recommendations:
    This analysis provides deep insights into customer behavior and product performance. The key findings, such as the customer segment distribution and the identification of key products, can be used to inform business strategies.

D. For next steps, one could consider:

  1. Targeted Marketing Strategies: Developing specific campaigns for different customer segments, especially for the 'Lost Customers' segment to encourage re-engagement.
  2. Product Optimization: Further analyzing the success factors of the top dual-performing products and applying these insights to other products.
  3. Predictive Modeling: Building churn prediction and/or CLV models (as we planned) to forecast future customer behavior.
  4. Dashboard Creation: Building an interactive dashboard to visualize these key findings and share them with stakeholders.
  5. This project demonstrates proficiency in data cleaning, customer behavior analysis using RFM, segmentation, and product performance analysis, all of which are essential skills in e-commerce data analysis.
