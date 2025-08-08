# Load libraries
library(tidyverse)
library(lubridate)
library(caret)
library(randomForest)
library(ggplot2)
library(cowplot)
library(reshape2)
library(readxl)
library(pROC)

# --- LOAD DATA ---
df1 <- read_excel("Online_Retail_Cleaned.xlsx")
df2 <- read.csv("rfm_segmented.csv")

# --- DATA CLEANING ---
df1 <- df1 %>%
  drop_na() %>%
  mutate(
    Quantity = as.numeric(Quantity),
    UnitPrice = as.numeric(UnitPrice),
    CustomerID = as.integer(CustomerID),
    InvoiceDate = as.POSIXct(InvoiceDate)
  ) %>%
  filter(Quantity > 0, UnitPrice > 0)

df1 <- df1 %>% mutate(TotalPrice = Quantity * UnitPrice)
df1 <- df1 %>% distinct()

# --- EXECUTIVE KPIs ---
kpi_total_customers <- n_distinct(df1$CustomerID)
kpi_total_sales <- sum(df1$TotalPrice)
kpi_avg_order_value <- mean(df1$TotalPrice)
kpi_total_orders <- n_distinct(df1$InvoiceNo)
kpi_top_country <- df1 %>% count(Country, sort=TRUE) %>% slice(1) %>% pull(Country)
cat("KPI Customers:", kpi_total_customers, "\n")
cat("KPI Sales:", kpi_total_sales, "\n")
cat("KPI Avg Order Value:", kpi_avg_order_value, "\n")
cat("KPI Orders:", kpi_total_orders, "\n")
cat("KPI Top Country:", kpi_top_country, "\n\n")

# --- RFM ANALYSIS ---
last_date <- max(df1$InvoiceDate)
rfm <- df1 %>%
  group_by(CustomerID) %>%
  summarise(
    Recency = as.numeric(difftime(last_date, max(InvoiceDate), units = "days")),
    Frequency = n_distinct(InvoiceNo),
    Monetary = sum(TotalPrice)
  )

rfm <- rfm %>%
  mutate(
    R_score = ntile(-Recency, 5),
    F_score = ntile(Frequency, 5),
    M_score = ntile(Monetary, 5),
    RFM_Score = as.integer(paste0(R_score, F_score, M_score))
  ) %>%
  mutate(
    Segment = case_when(
      RFM_Score >= 555 ~ "Best Customers",
      RFM_Score >= 454 ~ "Loyal Customers",
      RFM_Score >= 414 ~ "Recent but Infrequent",
      RFM_Score >= 333 ~ "Promising",
      RFM_Score >= 222 ~ "At Risk",
      RFM_Score >= 111 ~ "Lost Customers",
      TRUE ~ "Others"
    )
  )

# --- RFM SEGMENTATION VISUALIZATION ---
segment_counts <- rfm %>% count(Segment, sort=TRUE)
ggplot(segment_counts, aes(x=reorder(Segment, -n), y=n, fill=Segment)) +
  geom_bar(stat="identity") +
  labs(title="Customer Segments (RFM)", x="Segment", y="Number of Customers") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# Drill-down: Example for most populated segment
top_segment <- segment_counts$Segment[1]
rfm_top <- rfm %>% filter(Segment == top_segment)
summary(rfm_top[c("Recency", "Frequency", "Monetary")])

# --- PRODUCT ANALYSIS ---
product_stats <- df1 %>%
  group_by(StockCode, Description) %>%
  summarise(
    TotalQuantitySold = sum(Quantity),
    TotalRevenue = sum(TotalPrice),
    .groups = "drop"
  )
top_selling <- product_stats %>% arrange(desc(TotalQuantitySold)) %>% slice_head(n = 10)
most_profitable <- product_stats %>% arrange(desc(TotalRevenue)) %>% slice_head(n = 10)

ggplot(top_selling, aes(x=reorder(Description, -TotalQuantitySold), y=TotalQuantitySold)) +
  geom_bar(stat="identity", fill="steelblue") +
  labs(title="Top 10 Products by Quantity Sold", x="", y="Quantity Sold") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

ggplot(most_profitable, aes(x=reorder(Description, -TotalRevenue), y=TotalRevenue)) +
  geom_bar(stat="identity", fill="darkred") +
  labs(title="Top 10 Products by Revenue", x="", y="Revenue") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# --- TIME SERIES ANALYSIS ---
df1 <- df1 %>%
  mutate(InvoiceMonth = floor_date(InvoiceDate, "month"))

monthly_sales <- df1 %>%
  group_by(InvoiceMonth) %>%
  summarise(
    Orders = n_distinct(InvoiceNo),
    Revenue = sum(TotalPrice)
  )

ggplot(monthly_sales, aes(x=InvoiceMonth, y=Revenue)) +
  geom_line() + geom_point() +
  labs(title="Monthly Revenue Trend", x="Month", y="Revenue")

# --- COHORT/RETENTION ANALYSIS ---
df1 <- df1 %>%
  mutate(FirstPurchase = df1 %>% group_by(CustomerID) %>% mutate(First= min(InvoiceDate)) %>% pull(First),
         CohortMonth = floor_date(FirstPurchase, "month"),
         InvoiceMonth = floor_date(InvoiceDate, "month"),
         CohortIndex = as.numeric(difftime(InvoiceMonth, CohortMonth, units="weeks")) %/% 4 + 1)

cohort_counts <- df1 %>%
  group_by(CohortMonth, CohortIndex) %>%
  summarise(Customers = n_distinct(CustomerID), .groups="drop")

cohort_matrix <- cohort_counts %>% 
  pivot_wider(names_from = CohortIndex, values_from = Customers) %>%
  column_to_rownames("CohortMonth") %>% as.matrix()

heatmap(cohort_matrix, Rowv=NA, Colv=NA, scale="row", 
        main="Customer Retention Heatmap (Cohort Analysis)", xlab="Cohort Period", ylab="Cohort Month")

# --- CHURN PREDICTION ---
churn_threshold <- last_date - 90
churned_customers <- df1 %>%
  group_by(CustomerID) %>%
  summarise(last_purchase = max(InvoiceDate)) %>%
  filter(last_purchase < churn_threshold) %>%
  pull(CustomerID)

rfm <- rfm %>%
  mutate(Churn = ifelse(CustomerID %in% churned_customers, 1, 0))

features <- c("Recency", "Frequency", "Monetary", "R_score", "F_score", "M_score", "RFM_Score")
train_idx <- createDataPartition(rfm$Churn, p = 0.8, list = FALSE)
train_data <- rfm[train_idx, ]
test_data <- rfm[-train_idx, ]

rf_model <- randomForest(as.factor(Churn) ~ ., data = train_data[,c(features, "Churn")])
pred <- predict(rf_model, newdata = test_data, type="prob")[,2]
roc_obj <- roc(test_data$Churn, pred)
auc <- auc(roc_obj)
cat("Churn model AUC:", auc, "\n")
plot(roc_obj, main="ROC Curve - Churn Prediction")

imp <- importance(rf_model)
barplot(imp[,1], names.arg=rownames(imp), main="Churn Model Feature Importance", las=2)

# --- CLV PREDICTION ---
rfm$CLV <- rfm$Monetary
train_idx_clv <- createDataPartition(rfm$CLV, p = 0.8, list = FALSE)
train_data_clv <- rfm[train_idx_clv, ]
test_data_clv <- rfm[-train_idx_clv, ]
lm_clv <- lm(CLV ~ Recency + Frequency + Monetary + R_score + F_score + M_score + RFM_Score, data = train_data_clv)
pred_clv <- predict(lm_clv, newdata = test_data_clv)
cat("CLV Model - MAE:", mean(abs(pred_clv - test_data_clv$CLV)), "\n")
cat("CLV Model - R2:", summary(lm_clv)$r.squared, "\n")

# --- CORRELATION MATRIX ---
corr_vars <- rfm %>% select(all_of(features))
corr_matrix <- cor(corr_vars)
melted_corr <- melt(corr_matrix)
ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(midpoint=0, low="blue", mid="white", high="red", limits=c(-1,1)) +
  labs(title="Correlation Matrix", x="", y="") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# --- GEO ANALYSIS ---
geo_stats <- df1 %>%
  group_by(Country) %>%
  summarise(
    Customers = n_distinct(CustomerID),
    Revenue = sum(TotalPrice)
  ) %>%
  arrange(desc(Revenue)) %>% slice_head(n=10)

ggplot(geo_stats, aes(x=reorder(Country, Revenue), y=Revenue, fill=Country)) +
  geom_bar(stat="identity") +
  labs(title="Top 10 Countries by Revenue", x="Country", y="Revenue") +
  theme(axis.text.x = element_text(angle=45, hjust=1))

# --- ADVANCED: INTERACTIVE FILTERING EXAMPLE ---
# For a Shiny dashboard, you’d use input widgets to filter by date, segment, etc.
# Example filter:
# filtered_df <- df1 %>% filter(InvoiceDate >= as.Date("2011-01-01"), InvoiceDate <= as.Date("2011-12-31"))