# ============================================================
# 02 — EXPLORATORY DATA ANALYSIS
# ============================================================

library(dplyr)
library(e1071)
library(corrplot)

joined_clean <- readRDS("data/processed/joined_clean.rds")
dir.create("figures", showWarnings = FALSE)

numeric_cols <- c("Sales", "Profit", "Quantity", "Discount")

# Descriptive statistics.
for (col in numeric_cols) {
  cat("\n---", col, "---\n")
  cat(
    "Mean:", mean(joined_clean[[col]], na.rm = TRUE),
    "Median:", median(joined_clean[[col]], na.rm = TRUE),
    "SD:", sd(joined_clean[[col]], na.rm = TRUE),
    "Skewness:", skewness(joined_clean[[col]], na.rm = TRUE),
    "\n"
  )
}

# Sales distribution before and after log transformation.
png("figures/sales_distribution_raw.png", 1000, 700)
hist(joined_clean$Sales, breaks = 50,
     main = "Distribution of Sales", xlab = "Sales")
dev.off()

png("figures/sales_distribution_log.png", 1000, 700)
hist(joined_clean$Sales_log, breaks = 50,
     main = "Distribution of Log-Transformed Sales",
     xlab = "log(1 + Sales)")
dev.off()

# Outlier inspection.
png("figures/numeric_boxplots.png", 1200, 900)
par(mfrow = c(2, 2))
for (col in numeric_cols) {
  boxplot(joined_clean[[col]], main = paste("Boxplot of", col))
}
dev.off()
par(mfrow = c(1, 1))

cat(
  "Loss-making transactions (%):",
  mean(joined_clean$loss_making, na.rm = TRUE) * 100, "\n"
)

# Transaction-level correlations.
cor_matrix <- cor(joined_clean[, numeric_cols], use = "complete.obs")
print(round(cor_matrix, 2))

png("figures/transaction_correlation_matrix.png", 1000, 900)
corrplot(cor_matrix, method = "color", addCoef.col = "black")
dev.off()

# Return-pattern checks.
return_by_segment <- joined_clean %>%
  group_by(Segment) %>%
  summarise(return_rate = mean(returned), .groups = "drop")

return_by_shipping <- joined_clean %>%
  group_by(`Ship Mode`) %>%
  summarise(return_rate = mean(returned), .groups = "drop")

discount_by_return <- joined_clean %>%
  group_by(returned) %>%
  summarise(average_discount = mean(Discount, na.rm = TRUE),
            .groups = "drop")

print(return_by_segment)
print(return_by_shipping)
print(discount_by_return)

# Technology-category exploration.
technology_summary <- joined_clean %>%
  filter(Category == "Technology") %>%
  summarise(
    transactions = n(),
    unique_orders = n_distinct(`Order ID`),
    unique_customers = n_distinct(`Customer ID`),
    returned_orders = n_distinct(`Order ID`[returned == 1])
  )

print(technology_summary)
