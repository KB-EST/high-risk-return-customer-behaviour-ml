# ============================================================
# 03 — FEATURE ENGINEERING AND TARGET CONSTRUCTION
# ============================================================

library(dplyr)
library(psych)

joined_clean <- readRDS("data/processed/joined_clean.rds")

# Transform transaction-level records into customer-level behaviour.
customer_features <- joined_clean %>%
  group_by(`Customer ID`) %>%
  summarise(
    total_orders = n_distinct(`Order ID`),

    # Used ONLY to construct the target; excluded from predictors.
    returned_orders = n_distinct(`Order ID`[returned == 1]),
    return_rate = returned_orders / total_orders,

    aov = mean(Sales_log, na.rm = TRUE),
    category_conc = max(table(Category)) / n(),
    discount_seeking = mean(Discount, na.rm = TRUE),
    loss_making_rate = mean(loss_making, na.rm = TRUE),
    fast_shipping_share =
      mean(`Ship Mode` %in% c("Same Day", "First Class")),
    recency = as.numeric(
      max(joined_clean$`Order Date`) - max(`Order Date`)
    ),
    total_spend = sum(Sales, na.rm = TRUE),
    sales_variability = sd(Sales, na.rm = TRUE),
    .groups = "drop"
  )

# Final dissertation proxy target:
# high-return = return rate >= 20% OR >= 2 distinct returned orders.
customer_features <- customer_features %>%
  mutate(
    high_return_customer = factor(
      ifelse(return_rate >= 0.20 | returned_orders >= 2, 1, 0),
      levels = c(1, 0)
    )
  )

print(table(customer_features$high_return_customer))
print(prop.table(table(customer_features$high_return_customer)) * 100)

# Candidate behavioural variables: redundancy and PCA-suitability checks.
candidate_predictors <- customer_features %>%
  select(
    total_orders, aov, category_conc, discount_seeking,
    loss_making_rate, fast_shipping_share, recency,
    total_spend, sales_variability
  )

candidate_cor <- cor(candidate_predictors, use = "complete.obs")
print(round(candidate_cor, 2))
print(KMO(candidate_cor))

# Final leakage-free predictor set.
# return_rate and returned_orders are deliberately excluded because
# they directly define the target.
model_data <- customer_features %>%
  select(
    total_orders,
    aov,
    category_conc,
    discount_seeking,
    fast_shipping_share,
    recency,
    total_spend,
    high_return_customer
  )

stopifnot(!"return_rate" %in% names(model_data))
stopifnot(!"returned_orders" %in% names(model_data))

print(table(model_data$high_return_customer))

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
saveRDS(customer_features, "data/processed/customer_features.rds")
saveRDS(model_data, "data/processed/model_data.rds")
