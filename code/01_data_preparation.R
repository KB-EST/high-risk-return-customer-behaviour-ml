# ============================================================
# 01 — DATA PREPARATION
# Using Machine learning model to predict High-Risk Return Customer Behaviour 
# ============================================================

library(readxl)
library(duckdb)
library(DBI)
library(dplyr)

set.seed(123)

# Expected location of the original workbook.
data_path <- file.path("data", "Sample - Superstore.xls")

orders  <- read_excel(data_path, sheet = "Orders")
returns <- read_excel(data_path, sheet = "Returns")

cat("Orders dimensions:", dim(orders), "\n")
cat("Returns dimensions:", dim(returns), "\n")
cat("Duplicate Return Order IDs:", any(duplicated(returns$`Order ID`)), "\n")
cat("Unique returned Order IDs:", n_distinct(returns$`Order ID`), "\n")

# Integrate Orders and Returns using DuckDB.
con <- dbConnect(duckdb())
duckdb_register(con, "orders", orders)
duckdb_register(con, "returns", returns)

joined <- dbGetQuery(con, '
  SELECT
    o.*,
    CASE WHEN r."Order ID" IS NOT NULL THEN 1 ELSE 0 END AS returned
  FROM orders o
  LEFT JOIN (
    SELECT DISTINCT "Order ID"
    FROM returns
  ) r
    ON o."Order ID" = r."Order ID"
')

dbDisconnect(con, shutdown = TRUE)

# Join sanity check.
stopifnot(nrow(joined) == nrow(orders))
print(table(joined$returned))

# Remove fields not required for the analysis.
joined_clean <- joined[, !(names(joined) %in% c(
  "Customer Name", "Country/Region", "City",
  "State", "Postal Code", "Product Name"
))]

# Data-quality checks.
print(sapply(joined_clean, function(x) sum(is.na(x))))
cat("Duplicate Row IDs:", sum(duplicated(joined_clean$`Row ID`)), "\n")

# Transaction-level derived variables.
joined_clean <- joined_clean %>%
  mutate(
    Sales_log = log1p(Sales),
    loss_making = ifelse(Profit < 0, 1, 0)
  )

dir.create("data/processed", recursive = TRUE, showWarnings = FALSE)
saveRDS(joined_clean, "data/processed/joined_clean.rds")

cat("Prepared data saved to data/processed/joined_clean.rds\n")
