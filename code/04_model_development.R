# ============================================================
# 04 — MODEL DEVELOPMENT
# ============================================================

library(tidymodels)
library(themis)
library(ranger)
library(xgboost)
library(dplyr)

set.seed(123)

model_data <- readRDS("data/processed/model_data.rds")
model_data$high_return_customer <- factor(
  model_data$high_return_customer,
  levels = c("1", "0")
)

# 80/20 stratified split. The test set remains untouched during tuning.
data_split <- initial_split(
  model_data,
  prop = 0.80,
  strata = high_return_customer
)

train_data <- training(data_split)
test_data  <- testing(data_split)

print(table(train_data$high_return_customer))
print(table(test_data$high_return_customer))

# Five-fold stratified cross-validation.
set.seed(123)
cv_folds <- vfold_cv(
  train_data,
  v = 5,
  strata = high_return_customer
)

model_metrics <- metric_set(
  roc_auc, pr_auc, accuracy, sens, precision, f_meas
)

# SMOTE is applied inside resampling, not before the split.
make_smote_recipe <- function(ratio) {
  recipe(high_return_customer ~ ., data = train_data) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_smote(high_return_customer, over_ratio = ratio)
}

smote_ratios <- c("0.30" = 0.30, "0.50" = 0.50,
                  "0.75" = 0.75, "1.00" = 1.00)
recipes <- lapply(smote_ratios, make_smote_recipe)

# Logistic Regression.
log_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

set.seed(123)
log_results <- lapply(names(recipes), function(ratio) {
  workflow() %>%
    add_recipe(recipes[[ratio]]) %>%
    add_model(log_model) %>%
    fit_resamples(
      resamples = cv_folds,
      metrics = model_metrics,
      control = control_resamples(save_pred = TRUE)
    )
})
names(log_results) <- names(recipes)

# Random Forest.
rf_model <- rand_forest(
  trees = 500,
  mtry = tune(),
  min_n = tune()
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

rf_grid <- grid_regular(
  mtry(range = c(1L, 7L)),
  min_n(range = c(2L, 20L)),
  levels = 4
)

set.seed(123)
rf_results <- lapply(names(recipes), function(ratio) {
  workflow() %>%
    add_recipe(recipes[[ratio]]) %>%
    add_model(rf_model) %>%
    tune_grid(
      resamples = cv_folds,
      grid = rf_grid,
      metrics = model_metrics,
      control = control_grid(save_pred = TRUE)
    )
})
names(rf_results) <- names(recipes)

# XGBoost.
xgb_model <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

set.seed(123)
xgb_grid <- grid_latin_hypercube(
  trees(range = c(100L, 1000L)),
  tree_depth(range = c(2L, 8L)),
  learn_rate(range = c(-3, -1)),
  min_n(range = c(2L, 20L)),
  size = 20
)

set.seed(123)
xgb_results <- lapply(names(recipes), function(ratio) {
  workflow() %>%
    add_recipe(recipes[[ratio]]) %>%
    add_model(xgb_model) %>%
    tune_grid(
      resamples = cv_folds,
      grid = xgb_grid,
      metrics = model_metrics,
      control = control_grid(save_pred = TRUE)
    )
})
names(xgb_results) <- names(recipes)

# Compare the best PR-AUC for each SMOTE ratio.
log_smote_summary <- bind_rows(lapply(names(log_results), function(ratio) {
  collect_metrics(log_results[[ratio]]) %>%
    filter(.metric == "pr_auc") %>%
    mutate(model = "Logistic Regression", smote_ratio = ratio)
}))

rf_smote_summary <- bind_rows(lapply(names(rf_results), function(ratio) {
  show_best(rf_results[[ratio]], metric = "pr_auc", n = 1) %>%
    mutate(model = "Random Forest", smote_ratio = ratio)
}))

xgb_smote_summary <- bind_rows(lapply(names(xgb_results), function(ratio) {
  show_best(xgb_results[[ratio]], metric = "pr_auc", n = 1) %>%
    mutate(model = "XGBoost", smote_ratio = ratio)
}))

print(log_smote_summary)
print(rf_smote_summary)
print(xgb_smote_summary)

# Dissertation-selected final strategy: XGBoost + SMOTE 0.75.
best_xgb <- select_best(xgb_results[["0.75"]], metric = "pr_auc")

final_xgb_workflow <- workflow() %>%
  add_recipe(recipes[["0.75"]]) %>%
  add_model(xgb_model) %>%
  finalize_workflow(best_xgb)

dir.create("results", showWarnings = FALSE)
saveRDS(
  list(
    split = data_split,
    train = train_data,
    test = test_data,
    folds = cv_folds,
    metrics = model_metrics,
    xgb_grid = xgb_grid,
    log_results = log_results,
    rf_results = rf_results,
    xgb_results = xgb_results,
    best_xgb = best_xgb,
    final_xgb_workflow = final_xgb_workflow
  ),
  "results/model_development_objects.rds"
)
