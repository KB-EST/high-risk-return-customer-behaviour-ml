# ============================================================
# 05 — MODEL EVALUATION AND TECHNOLOGY CASE STUDY
# ============================================================

library(tidymodels)
library(themis)
library(xgboost)
library(dplyr)
library(purrr)

set.seed(123)

objects <- readRDS("results/model_development_objects.rds")
joined_clean <- readRDS("data/processed/joined_clean.rds")
model_data <- readRDS("data/processed/model_data.rds")

train_data <- objects$train
test_data <- objects$test
cv_folds <- objects$folds
model_metrics <- objects$metrics
xgb_grid <- objects$xgb_grid
final_xgb_workflow <- objects$final_xgb_workflow

# Re-estimate selected XGBoost performance using five-fold CV.
set.seed(123)
xgb_final_cv <- fit_resamples(
  final_xgb_workflow,
  resamples = cv_folds,
  metrics = model_metrics,
  control = control_resamples(save_pred = TRUE)
)

print(collect_metrics(xgb_final_cv))
xgb_cv_predictions <- collect_predictions(xgb_final_cv)

# Threshold analysis using out-of-fold predictions.
threshold_results <- map_dfr(
  seq(0.20, 0.50, by = 0.05),
  function(threshold) {
    preds <- xgb_cv_predictions %>%
      mutate(
        threshold_class = factor(
          ifelse(.pred_1 >= threshold, "1", "0"),
          levels = c("1", "0")
        )
      )

    tibble(
      threshold = threshold,
      sensitivity = sens_vec(preds$high_return_customer,
                             preds$threshold_class),
      precision = precision_vec(preds$high_return_customer,
                                preds$threshold_class),
      f1 = f_meas_vec(preds$high_return_customer,
                      preds$threshold_class)
    )
  }
)

print(threshold_results)
selected_threshold <- 0.45

# Fit the selected workflow on training data.
final_xgb_fit <- fit(final_xgb_workflow, data = train_data)

# Feature importance.
xgb_engine <- extract_fit_engine(final_xgb_fit)
xgb_importance <- xgboost::xgb.importance(model = xgb_engine)
print(xgb_importance)

dir.create("results", showWarnings = FALSE)
write.csv(xgb_importance,
          "results/xgboost_feature_importance.csv",
          row.names = FALSE)
write.csv(threshold_results,
          "results/threshold_analysis.csv",
          row.names = FALSE)

# Independent test evaluation.
# NOTE: the original pipeline labelled this as threshold 0.45 but used
# 0.30 in one condition. This cleaned version consistently uses 0.45.
test_probabilities <- predict(final_xgb_fit, test_data, type = "prob") %>%
  bind_cols(test_data %>% select(high_return_customer))

test_predictions <- test_probabilities %>%
  mutate(
    pred_class_045 = factor(
      ifelse(.pred_1 >= selected_threshold, "1", "0"),
      levels = c("1", "0")
    )
  )

print(conf_mat(
  test_predictions,
  truth = high_return_customer,
  estimate = pred_class_045
))

test_class_metrics <- metric_set(
  accuracy, sens, precision, f_meas
)(
  test_predictions,
  truth = high_return_customer,
  estimate = pred_class_045
)

test_probability_metrics <- metric_set(
  roc_auc, pr_auc
)(
  test_probabilities,
  truth = high_return_customer,
  .pred_1
)

print(test_class_metrics)
print(test_probability_metrics)

write.csv(
  bind_rows(test_class_metrics, test_probability_metrics),
  "results/final_test_metrics.csv",
  row.names = FALSE
)

# Sensitivity analysis: remove dominant total_orders predictor.
model_data_no_orders <- model_data %>% select(-total_orders)

set.seed(123)
split_no_orders <- initial_split(
  model_data_no_orders,
  prop = 0.80,
  strata = high_return_customer
)
train_no_orders <- training(split_no_orders)

set.seed(123)
cv_no_orders <- vfold_cv(
  train_no_orders,
  v = 5,
  strata = high_return_customer
)

recipe_no_orders <- recipe(
  high_return_customer ~ .,
  data = train_no_orders
) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(high_return_customer, over_ratio = 0.75)

xgb_tuned <- boost_tree(
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune(),
  min_n = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

set.seed(123)
xgb_no_orders <- workflow() %>%
  add_recipe(recipe_no_orders) %>%
  add_model(xgb_tuned) %>%
  tune_grid(
    resamples = cv_no_orders,
    grid = xgb_grid,
    metrics = model_metrics
  )

cat("\nBest PR-AUC without total_orders:\n")
print(show_best(xgb_no_orders, metric = "pr_auc", n = 5))

# Technology / electronic-device case study.
technology_data <- joined_clean %>%
  filter(Category == "Technology")

technology_features <- technology_data %>%
  group_by(`Customer ID`) %>%
  summarise(
    total_orders = n_distinct(`Order ID`),
    returned_orders = n_distinct(`Order ID`[returned == 1]),
    return_rate = returned_orders / total_orders,
    aov = mean(Sales_log, na.rm = TRUE),
    subcategory_conc = max(table(`Sub-Category`)) / n(),
    discount_seeking = mean(Discount, na.rm = TRUE),
    fast_shipping_share =
      mean(`Ship Mode` %in% c("Same Day", "First Class")),
    recency = as.numeric(
      max(technology_data$`Order Date`) - max(`Order Date`)
    ),
    total_spend = sum(Sales, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    high_return_customer = factor(
      ifelse(return_rate >= 0.20 | returned_orders >= 2, 1, 0),
      levels = c(1, 0)
    )
  )

technology_model_data <- technology_features %>%
  select(
    total_orders, aov, subcategory_conc, discount_seeking,
    fast_shipping_share, recency, total_spend,
    high_return_customer
  )

set.seed(123)
technology_split <- initial_split(
  technology_model_data,
  prop = 0.80,
  strata = high_return_customer
)
technology_train <- training(technology_split)
technology_test <- testing(technology_split)

set.seed(123)
technology_cv <- vfold_cv(
  technology_train,
  v = 5,
  strata = high_return_customer
)

technology_recipe <- recipe(
  high_return_customer ~ .,
  data = technology_train
) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(high_return_customer, over_ratio = 0.75)

technology_xgb <- workflow() %>%
  add_recipe(technology_recipe) %>%
  add_model(xgb_tuned)

set.seed(123)
technology_tune <- tune_grid(
  technology_xgb,
  resamples = technology_cv,
  grid = xgb_grid,
  metrics = model_metrics,
  control = control_grid(save_pred = TRUE)
)

best_technology_xgb <- select_best(
  technology_tune,
  metric = "pr_auc"
)

technology_final_workflow <- finalize_workflow(
  technology_xgb,
  best_technology_xgb
)

technology_fit <- fit(
  technology_final_workflow,
  data = technology_train
)

technology_probabilities <- predict(
  technology_fit,
  technology_test,
  type = "prob"
) %>%
  bind_cols(technology_test %>% select(high_return_customer))

technology_predictions <- technology_probabilities %>%
  mutate(
    pred_class_045 = factor(
      ifelse(.pred_1 >= selected_threshold, "1", "0"),
      levels = c("1", "0")
    )
  )

print(conf_mat(
  technology_predictions,
  truth = high_return_customer,
  estimate = pred_class_045
))

technology_class_metrics <- metric_set(
  accuracy, sens, precision, f_meas
)(
  technology_predictions,
  truth = high_return_customer,
  estimate = pred_class_045
)

technology_probability_metrics <- metric_set(
  roc_auc, pr_auc
)(
  technology_probabilities,
  truth = high_return_customer,
  .pred_1
)

print(technology_class_metrics)
print(technology_probability_metrics)

write.csv(
  bind_rows(technology_class_metrics, technology_probability_metrics),
  "results/technology_test_metrics.csv",
  row.names = FALSE
)
