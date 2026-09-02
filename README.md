# Using Machine Learning Techniques to Predict High-Risk Return Behaviour in Online Retail E-Commerce: A Case Study of Electronic Devices


##  Project Overview

The growth of e-commerce has increased the volume of product returns, creating financial and operational challenges for online retailers. Although many returns are legitimate, repeated or unusually high return behaviour may require further investigation. This project investigated whether supervised machine learning could predict high-risk return behaviour in online retail e-commerce, with a case-study focus on electronic-device purchases.

Transactional order and return data were integrated and transformed into customer-level behavioural features, including purchase frequency, average order value, discount-seeking behaviour, loss-making purchase rate, fast-shipping preference and recency. A binary target was constructed to represent high-risk return behaviour. During model development, target leakage was identified and removed, while class imbalance was addressed by evaluating SMOTE ratios of 0.30, 0.50, 0.75 and 1.00.

Three supervised machine learning models — Logistic Regression, Random Forest and XGBoost — were developed and evaluated using five-fold cross-validation. Performance was assessed using accuracy, sensitivity, precision, F1-score, ROC-AUC and PR-AUC, with particular attention given to minority-class performance.

XGBoost demonstrated the strongest overall predictive performance and was selected as the final model. On the independent test set, it achieved a ROC-AUC of **0.610** and PR-AUC of **0.248**. At a classification threshold of **0.45**, the model achieved **67.9% accuracy**, **36.4% sensitivity**, **17.8% precision** and an **F1-score of 0.239**. Feature-importance analysis identified purchase frequency (`total_orders`) as the strongest predictor, while sensitivity analysis showed a substantial reduction in predictive performance when this feature was removed.

Overall, the results showed that customer-level behavioural data contain useful but limited predictive information for identifying high-risk return behaviour. Machine learning may therefore support **risk screening and decision-making** rather than independently determining return abuse. The project also demonstrated the importance of addressing class imbalance, preventing target leakage and evaluating predictive models on unseen data before practical implementation.

---

##  Research Aim

To investigate how machine learning techniques can be used to predict high-risk return behaviour in online retail e-commerce, using electronic devices as a case study.

##  Research Objectives

1. Review existing research relating to high-risk return behaviour, online retail e-commerce and the application of machine learning techniques.
2. Develop an appropriate research strategy for investigating high-risk return behaviour using machine learning.
3. Implement a machine learning solution using electronic-device transaction data as a case study.
4. Critically evaluate the implementation using appropriate performance measures and analysis.
5. Evaluate the overall project findings and draw conclusions from the analysis.

---

#  CRISP-DM Methodology

The project followed the **Cross-Industry Standard Process for Data Mining (CRISP-DM)** to structure the analytical workflow from understanding the research objective through data preparation, modelling and evaluation.

The workflow followed:

**Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Practical Application**

---

## 1. Business Understanding

Product returns are a normal and necessary part of online retail. However, customers demonstrating repeated or unusually high return activity can create additional reverse-logistics costs, inventory-management challenges and financial losses.

The analytical challenge is that high return activity does not necessarily represent fraudulent behaviour. Customers may legitimately return products because of dissatisfaction, incorrect purchases, product quality, delivery problems or other reasons.

For this reason, the project was designed as a **high-risk return behaviour classification problem**, rather than a fraud-detection system.

The objective was to investigate whether patterns contained within customer transaction histories could be used to identify customers exhibiting elevated return behaviour and support further risk assessment.

---

## 2. Data Understanding

### Dataset

The project used the **Sample Superstore dataset**, a publicly available retail dataset containing transactional information from a US-based retail business.

The analysis combined two main tables:

- **Orders** — customer purchases, products, sales, discounts, shipping and profitability.
- **Returns** — records identifying orders that were returned.

The tables were integrated using `Order ID`.

Following integration and cleaning, the transaction-level dataset contained:

| Data | Count |
|---|---:|
| Transaction records | 9,994 |
| Unique orders | 5,009 |
| Unique customers | 793 |

The modelling problem focused on **customer behaviour rather than individual transactions**. The transaction-level dataset was therefore transformed into one behavioural profile for each customer.

### Exploratory Data Analysis

EDA was conducted before modelling to understand the structure and quality of the data and identify patterns that could influence feature engineering.

The analysis examined:

- distributions of Sales, Profit, Quantity and Discount;
- skewness and outliers;
- relationships between numerical variables;
- return behaviour across customer segments;
- return behaviour across shipping modes;
- discount patterns by return status;
- customer purchasing and spending behaviour.

Sales were strongly right-skewed, so a log transformation was investigated to reduce the influence of extreme transaction values.

Correlation analysis was also used later in the project to identify potentially redundant behavioural predictors.

---

## 3. Data Preparation

### Data Integration

The Orders and Returns tables were imported separately and joined using `Order ID`.

The workflow was:

`Orders + Returns → Data Integration → Cleaning → Transaction-Level Dataset → Customer Aggregation → Behavioural Feature Engineering → Target Construction`

### Customer-Level Feature Engineering

Individual transaction records were aggregated into customer-level behavioural profiles.

Candidate features represented different aspects of customer purchasing behaviour, including:

- purchase frequency;
- spending behaviour;
- average transaction value;
- product-category concentration;
- discount-seeking behaviour;
- loss-making purchases;
- shipping preferences;
- recency;
- sales variability;
- return behaviour.

After redundancy assessment and leakage control, seven predictors were retained for modelling.

| Feature | Behaviour Represented |
|---|---|
| `total_orders` | Purchase frequency |
| `aov` | Average log-transformed sales |
| `category_conc` | Product-category concentration |
| `discount_seeking` | Discount behaviour |
| `fast_shipping_share` | Preference for expedited shipping |
| `recency` | Recency of customer activity |
| `total_spend` | Cumulative customer expenditure |

### Dimensionality Assessment

Principal Component Analysis (PCA) was considered as a potential dimensionality-reduction technique.

However, the overall Kaiser-Meyer-Olkin (KMO) value was **0.49**, indicating limited shared variance between the behavioural variables.

PCA was therefore not applied.

Retaining the original behavioural variables also preserved interpretability, which was important because one of the objectives of the project was to determine which customer behaviours contributed to prediction.

### Target Construction

The dataset did not contain a confirmed label identifying return abuse.

A transparent proxy target was therefore created to represent **high-return behaviour**.

A customer was classified as high-return where:

**Return rate ≥ 20% OR the customer had at least two distinct returned orders.**

This produced:

| Class | Customers |
|---|---:|
| High-return | 109 |
| Non-high-return | 684 |
| **Total** | **793** |

This distribution revealed a substantial class imbalance.

### Target Leakage Detection

An important issue was discovered during initial model development.

The first modelling attempt produced unusually high performance, including approximately:

- Accuracy: **0.983**
- F1-score: **0.844**
- PR-AUC: **0.928**
- ROC-AUC: **0.980**
- Sensitivity: **0.887**

Rather than accepting these results, the predictor set was investigated.

The investigation showed that variables containing return information — including `return_rate` and returned-order information — had been included as predictors even though the same information was used to construct the target.

This meant the model could partially reconstruct the target definition instead of learning independent behavioural relationships.

This was identified as **target leakage**.

The initial results were therefore rejected, the leakage variables were removed, and all models were rebuilt using only predictors that were independent of the target definition.

This produced a more realistic modelling problem and prevented artificially inflated performance.

### Train/Test Split

The final customer-level dataset was divided using an **80/20 stratified train-test split**.

Stratification preserved the proportion of high-return and non-high-return customers across both partitions.

The test set remained completely separate during model development and was used only for final independent evaluation.

---

## 4. Modelling

### Five-Fold Stratified Cross-Validation

Model development and hyperparameter selection were conducted using **five-fold stratified cross-validation** on the training data.

Cross-validation was used to obtain more reliable estimates of model performance than relying on a single training split.

Stratification ensured that each fold maintained approximately the same high-return/non-high-return class distribution.

The independent test set was not used for:

- model selection;
- hyperparameter tuning;
- SMOTE-ratio selection;
- threshold development.

This reduced the risk of optimising the modelling decisions around the final test observations.

### Handling Class Imbalance with SMOTE

Only **109 of the 793 customers** belonged to the high-return class.

A model could therefore achieve relatively high accuracy by predicting most customers as non-high-return while performing poorly on the customers the project was actually trying to identify.

The **Synthetic Minority Over-Sampling Technique (SMOTE)** was used to increase minority-class representation during model training.

SMOTE generates synthetic minority observations using neighbouring minority examples rather than simply duplicating existing observations.

Importantly, SMOTE was applied **only within the training and cross-validation workflow**.

The independent test set retained its original class distribution.

This prevented synthetic observations from influencing the final evaluation and reduced the risk of data leakage.

### SMOTE Ratio Selection

Instead of automatically balancing both classes equally, four oversampling ratios were evaluated:

- **0.30**
- **0.50**
- **0.75**
- **1.00**

Each ratio was evaluated using the same five-fold stratified cross-validation framework.

Lower SMOTE ratios generally preserved higher overall accuracy but often produced extremely weak minority-class detection.

Increasing the oversampling ratio improved sensitivity but introduced additional false-positive predictions.

The **0.75 SMOTE workflow** provided a useful balance for the final XGBoost modelling strategy and was selected for further model development.

### Models

Three supervised classification algorithms were compared.

#### Logistic Regression

Logistic Regression was used as an interpretable baseline classifier.

It provided a reference point for determining whether more complex ensemble models produced meaningful improvements.

#### Random Forest

Random Forest was introduced as a tree-based ensemble method capable of modelling non-linear relationships and interactions between behavioural variables.

#### XGBoost

XGBoost was evaluated as a gradient-boosting approach capable of sequentially improving predictions by learning from errors made by previous trees.

The three models therefore provided a comparison between:

**Interpretable linear model → Bagged tree ensemble → Boosted tree ensemble**

### Model Selection Metric

Because the target was imbalanced, **accuracy was not used as the primary model-selection metric**.

Some model configurations achieved accuracy above 85% while identifying almost none of the high-return customers.

For example, XGBoost with SMOTE 0.30 achieved approximately **86.3% accuracy but zero sensitivity**.

The primary model-selection metric was therefore **PR-AUC (Precision-Recall Area Under the Curve)**.

PR-AUC was supported by:

- ROC-AUC;
- sensitivity/recall;
- precision;
- F1-score;
- accuracy;
- confusion-matrix analysis.

This placed greater emphasis on the model's ability to identify the minority high-return class.

---

## 5. Evaluation

### Model Comparison

The three algorithms demonstrated different strengths.

Logistic Regression increased sensitivity under stronger oversampling but generated a large number of false-positive predictions.

Random Forest improved ranking performance but generally retained relatively low sensitivity.

XGBoost provided the strongest overall combination of PR-AUC and minority-class performance under the selected modelling strategy.

For the selected **SMOTE 0.75 XGBoost workflow**, cross-validation produced approximately:

| Metric | Score |
|---|---:|
| Accuracy | 0.806 |
| Sensitivity | 0.358 |
| Precision | 0.325 |
| F1-score | 0.335 |
| ROC-AUC | 0.690 |
| PR-AUC | 0.374 |

### XGBoost Hyperparameter Tuning

XGBoost was tuned across multiple hyperparameter combinations using PR-AUC as the primary optimisation metric.

The selected configuration used:

| Hyperparameter | Selected Value |
|---|---:|
| Trees | 173 |
| Maximum tree depth | 2 |
| Learning rate | 0.00241 |
| Minimum observations per node (`min_n`) | 2 |

The selected tuning configuration achieved a mean five-fold cross-validated PR-AUC of approximately **0.370**.

### Classification Threshold Analysis

The standard classification threshold of **0.50** was not automatically assumed to be optimal.

Out-of-fold predictions from the selected XGBoost model were evaluated across thresholds from **0.20 to 0.50**.

| Threshold | Sensitivity | Precision | F1 |
|---:|---:|---:|---:|
| 0.20 | 1.000 | 0.137 | 0.241 |
| 0.25 | 1.000 | 0.137 | 0.241 |
| 0.30 | 1.000 | 0.137 | 0.241 |
| 0.35 | 0.920 | 0.203 | 0.333 |
| 0.40 | 0.805 | 0.201 | 0.322 |
| 0.45 | 0.51  | 0.249 |0.336  |
| 0.50 | 0.391 | 0.262 | 0.313 |

A threshold of **0.45** was selected because it produced the highest F1-score among the tested thresholds while improving sensitivity compared with the default 0.50 threshold.

The threshold adjustment increased the number of high-return customers detected while accepting a moderate reduction in precision.

### Independent Test Evaluation

After all modelling decisions had been made, the final XGBoost workflow was evaluated on the untouched test set.

| Metric | Independent Test |
|---|---:|
| Accuracy    | 0.679 |
| Sensitivity | 0.364 |
| Precision   | 0.178 |
| F1-score    | 0.239 |  
| ROC-AUC     | 0.610 |
| PR-AUC      | 0.248 |

At the 0.45 threshold, the test-set confusion matrix contained:

- **8 True Positives**
- **37 False Positives**
- **14 False Negatives**
- **100 True Negatives**

Performance was lower than the cross-validation results, demonstrating that the behavioural relationships learned during training did not generalise as strongly to unseen customers.

This result was important because it provided a more realistic assessment of the model than relying only on cross-validation performance.

### Feature Importance

Feature-importance analysis showed that:

**`total_orders` — customer purchase frequency — was the dominant predictor in the final XGBoost model.**

This suggested that customers with greater purchasing activity provided an important predictive signal for the constructed high-return target.

However, this does **not** mean frequent customers are return abusers.

Customers who make more purchases naturally have more opportunities to return products, so purchase frequency must be interpreted as a predictive relationship rather than evidence of fraudulent intent.

### Sensitivity Analysis

Because `total_orders` dominated feature importance, an additional sensitivity analysis was performed.

The variable was removed and XGBoost was re-tuned.

The best cross-validated PR-AUC fell from approximately **0.39 to 0.215**.

This confirmed that a substantial proportion of the model's predictive ranking ability depended on purchase frequency.

The experiment also demonstrated that the remaining behavioural features contained some predictive information, but considerably less than `total_orders`.

---

## 6. Electronic-Device Case Study

Because the dissertation specifically focused on electronic-device purchases, an additional evaluation was conducted using transactions from the **Technology** category.

The independent Technology evaluation contained **138 customers**.

| Metric | Wider Test Set | Technology Case Study |
|---|---:|---:|
| ROC-AUC | 0.610 | **0.654** |
| PR-AUC | 0.248 | **0.423** |
| Sensitivity | 0.364 | **0.591** |

Performance improved across all three measures within the Technology-specific analysis.

PR-AUC increased from **0.248 to 0.423**, while sensitivity increased from **36.4% to 59.1%**.

The results suggest that the available behavioural features contained more useful predictive information when the analysis was focused on the product category most relevant to the research.

However, the results still do not establish fraudulent intent and should be interpreted as evidence of elevated return behaviour rather than confirmed return abuse.

---

##  Key Findings

1. **XGBoost provided the strongest overall predictive performance** among the three algorithms evaluated.

2. **Class imbalance significantly affected model behaviour.** High accuracy did not necessarily mean that high-return customers were being identified successfully.

3. **SMOTE improved minority-class learning**, but the amount of oversampling involved a trade-off between detecting more high-return customers and generating additional false positives.

4. **Target leakage can create misleadingly strong machine-learning results.** The initial near-perfect model was rejected after identifying return-derived predictors that leaked information from the target.

5. **Purchase frequency (`total_orders`) was the dominant predictive feature.** Removing it substantially reduced model performance.

6. **Independent test performance was considerably lower than cross-validation performance**, reinforcing the importance of evaluating models on completely unseen data.

7. **Technology-specific evaluation produced stronger minority-class performance** than the wider customer analysis, particularly for PR-AUC and sensitivity.

8. High-risk return behaviour should be interpreted as a **risk signal rather than proof of fraudulent return abuse**.

---

##  Practical Application

The model should not be used to automatically label customers as fraudulent or block product returns.

Instead, the modelling framework could form part of a broader **risk-screening or decision-support process**.

For example, customers receiving higher predicted risk scores could be prioritised for additional review alongside other information such as:

- product-return reasons;
- product condition;
- account history;
- refund patterns;
- payment behaviour;
- delivery information;
- customer-service interactions.

A production system would also require substantially more representative retailer data, verified outcomes, continuous model monitoring and appropriate customer-protection controls before automated decisions were considered.

---

##  Limitations

The project has several important limitations.

**Proxy target:**  
The dataset does not contain independently verified return-abuse or fraud labels. The target therefore represents elevated return behaviour rather than confirmed fraudulent intent.

**Dataset scope:**  
The analysis uses a single publicly available retail dataset. The findings should not automatically be generalised to other retailers, countries or customer populations.

**Limited customer-level sample:**  
Although the transaction dataset contained 9,994 records, aggregation produced only 793 customer-level observations for modelling.

**Class imbalance:**  
Only 109 customers belonged to the high-return class, creating challenges for minority-class learning.

**Dependence on purchase frequency:**  
XGBoost relied heavily on `total_orders`, and sensitivity analysis demonstrated a substantial decline in performance when this feature was removed.

**Generalisation:**  
The decline from cross-validation to independent test performance shows that the model's predictive relationships were not equally strong for unseen customers.

**Behaviour does not establish intent:**  
The available variables describe purchasing behaviour but cannot explain why a customer returned a product.

---

##  Future Improvements

Future development could improve the modelling framework by using:

- larger customer-level datasets;
- verified return-abuse labels;
- detailed return-reason information;
- product-condition data;
- refund and payment behaviour;
- customer-service interactions;
- temporal behavioural features;
- additional product-specific information;
- external validation using data from another retailer.

Further modelling could also investigate alternative imbalance strategies, additional machine-learning algorithms, probability calibration and cost-sensitive classification.

---

##  Technologies & Techniques

**Language:** R

**Data Processing & Querying:** SQL, DuckDB

**Data Science:** Data Cleaning, Exploratory Data Analysis, Customer-Level Aggregation, Feature Engineering, Feature Selection

**Machine Learning:** Logistic Regression, Random Forest, XGBoost

**Imbalanced Learning:** SMOTE

**Model Validation:** Stratified Train/Test Split, Five-Fold Stratified Cross-Validation

**Evaluation:** PR-AUC, ROC-AUC, Accuracy, Sensitivity, Precision, F1-score, Confusion Matrix

**Model Interpretation:** Feature Importance, Threshold Analysis, Sensitivity Analysis

**Statistical Assessment:** Correlation Analysis, KMO, PCA Suitability Assessment

---

##  Repository Structure

```text
high-risk-return-customer-behaviour-ml/
│
├── README.md
├── LICENSE
│
├── data/
│   └── README.md
│
├── code/
│   ├── 01_data_preparation.R
│   ├── 02_exploratory_analysis.R
│   ├── 03_feature_engineering.R
│   ├── 04_modelling.R
│   └── 05_evaluation.R
│
├── figures/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   ├── smote_comparison.png
│   ├── model_comparison.png
│   ├── threshold_analysis.png
│   └── feature_importance.png
│
└── results/
    └── model_results.csv
```

---

##  Author

**Abdulkabir Bolaji Suleiman**
