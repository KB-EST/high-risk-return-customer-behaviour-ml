# Using Machine Learning Techniques to Predict High-Risk Return Behaviour in Online Retail E-Commerce: A Case Study of Electronic Devices

## 📌 Project Overview

The growth of e-commerce has increased the volume of product returns, creating financial and operational challenges for online retailers. Although many returns are legitimate, repeated or unusually high return behaviour may require further investigation. This project investigated whether supervised machine learning could predict high-risk return behaviour in online retail e-commerce, with a case-study focus on electronic-device purchases.

Transactional order and return data were integrated and transformed into customer-level behavioural features, including purchase frequency, average order value, discount-seeking behaviour, loss-making purchase rate, fast-shipping preference and recency. A binary target was constructed to represent high-risk return behaviour. During model development, target leakage was identified and removed, while class imbalance was addressed by evaluating SMOTE ratios of 0.30, 0.50, 0.75 and 1.00.

Three supervised machine learning models — Logistic Regression, Random Forest and XGBoost — were developed and evaluated using five-fold cross-validation. Performance was assessed using accuracy, sensitivity, precision, F1-score, ROC-AUC and PR-AUC, with particular attention given to minority-class performance.

XGBoost demonstrated the strongest overall predictive performance and was selected as the final model. On the independent test set, it achieved a ROC-AUC of **0.610** and PR-AUC of **0.248**. At a classification threshold of **0.45**, the model achieved **67.9% accuracy**, **36.4% sensitivity**, **17.8% precision** and an **F1-score of 0.239**. Feature-importance analysis identified purchase frequency (`total_orders`) as the strongest predictor, while sensitivity analysis showed a substantial reduction in predictive performance when this feature was removed.

Overall, the results showed that customer-level behavioural data contain useful but limited predictive information for identifying high-risk return behaviour. Machine learning may therefore support **risk screening and decision-making** rather than independently determining return abuse. The project also demonstrated the importance of addressing class imbalance, preventing target leakage and evaluating predictive models on unseen data before practical implementation.

---

## 🎯 Research Aim

To investigate how machine learning techniques can be used to predict high-risk return behaviour in online retail e-commerce, using electronic devices as a case study.

## ✅ Research Objectives

1. Review existing research relating to high-risk return behaviour, online retail e-commerce and the application of machine learning techniques.
2. Develop an appropriate research strategy for investigating high-risk return behaviour using machine learning.
3. Implement a machine learning solution using electronic-device transaction data as a case study.
4. Critically evaluate the implementation using appropriate performance measures and analysis.
5. Evaluate the overall project findings and draw conclusions from the analysis.

---

## 🔄 Project Methodology — CRISP-DM

The project followed the **Cross-Industry Standard Process for Data Mining (CRISP-DM)**, providing a structured framework for moving from understanding the return-behaviour problem through data preparation, machine learning and final model evaluation.

The project was organised around the following stages:

**Business Understanding → Data Understanding → Data Preparation → Modelling → Evaluation → Practical Application**
