# Logistic Regression with L2 Regularization

In this project, we extend logistic regression with L2 regularization to allow different levels of regularization for different regression coefficients. This extension aims to refine the regularization structure, particularly considering the constant term, continuous-valued features, and binary-valued features.

## Background

We begin by revisiting logistic regression with L2 regularization, which minimizes a specific error function. However, we now introduce a more sophisticated regularization approach, allowing for individual regularization coefficients for each regression coefficient.

### Extension of L2 Regularization

The extension involves introducing three regularization coefficients:
- $\lambda_0$ for the constant term
- $\lambda_1$ for continuous-valued features
- $\lambda_2$ for binary-valued features

This refined regularization structure aims to optimize model performance by adapting regularization to different feature types.

## Implementation

We implement the extension through a Python class named `mylogistic_l2`, facilitating model training and prediction. This class employs the Newton-Raphson optimization method for training, necessitating the derivation of the gradient and Hessian of the modified error function.

### Usage Example

```python
logic1 = mylogistic_l2(reg_vec=lambda_vec, max_iter=1000, tol=1e-5, add_intercept=True)
logic1.fit(X_train, Y_train)
ypred = logic1.predict(X_test)
```

## Dataset

We utilize the "Adult" dataset from the UCI machine learning repository. The goal is to predict income levels based on various features. The dataset undergoes cleaning to prepare it for model testing, including label value encoding and handling missing values.

### Questions to Answer

1. **Data Preparation (Q1.1)**: Clean the dataset and create training and test arrays.
2. **Gradient and Hessian (Q1.2)**: Derive the gradient and Hessian matrix for the modified error function.
3. **Model Construction (Q1.3)**: Create the `mylogistic_l2` class and evaluate test accuracy for different regularization scenarios.
4. **Hyperparameter Tuning (Q1.4)**: Perform grid search to determine the best regularization coefficients for continuous-valued and binary-valued features.
5. **Comparison with Sklearn (Q1.5)**: Train and test the model using `sklearn.linear_model.LogisticRegression` and compare results.
