# project 1 fys-stk3155

## group members:

egil furnes, bror johannes tidemand ruud, ã…dne rolstad

## project description
this project explores the challenge of linear regression on runges function, using ols, ridge and lasso regression. in addition, we implement methods for gradient descent, and use resampling methods such as bootstrap and cross-validation to explore the bias-variance tradeoff.

## required packages
to install the required packages, run the following command in terminal: !pip install -r ../requirements.txt


## descripton of notebooks and python files
### python files: 
- **gradient_descent.py**: implementation of the following gradient descent methods for ridge, ols and lasso regression: standard gd, gd with momentum, adagrad, rmsprop, adam.
- **ols.py**: contains function that returns closed-form ols parameters.
- **plot_style.py**: sets code style for entire project, ensuring that every plot has the same styling.
- **polynomial_features.py**: contains a function that returns a polynomial feature matrix, used in linear regression. can return feature matrices either with or without intercept column.
- **prepare_data.py**: contains a function for preparing the data used in the project. performs train-test split, defines the runge function, etc.
- **ridge.py**: contains function that returns the closed-form ridge parameters.
- **runge.py**: used for plotting the runge function during the project.
- **set_latex_params.py**: configuration for matplotlib plots in latex-style format.
- **stochastic_gradient_descent.py**: implementation of stochastic gradient descent functions for ridge, ols and lasso regression. standard gd, gd with momentum, adagrad, rmsprop, adam.

### notebooks:
- **exa.ipynb**: solves exercise a of the given problem set. ols regression analysis, plotting mse scores, r2-scores, parameters and fits for different polynomial degrees.
- **exb.ipynb**: solves exercise b of the given problem set. ridge regression analysis, plotting mse and r2 scores, parameters and fits for various lambda values. calulate parameters for different polynomial degrees and lambdas to find optimal parameters
- **exc.ipynb**: solves exercise c of the given problem set. testing and validation of standard gradient descent implementations for ols and ridge. computing parameters, mses, and r2-scores for different parameters and degrees, and comparing with closed-form solutions for ridge and ols. plotting results
- **exd.ipynb**: solves exercise d of the given problem set. testing and validation of gradient descent implementations for ols and ridge. implementations include gd with momentum, adagrad, rmsprop and adam. computing parameters, mses, and r2-scores for different parameters and degrees, and comparing with closed-form solutions for ridge and ols. plotting results, comparing different gd methods.
- **exe.ipynb**: solves exercise e of the given problem set. testing and validation of gradient descent method for lasso regression. comparing results with scikit-learn. plotting results.
- **exf.ipynb**: solves exercise f of the given problem set. testing stochastic gradient descent methods for ridge, ols and lasso regression. comparing results with closed-form solutions (ols and ridge) and scikit-learn (lasso). plotting results.
- **exg.ipynb**: solves exercise g of the given problem set. study bias-variance tradeoff using bootstrap resampling.
- **exh.ipynb**: solves exercise h of the given problem set. study bias-variance trade-off using k-fold cross validation.
