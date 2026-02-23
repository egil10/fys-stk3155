# fys-stk3155/fys-stk4155: applied data analysis and machine learning

**course:** applied data analysis and machine learning (fys-stk3155 / fys-stk4155)  
**institution:** university of oslo  
**academic year:** 2024-2025

this repository contains implementations, analyses, and reports for the three main projects in the data analysis and machine learning course. the projects progressively build from fundamental linear regression methods to advanced neural network architectures, providing hands-on experience with both theoretical foundations and practical implementations.

## repository overview

this repository is organized into projects and lectures:

- **project1**: linear regression and gradient descent methods
- **project2**: feed-forward neural networks
- **project3**: advanced topics (in progress)
- **lectures**: weekly exercises and lecture materials

each project includes complete implementations, comprehensive analyses, jupyter notebooks, generated visualizations, and detailed reports.

## project structure

```
fys-stk3155/
├── project1/              # linear regression and optimization
│   ├── code/               # implementation and analysis code
│   ├── readme.md           # project 1 documentation
│   └── fys-stk3155 project 1 - the great regression.pdf
│
├── project2/              # neural networks
│   ├── code/               # neural network implementation
│   ├── readme.md           # project 2 documentation
│   └── fys-stk3155 project 2 - neural networking.pdf
│
├── project3/              # player value prediction
│   ├── code/               # data processing and analysis
│   ├── readme.md           # project 3 documentation
│   └── fys-stk3155 project 3 - predicting footballer market value using machine learning.pdf
│
├── lectures/              # course materials and exercises
│
├── requirements.txt       # global dependencies
└── readme.md               # this file
```

## project 1: linear regression and gradient descent

**focus:** fundamental regression methods, optimization algorithms, and model evaluation techniques.

### objectives

- implement and analyze ordinary least squares (ols), ridge, and lasso regression
- develop gradient descent optimization algorithms from scratch
- study bias-variance tradeoff using resampling methods
- compare closed-form solutions with iterative optimization methods

### key components

**core implementations:**
- `ols.py`: closed-form ols parameter estimation
- `ridge.py`: closed-form ridge regression with l2 regularization
- `gradient_descent.py`: standard gradient descent for ols, ridge, and lasso
- `stochastic_gradient_descent.py`: stochastic variants with advanced optimizers (momentum, adagrad, rmsprop, adam)
- `polynomial_features.py`: polynomial feature matrix generation
- `prepare_data.py`: data preparation and runge function definition

**analysis notebooks:**
- `exa.ipynb`: ols regression analysis on runge function
- `exb.ipynb`: ridge regression with hyperparameter tuning
- `exc.ipynb`: validation of gradient descent implementations
- `exd.ipynb`: advanced gradient descent methods comparison
- `exe.ipynb`: lasso regression implementation and validation
- `exf.ipynb`: stochastic gradient descent methods
- `exg.ipynb`: bias-variance analysis using bootstrap resampling
- `exh.ipynb`: bias-variance analysis using k-fold cross-validation

### main results

- comprehensive comparison of ols, ridge, and lasso regression methods
- performance analysis of various gradient descent algorithms
- systematic study of bias-variance tradeoff through resampling techniques
- hyperparameter optimization for regularization parameters
- validation against scikit-learn implementations

### dependencies

- numpy
- scikit-learn
- matplotlib
- pandas

for detailed information, see [project1/readme.md](project1/readme.md).

## project 2: feed-forward neural networks

**focus:** building neural networks from scratch, implementing backpropagation, and applying to regression and classification problems.

### objectives

- implement a complete feed-forward neural network framework from scratch
- develop the backpropagation algorithm for gradient computation
- test multiple activation functions (sigmoid, relu, leakyrelu)
- implement various optimization algorithms (sgd, rmsprop, adam)
- apply neural networks to regression and classification tasks
- compare neural network performance with traditional methods

### key components

**core implementations:**
- `neural_network.py`: main ffnn class with backpropagation algorithm
- `activations.py`: activation functions (sigmoid, relu, leakyrelu, softmax, linear) with derivatives
- `losses.py`: loss functions (mse, cross-entropy, bce) with gradient computations
- `optimizers.py`: optimization algorithms (sgd, momentum, adagrad, rmsprop, adam)
- `prepare_data.py`: data preparation utilities
- `plot_style.py`: visualization utilities

**analysis notebooks:**
- `exa.ipynb`: analytical derivation of cost functions and activation derivatives
- `exb.ipynb`: basic neural network implementation for regression on runge function
- `exb_with_noise.ipynb`: regression analysis with noise
- `exc.ipynb`: validation against scikit-learn and automatic differentiation
- `exd.ipynb`: activation function and network architecture analysis
- `exe.ipynb`: l1 and l2 regularization analysis
- `exf.ipynb`: mnist classification using neural networks

**testing:**
- comprehensive unit tests for all components
- gradient verification using automatic differentiation
- validation against standard ml libraries

### main results

- successful implementation of neural networks matching library performance
- comparison of neural networks with ols, ridge, and lasso regression
- analysis of activation function impact on training dynamics
- hyperparameter optimization for network architecture and learning rates
- classification accuracy on mnist dataset
- regularization effects and comparison with traditional methods

### dependencies

- numpy
- matplotlib
- scikit-learn

for detailed information, see [project2/readme.md](project2/readme.md).

## project 3: player value prediction

**focus:** time-series modeling and feature engineering for predicting football player market values using transfermarkt data.

### objectives

- process and aggregate football player data from transfermarkt
- engineer temporal features (cumulative statistics, lag features)
- implement rnn models for sequential player performance data
- compare neural network architectures (mlp, rnn) with traditional regression methods
- analyze the impact of player nationality and other static features on market value

### key components

**data processing:**
- `data_aggregating.ipynb`: basic feature extraction and dataset creation
- `data_agg_cumlag.ipynb`: cumulative and lag feature engineering
- `data_agg_nationality.ipynb`: nationality-enhanced feature set with one-hot encoding
- `convert_csv_to_parquet.py`: utility to compress large csv files to parquet format

**analysis notebooks:**
- `ridge_analysis.ipynb`: ridge regression baseline models
- `nn_analysis.ipynb`: neural network analysis on basic feature set
- `nn_analysis_nationality.ipynb`: neural network analysis with nationality features
- `rnn.ipynb`: rnn implementation for sequential player data

**data structure:**
- raw data: `code/data/` (players, valuations, game events)
- processed data: `code/data_processed/` (tabular datasets, rnn sequences, metadata)

### main results

- multiple feature engineering approaches (basic, cumulative, nationality-enhanced)
- comparison of ridge regression, mlp, and rnn models
- analysis of feature importance and model performance
- temporal feature engineering for player value prediction

### dependencies

- numpy, pandas
- pytorch (for rnn models)
- scikit-learn
- matplotlib, seaborn

for detailed information, see [project3/readme.md](project3/readme.md).

## installation and setup

### prerequisites

- python 3.8 or higher
- pip package manager
- jupyter notebook (for running analysis notebooks)

### quick start

1. clone the repository:
   ```bash
   git clone <repository-url>
   cd fys-stk3155
   ```

2. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. launch jupyter notebook:
   ```bash
   jupyter notebook
   ```

- Textbook references: See individual project README files

## Contributing

This repository contains coursework submissions. For questions or issues related to the course material, please refer to the course website or contact the course instructors.

## License

This repository contains academic coursework for FYS-STK3155/FYS4155 at the University of Oslo. The code and reports are intended for educational purposes as part of the course requirements.

## Acknowledgments

- Course instructors and teaching assistants at the University of Oslo
- Authors of referenced textbooks and resources (Nielsen, Goodfellow et al., Raschka et al.)
- Scikit-Learn, NumPy, and Matplotlib development teams

## Contact

For questions about this repository or the implementations, please refer to the individual project README files or the course documentation.

---

**Note:** This repository is maintained as part of coursework for FYS-STK3155/FYS4155. All implementations are original work developed for learning and educational purposes.
