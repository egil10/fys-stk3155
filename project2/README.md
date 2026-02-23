# project 2 — neural networks for classification and regression

**course:** data analysis and machine learning (fys-stk3155/fys4155)  
**institution:** university of oslo, norway  
**project window:** october 14, 2025 – november 10, 2025 (deadline at midnight)  
**group members:** bror johannes tidemand ruud (640394), egil furnes (693784), & �dne rolstad (693783).

---

## 1. contents

1. deliverables  
2. project brief  
3. methods to implement and analyse  
4. background literature  
5. reporting guidelines for numerical projects  
6. electronic delivery format

---

## 2. deliverables

please form a group (2–3 students recommended) on canvas before you start. when submitting project 2 on canvas, upload the following as a group:

- `project2_report.pdf`: approximately 5 000 words (≈10–12 pages) excluding references and appendices, with 10–15 figures in the main text. additional figures/tables may go in appendices or the repository.
- a canvas comment linking to the github repository (or subfolder) that mirrors the submission. the repository must include a printable version of the report (pdf) and the complete code base.
- `code/`: folder containing all python modules and jupyter notebooks needed to reproduce results. make runs deterministic by setting seeds for random initialisation and train/test splits.
- `readme.md`: this file. include group member names, installation instructions, notebook descriptions, and usage guidance.
- any supplemental material (figures, tables, logs) referenced in the report.

always cite external material (lecture notes, textbooks, libraries, web resources, ai-generated code or text). if you rely on tools such as chatgpt, include citations and, when possible, append the prompts and responses as supplemental material.

feel free to propose alternative data sets, provided you justify your choice and compare your findings to published results. the uci ml repository and kaggle are good starting points.

---

## 3. project brief

the goal is to implement, from scratch, a feed-forward neural network (ffnn) capable of solving both regression and multiclass classification problems. the project mirrors the week 41–42 lecture material and extends project 1 by reusing gradient descent components.

### 3.1 core tasks

| part | focus | key requirements |
|------|-------|------------------|
| a | analytical warm-up | derive mse, binary cross-entropy, and multiclass cross-entropy losses (with/without l1/l2) and the gradients of sigmoid, relu, leaky relu |
| b | ffnn regression | implement ffnn with flexible architecture, initialise weights/biases, use sigmoid hidden layers and appropriate output activation. compare to ols results from project 1 on the 1d runge function. explore architectures (1–2 hidden layers, 50/100 neurons). |
| c | library comparison | benchmark against `scikit-learn`, tensorflow/keras, or pytorch implementations. validate gradients with autograd or jax (optional but recommended). |
| d | activation & depth study | compare sigmoid, relu, leaky relu; vary layer depth/width; discuss overfitting/bias-variance effects. |
| e | regularisation | add l1/l2 penalties (� hyperparameters). compare with ridge and lasso from project 1 using the same data splits. |
| f | classification | adapt network for mnist classification using softmax cross-entropy. report accuracy, explore scaling, hyperparameters, architectures, and compare with logistic regression or library baselines. |
| g | critical evaluation | summarise strengths/weaknesses of each algorithm for regression vs classification, including optimisation strategies. |

### 3.2 suggested datasets

- regression: 1d runge function `f(x) = 1 / (1 + 25x�)` (baseline), optional 2d runge or more complex surfaces.
- classification: mnist (full dataset via `fetch_openml('mnist_784')`), optional fashion-mnist or other public datasets.

scaling inputs (e.g., dividing mnist pixel values by 255) is recommended. use consistent train/test splits (e.g., `train_test_split` with fixed `random_state`).

---

## 4. methods to implement and analyse

### 4.1 required implementation

1. reuse project 1 regression code (ols, ridge, lasso) as regression benchmarks.  
2. implement an ffnn with:
   - configurable layers, neurons per layer, and activation per layer (sigmoid, relu, leaky relu, linear, softmax).
   - switchable loss functions (mse for regression, cross-entropy for classification).
   - optional l1/l2 regularisation on weights/biases (for gradient computation).  
3. implement backpropagation to compute gradients.  
4. reuse and adapt gradient descent optimisers from project 1: plain gd, sgd, sgd+momentum, rmsprop, adam.  
5. integrate data scaling and train/test splitting utilities (preferably via `scikit-learn`).  
6. track performance metrics: mse for regression, accuracy (and optional confusion matrices) for classification.

### 4.2 required analysis

- discuss pros/cons of project 1 methods vs neural networks on the provided regression task.  
- examine hyperparameter effects (layers, neurons, activations, l1/l2) and highlight the most insightful results (heatmaps recommended).  
- evaluate neural network strengths/limitations for regression and classification.  
- compare optimisation strategies and learning rates.  

### 4.3 optional extensions (choose at least two)

- logistic regression baseline (equivalent to ffnn with no hidden layers).  
- automatic differentiation check (autograd/jax) for gradients.  
- comparison with pytorch or similar ml frameworks.  
- alternate datasets (fashion-mnist, 2d runge, etc.).  
- confusion matrix analysis of the best classifier.  
- any other well-justified extension agreed upon with course staff.

---

## 5. repository structure

```
project 2/
├── code/
│   ├── implementations/
│   │   ├── neural_network.py      # ffnn definition, backpropagation, training loops
│   │   ├── activations.py         # sigmoid, relu, leakyrelu, linear, softmax + derivatives
│   │   ├── losses.py              # mse, bce, cross-entropy + gradients, l1/l2 penalties
│   │   ├── optimizers.py          # plain gd, sgd, momentum, rmsprop, adam
│   │   ├── prepare_data.py        # scaling, train/test splitting, polynomial features
│   │   └── plot_style.py          # shared matplotlib styling utilities
│   ├── code_project1/             # benchmarks from project 1 (ols, ridge, lasso)
│   ├── notebooks/                 # results and analysis notebooks (see below)
│   ├── plots/                     # exported figures used in the report
│   ├── tables/                    # generated tables/metrics
│   ├── testing/                   # unit tests validating each module
│   └── requirements.txt           # python dependencies
├── project2.ipynb                 # assignment text
└── readme.md                      # this document
```

### notebook overview

- `notebooks/exa.ipynb`: analytical derivations for loss functions and activations.  
- `notebooks/exb.ipynb`: baseline ffnn regression on runge function.  
- `notebooks/exb_with_noise.ipynb`: regression robustness with injected noise.  
- `notebooks/exc.ipynb`: comparison with `scikit-learn`, tensorflow/keras, or pytorch baselines.  
- `notebooks/exd.ipynb`: activation and depth experiments, heatmaps, overfitting discussion.  
- `notebooks/exe.ipynb`: l1/l2 regularisation sweeps, comparison with ridge/lasso.  
- `notebooks/exf.ipynb`: mnist classification, accuracy metrics, optional confusion matrices.  

adapt notebook paths/imports if you restructure directories.

---

## 6. installation and execution

### 6.1 prerequisites

- python ≥ 3.8  
- `pip` (or `conda`)  
- (optional) gpu-enabled environment for faster mnist training

### 6.2 setup

```powershell
cd "project 2/code"
python -m venv .venv
.venv\scripts\activate.ps1          # powershell on windows
pip install --upgrade pip
pip install -r requirements.txt
```

install jupyter if needed:

```powershell
pip install jupyter
```

### 6.3 running notebooks

```powershell
cd "project 2/code"
jupyter notebook
```

ensure the notebooks can import modules from `implementations/`. most notebooks prepend the project root to `sys.path`; adjust if you change the folder layout.

### 6.4 running experiments via cli

example usage (replace with actual script entry points if provided):

```powershell
python implementations/prepare_data.py --dataset mnist --scale
python implementations/neural_network.py --config configs/run_regression.yaml
```

scripts should accept seeds via command-line flags or configuration files to preserve reproducibility.

### 6.5 running tests

```powershell
cd "project 2/code"
python -m pytest testing/
```

---

## 7. results management

- figures stored in `plots/` should be referenced in the report (10–15 in main text). provide captions, axis labels, and consistent styling.  
- tables in `tables/` summarise hyperparameter sweeps, benchmark comparisons, and evaluation metrics.  
- maintain a changelog or experimental log (e.g., `logs/` or appendix) documenting key runs, learning rates, seeds, and remarks on convergence behaviour.

---

## 8. background literature

- nielsen, m. **neural networks and deep learning**. http://neuralnetworksanddeeplearning.com/  
- goodfellow, i., bengio, y., courville, a. **deep learning** (ch. 6–8). https://www.deeplearningbook.org/  
- raschka, s. et al. **machine learning with pytorch and scikit-learn** (ch. 11–13). https://sebastianraschka.com/blog/2022/ml-pytorch-book.html  
- uio course compendium and lecture notes weeks 41–42: https://compphysics.github.io/machinelearning/  
- optimisation reference: goodfellow et al., *deep learning*, chapter 8.  
- optional comparative study: https://medium.com/ai-in-plain-english/comparison-between-logistic-regression-and-neural-networks-in-classifying-digits-dc5e85cd93c3

---

## 9. writing and referencing guidelines

- follow scientific/technical report conventions: abstract, introduction, methods, results, discussion, conclusion, references, appendices.  
- include pseudocode or algorithm summaries for major implementations.  
- provide well-commented source code (in notebooks or `.py` files).  
- validate against analytical solutions or known baselines where possible.  
- assess numerical stability and discuss uncertainty (e.g., variance across seeds).  
- reflect on lessons learned, potential improvements, and open questions.  
- keep a lab log of experiments, parameter choices, and runtime observations.

when citing software or datasets, follow the recommended citation style (e.g., `scikit-learn` guidelines, kaggle dataset citations). cite ai tools (chatgpt or others) if used, and retain interaction records as supplemental material when feasible.

---

## 10. electronic delivery format

- submit via canvas: https://www.uio.no/english/services/it/education/canvas/  
- upload only the report pdf or a link to the github repository if permitted.  
- source code lives in github/gitlab (or equivalent); include instructions for reproducing key results. avoid uploading third-party library files unless modified.  
- in the repository, maintain a folder of selected outputs (plots, tables, logs) referenced in the report.

collaboration is encouraged; ensure all contributions are acknowledged in the report and repository.

---

## 11. contact

for clarifications, use course discussion forums, lab sessions, or contact the teaching team. please notify us early if you plan major deviations (alternative datasets, architectures, tooling) so we can agree on evaluation criteria.

