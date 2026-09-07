# AI and Ethics: Fairness & Interpretability in Machine Learning

**Course**: AI and Ethics (Spring 2026) | **Indian Institute of Technology, Kharagpur** 

This repository contains the complete codebase, datasets, and reports for the AI and Ethics course assignments. The project covers two core themes in ethical AI:
1. **Fair Classifiers (Assignment 1)**: Implementing fairness constraints (decision boundary covariance, fairness penalty, and projections) from scratch to mitigate bias against sensitive groups on synthetic and real-world (Adult) datasets.
2. **Explainable AI (Assignment 2)**: Implementing LIME (Local Interpretable Model-agnostic Explanations) and SP-LIME (Submodular Optimization for global explanations) from scratch to analyze the fairness-accuracy trade-offs of machine learning models on the Bank Marketing dataset.

---

## Repository Structure

The project files are organized as follows:

```text
AIETH-Assignment/
├── Assignment_2_25CS60R36/                      # Submission package for Assignment 2
│   ├── Assignment1_25CS60R36_classifier_taskB.py
│   ├── Assignment1_25CS60R36_classifier_taskC.py
│   ├── Assignment2_25CS60R36_taskXY.py
│   ├── Assignment2_25CS60R36_taskZ_lime.py
│   ├── Assignment2_25CS60R36_taskZ_submodular.py
│   ├── 25CS60R36_AIETH_Assignment_2_report.pdf
│   └── README.md                               # Assignment 2 walkthrough

```

---

## Requirements & Setup

### Dependencies
All algorithms and constraints are implemented from scratch using basic linear algebra and numerical packages. No high-level machine learning frameworks (such as scikit-learn or lime) are used in any of the core algorithms.

- **Python Version**: 3.10 or higher
- **Required Packages**:
  - `numpy`: Numerical processing, vectorization, and matrix operations.
  - Standard libraries: `os`, `csv`, `pickle`, `random`.

### Dataset Setup
1. **Adult Dataset**: Placed under `Assignment 1/adult/` (already present).
2. **Bank Marketing Dataset**: Placed under `bank+marketing/` (already present).

---

## Assignment 1: Fair Machine Learning Classifiers

### Core Methodology
This assignment focuses on training binary classifiers while ensuring fairness with respect to a sensitive attribute $z \in \{0, 1\}$. Fairness is evaluated using the **p%-rule**, defined as:
$$\text{p\%-rule} = \min \left( \frac{P(\hat{y}=1 | z=1)}{P(\hat{y}=1 | z=0)}, \frac{P(\hat{y}=1 | z=0)}{P(\hat{y}=1 | z=1)} \right) \times 100$$
A p%-rule of $100\%$ indicates perfect statistical parity, while a rule below $80\%$ suggests potential disparity.

#### Task A: Data Generator (`Assignment_1_25CS60R36_Data_generator.py`)
Generates synthetic 2D datasets using two multivariate Gaussians (representing classes $y=1$ and $y=-1$). The class-conditional distributions are defined as:
- For $y=1$: Mean $\mu_+ = [1, -3]^T$, Covariance $\Sigma_+ = \begin{bmatrix} 3 & 1 \\ 1 & 2 \end{bmatrix}$
- For $y=-1$: Mean $\mu_- = [-3, 2]^T$, Covariance $\Sigma_- = \begin{bmatrix} 7 & 1 \\ 1 & 10 \end{bmatrix}$

To induce a controllable correlation between features and the sensitive attribute $z$, the coordinates are rotated by a parameter angle $\phi \in \{\pi, \pi/2, \pi/4, \pi/6, \pi/8\}$ using a rotation matrix:
$$R_\phi = \begin{bmatrix} \cos\phi & -\sin\phi \\ \sin\phi & \cos\phi \end{bmatrix}$$
The sensitive attribute $z_i \in \{0, 1\}$ for each point $x_i$ is generated using a Bernoulli distribution where the probability $P(z_i=1 | x_i)$ is derived from the true class conditional Gaussian densities:
$$P(z_i = 1 | x_i) = \frac{p(x_{rot,i} | y=1)}{p(x_{rot,i} | y=1) + p(x_{rot,i} | y=-1)}$$
This generates synthetic datasets where the bias of the sensitive attribute varies deterministically according to the rotation angle $\phi$.

#### Task B: Decision Boundary Covariance Constraint (`Assignment_1_25CS60R36_classifier_taskB.py`)
This script implements standard unconstrained Logistic Regression and a version constrained by decision boundary covariance. The decision boundary covariance measures the correlation between the sensitive attribute $z$ and the signed distance of feature vectors to the decision boundary:
$$\operatorname{Cov}(z, d(x, \theta)) \approx \frac{1}{N} \sum_{i=1}^N (z_i - \bar{z}) x_i^T \theta = \theta^T v$$
where $v = \frac{1}{N} \sum_{i=1}^N (z_i - \bar{z}) x_i$ is the covariance vector. 

The training script enforces the constraint $|\theta^T v| \leq c$ using a projection-based gradient update. During each epoch of gradient descent, if the parameter vector violates the boundary covariance threshold:
1. The script first performs a standard gradient step: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}(\theta)$.
2. If $|\theta^T v| > c$, the parameter vector is projected back onto the constraint boundary:
   $$\theta \leftarrow \theta - \frac{\theta^T v - \operatorname{sign}(\theta^T v)c}{\|v\|_2^2} v$$

#### Task C: Fairness Penalty Optimization (`Assignment_1_25CS60R36_classifier_Task_C.py`)
Rather than projecting parameters back to the constraint boundary, Task C formulates the covariance constraint as a regularized penalty term in the loss function:
$$\min_{\theta} \mathcal{L}(\theta) + \gamma \left| \theta^T v \right|$$
The gradient of this penalized objective function is computed as:
$$\nabla_\theta \text{Obj} = \nabla_\theta \mathcal{L}(\theta) + \gamma \operatorname{sign}(\theta^T v) v$$
The parameter vector $\theta$ is updated via subgradient descent. This implementation is particularly effective when trying to continuous control the fairness-accuracy trade-off by adjusting the penalty hyperparameter $\gamma$.

#### Task D: Projection-based Fairness with Noisy Labels (`Assignment_1_25CS60R36_classifier_Task_D.py`)
This task evaluates model robustness when target labels $y$ are subject to noise. The true log-odds are perturbed by adding zero-mean Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$ before thresholding to produce noisy labels:
$$y_{\text{noisy}} = \operatorname{sign}\left(\log\left(\frac{p(x | y=1)}{p(x | y=-1)}\right) + \epsilon\right)$$
To ensure strict fairness (equivalent to $c = 0$), the script trains a model by projecting the loss gradient orthogonally to the covariance vector $v$ at every iteration:
$$\nabla_{\text{proj}} = \nabla \mathcal{L}(\theta) - \frac{\langle \nabla \mathcal{L}(\theta), v \rangle}{\|v\|_2^2} v$$
$$\theta \leftarrow \theta - \eta \nabla_{\text{proj}}$$
This projects out the gradient component that moves the decision boundary in a direction correlated with the sensitive attribute, ensuring the boundary covariance remains zero throughout training.

### How to Run Assignment 1

First, generate the synthetic datasets:
```bash
python3 "Assignment 1/Assignment_1_25CS60R36_Data_generator.py"
```

Then, run the respective classifier scripts:
- **Task B (Covariance Constraint)**:
  ```bash
  python3 "Assignment 1/Assignment_1_25CS60R36_classifier_taskB.py"
  ```
- **Task C (Fairness Penalty)**:
  ```bash
  python3 "Assignment 1/Assignment_1_25CS60R36_classifier_Task_C.py"
  ```
- **Task D (Noisy Labels & Projection)**:
  ```bash
  python3 "Assignment 1/Assignment_1_25CS60R36_classifier_Task_D.py"
  ```

---

## Assignment 2: Interpreting Fairness-Accuracy Trade-offs using LIME

### Core Methodology
This assignment implements LIME and SP-LIME from scratch to explain predictions locally and globally on the **Bank Marketing** dataset. We compare two Logistic Regression models:
1. **Accurate Model**: Standard unconstrained Logistic Regression.
2. **Fair Model**: Logistic Regression with a covariance fairness penalty ($\gamma = 5.0$) built from Assignment 1.

#### Task X: Model Preprocessing & Training (`Assignment2_25CS60R36_taskXY.py`)
The script processes raw data from `bank-full.csv`:
- **Categorical Columns**: Job, marital status, education, default status, housing, personal loan, contact type, month, and outcome are one-hot encoded.
- **Numerical Columns**: Age, balance, day, duration, campaign, pdays, and previous occurrences are normalized using Z-score standardization:
  $$x_{\text{std}} = \frac{x - \mu}{\sigma}$$
- **Sensitive Attribute**: $z = 1$ if age $\geq 39$, else $0$.
- **Target Variable**: $y = 1$ if the customer subscribed ("yes"), else $0$.

The preprocessing yields a 49-dimensional continuous feature vector. The script trains the unconstrained and covariance-penalized models, selects 5 instances of interest (specifically where the models disagree on predicted outcomes), and prepares their local structures.

#### Task Y: Neighborhood Generation (`Assignment2_25CS60R36_taskXY.py`)
LIME explains a model locally by perturbing the input vector in an interpretable representation space. We select a set of 6 binary interpretable features:
- `age` ($\geq 39$)
- `balance` ($\geq 448$)
- `housing` (has housing loan)
- `education` (primary or secondary)
- `marital` (married)
- `loan` (has personal loan)

For a selected instance, the script generates $N=100$ perturbed samples by randomly flipping one binary attribute in the interpretable space $x' \in \{0, 1\}^6$. To pass these samples back into the black-box classifier, a reconstruction function maps $z' \in \{0,1\}^6$ back into the continuous and categorical feature representation. Continuous features are reconstructed using fixed category means (e.g., if age binary indicator is $0$, age is set to $30$; if $1$, it is set to $45$), while other categorical values are mapped to their categorical equivalents.

#### Task Z: Local LIME Models (`Assignment2_25CS60R36_taskZ_lime.py`)
For each selected instance, the script computes the Hamming distance between the original binary interpretable vector $x'$ and each neighborhood sample $z'$:
$$D_{\text{Hamming}}(x', z') = \sum_{j=1}^K \mathbb{I}(x'_j \neq z'_j)$$
These distances are converted into local similarity weights using a Gaussian kernel:
$$\pi_x(z') = \exp\left( - \frac{D_{\text{Hamming}}(x', z')^2}{\sigma^2} \right)$$
where the kernel width is set to $\sigma = 0.75 \sqrt{K}$ ($K = 6$). 

The script then fits a local weighted linear model to explain the predicted probabilities $f(z')$:
$$\theta_{\text{local}} = (Z'^T W Z')^{-1} Z'^T W f(Z')$$
where $Z'$ is the matrix of interpretable features (with a column of ones added for the intercept), and $W = \operatorname{diag}(\pi_x(z'))$. The weighted linear model coefficients are computed using the Moore-Penrose pseudoinverse (`np.linalg.pinv`) to guarantee numerical stability.

#### Task Z.2: Global Importance via SP-LIME (`Assignment2_25CS60R36_taskZ_submodular.py`)
To aggregate local explanations into a global view, the script implements a submodular-style selection proxy over a representative subset of $M=100$ samples. For each sample, it runs neighborhood generation and fits a local LIME model. The global importance score $I_j$ for interpretable feature $j$ is computed as:
$$I_j = \sqrt{\sum_{i=1}^M |w_{ij}|}$$
where $w_{ij}$ is the local LIME coefficient for feature $j$ in instance $i$. The top 3 features with the highest global importance scores are selected as the most influential variables driving predictions across the dataset.

---

### Assignment 2 Empirical Results

#### Model Performance Comparison
| Model | Train Accuracy | Test Accuracy | p%-rule (Fairness) |
| :--- | :---: | :---: | :---: |
| **Accurate (Unconstrained)** | 89.94% | 90.20% | 98.70% |
| **Fair ($\gamma = 5.0$)** | 89.97% | 90.29% | 85.26% |

*Note: The unconstrained model is naturally fair on this dataset for the age attribute, showing that the fairness constraint interacted dynamically with the naturally fair state.*

#### Local Explanations Example (LIME Coefficients for Instance 68)
| Interpretable Feature | Acc-Model Weight | Fair-Model Weight |
| :--- | :---: | :---: |
| **age** | -0.0076 | -0.0152 |
| **balance** | -0.0026 | -0.0026 |
| **housing** | -0.1265 | -0.1287 |
| **education** | -0.0194 | -0.0194 |
| **marital** | -0.0384 | -0.0416 |
| **loan** | -0.0811 | -0.0827 |

#### Top 3 Globally Important Features (SP-LIME)
1. **housing**: Strongest negative predictor of subscribing to a term deposit.
2. **loan**: Significant negative impact on subscription probability.
3. **marital**: Higher predictive importance than age or balance.

---

### How to Run Assignment 2

You can run the entire pipeline step-by-step using the root scripts:

1. **Preprocess and Train Models**:
   ```bash
   python3 Assignment2_25CS60R36_taskXY.py
   ```
   *This trains the models, selects the test instances, generates the neighborhoods, and saves intermediate data to `task_xy_data.pkl`.*

2. **Run LIME Local Explanations**:
   ```bash
   python3 Assignment2_25CS60R36_taskZ_lime.py
   ```
   *This fits local linear models and prints local explanations for the selected test instances, saving output to `task_z_lime_data.pkl`.*

3. **Run SP-LIME Global Feature Importance**:
   ```bash
   python3 Assignment2_25CS60R36_taskZ_submodular.py
   ```
   *This runs global submodular optimization over the subset to output the top 3 features.*

---

## Contact & Reports
For theoretical derivations, experimental plots, and complete analysis:
- Refer to the detailed reports:
  - **Assignment 1 PDF**: [25CS60R36_AIETH_Assignment_report.pdf](file:///Users/jarvis/IIT%20KGP%20Sem%202/AI%20and%20Ethics/AIETH%20Assignment/Assignment%201/25CS60R36_AIETH_Assignment_report.pdf)
  - **Assignment 2 PDF**: [25CS60R36_AIETH_Assignment_2_report.pdf](file:///Users/jarvis/IIT%20KGP%20Sem%202/AI%20and%20Ethics/AIETH%20Assignment/Assignment_2_25CS60R36/25CS60R36_AIETH_Assignment_2_report.pdf)
  - **Assignment 2 Markdown**: [Assignment2_25CS60R36_report.md](file:///Users/jarvis/IIT%20KGP%20Sem%202/AI%20and%20Ethics/AIETH%20Assignment/Assignment2_25CS60R36_report.md)
# EquiLIME
