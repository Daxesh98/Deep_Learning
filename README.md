Deep_Learning
==============================
Title: Employee Attrition Prediction and Model Evaluation
Objective:
The primary goal of this project is to build and evaluate machine learning models to predict employee attrition based on various features. The project uses deep learning models and other techniques to analyze and predict employee turnover, providing insights that can help organizations improve retention strategies.

Components:
Data Loading:

The project starts by loading employee attrition data from a CSV file. The data is used to understand employee behavior and factors leading to attrition.
Data Preprocessing:

Preprocessing involves preparing the data for modeling, including feature scaling and encoding. This is done using a function that creates dummy variables and scales the features to ensure they are suitable for training machine learning models.
Model Training and Evaluation:

Deep Learning Model: A simple deep learning model is created using TensorFlow/Keras. This model is trained on the preprocessed data to predict employee attrition. It includes creating, compiling, and fitting the model, followed by evaluation.
Model with Learning Rate Scheduling: An additional model is trained with a learning rate scheduler to optimize the learning process. This model includes a callback that adjusts the learning rate during training to improve convergence.
Activation Function Model: This model uses different activation functions to explore their impact on model performance. The model is trained and evaluated using these activation functions, and the results are compared.
Visualization:

Learning Rate vs. Loss: A plot is generated to visualize how different learning rates affect the loss during training. This helps in understanding the impact of learning rate on model performance.
Loss Curves: The loss curves for the training process are plotted to visualize how the loss changes over epochs. This helps in diagnosing the training process and identifying any potential issues such as overfitting or underfitting.
Model Evaluation:

The performance of the trained models is evaluated using metrics such as accuracy, confusion matrix, and classification report. This provides insights into how well the models are performing and their effectiveness in predicting employee attrition.
A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
