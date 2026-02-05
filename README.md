# Oversampling and Model Evaluation Project

This repository contains the code, datasets, and results generated during the development of this work, focused on oversampling techniques and the evaluation of classification models. 

## Project contents

The submitted package has the following structure:

├── Datasets/        # Contains the 25 datasets used, in CSV format
├── src/             # Project source code
│   ├── custom_smote.py   # Implementation of the proposed SMOTE variant
│   ├── experiments.py    # Functions to run experiments, hyperparameter optimization, and model evaluation
│   ├── utils.py          # Auxiliary functions used in the project
│   └── pipeline.py       # Main pipeline that integrates data loading, preprocessing, oversampling, training, and evaluation
└── requirements.txt      # Required libraries to run the project

To execute the project: python -m src.pipeline 
