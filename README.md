# Real Estate Price Predictor

This project is a Machine Learning application designed to predict housing prices in Boston. It utilizes a Regression model (Random Forest) trained on historical housing data to provide accurate price estimations based on various features like location, room count, and neighborhood demographics.

## Features
- **Data Preprocessing:** Cleaning and handling of the Boston Housing dataset.
- **Model Training:** Implementation of a Random Forest Regression model.
- **Evaluation:** Analysis of model performance using metrics like RMSE and R-squared.
- **Modular Structure:** Organized code with dedicated folders for data, notebooks, and source code.

## Project Structure
- `data/`: Contains the raw and processed datasets.
- `models/`: Saved versions of the trained machine learning models.
- `notebooks/`: Jupyter Notebooks used for Exploratory Data Analysis (EDA) and experimentation.
- `src/`: Python source scripts for training (`train.py`) and utility functions.
- `requirements.txt`: List of Python dependencies required to run the project.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Saksham52/Real_Estate_Price_predictor.git](https://github.com/Saksham52/Real_Estate_Price_predictor.git)
   cd Real_Estate_Price_predictor

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Usage:
   - To train the model run the training script located in the src directory
   ```bash
   python src/train.py

## Technologies Used:
1. Language: Python
2. Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
3. Algorithms: Random Forest Regressor

## Author:
Saksham Adhau
   
