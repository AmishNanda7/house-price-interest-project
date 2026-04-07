# House Price and Interest Rate Project

## Project Title
Predicting Los Angeles Home Values Using Mortgage Interest Rates

## Project Goal
This project predicts next month’s Los Angeles home value using Zillow home value data and U.S. mortgage interest rate data.

## Datasets Used
1. Zillow Metro ZHVI dataset   link- https://www.zillow.com/research/data/
2. FRED MORTGAGE30US dataset   link- https://fred.stlouisfed.org/data/MORTGAGE30US

## Methods Used
- Data cleaning
- Reshaping Zillow data
- Monthly aggregation
- Merging by date
- Linear Regression
- Random Forest comparison
- Streamlit app for interactive prediction

## Main Result
Linear Regression performed much better than Random Forest for this project.

## Files
- notebooks/house_price_interest_project.ipynb
- apps/streamlit_app.py
- data/processed/merged_house_price_interest.csv
- data/processed/final_model_data.csv
