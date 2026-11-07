# BigMart Sales Prediction

Predicts item-level sales across retail outlets using an **XGBoost Regressor**, with regularization and early stopping to control overfitting.

## Problem
Retailers need accurate sales forecasts to plan inventory, pricing, and promotions. This project predicts `Item_Outlet_Sales` from product and outlet attributes.

## Dataset
BigMart Sales data (`Train.csv`): 8,523 rows, 11 features (item weight, fat content, visibility, type, MRP; outlet identifier, size, location, type, establishment year) and the target `Item_Outlet_Sales`.

## Approach
1. Filled missing `Item_Weight` values with the column mean
2. Standardized `Item_Fat_Content` labels (`LF`, `low fat` → `Low Fat`; `reg` → `Regular`)
3. Label-encoded categorical features
4. Split data into train (5,454), validation (1,364), and test (1,705) sets
5. Trained an XGBoost Regressor with early stopping on the validation set
6. Evaluated with R²

## Reducing Overfitting
The default XGBoost model has no regularization, so it fits training data too closely. The final model uses:

- Shallow trees (`max_depth=3`) and `min_child_weight=10`
- Split penalty (`gamma=1`)
- Row and column sampling (`subsample=0.7`, `colsample_bytree=0.7`)
- L1/L2 regularization (`reg_alpha=1`, `reg_lambda=10`)
- Low learning rate (`0.02`) with early stopping (50 rounds)
- A separate validation set, so the test score stays unbiased

## Results

| Dataset | R² Score |
|---|---|
| Train | 0.627 |
| Validation | 0.584 |
| Test | 0.587 |

The train-test gap is about 0.04, so the model generalizes well and overfitting is under control. Validation and test scores are nearly identical, which shows the estimate is stable.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

## Run Locally
```bash
git clone https://github.com/<your-username>/bigmart-sales-prediction.git
cd bigmart-sales-prediction
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
jupyter notebook
```

## Future Work
Log-transform the target, impute zero `Item_Visibility`, try one-hot encoding, compare with LightGBM and CatBoost, and deploy as a Streamlit app.

## Author
Sanchit Sahu 
