# House-Of-Prices-Ensemble-Tuning-Model-Blending
THE GOAL
This notebook shares a study and explaines advanced regression techniques used to predict house prices.

Each feature of the dataset represents a different attribute of each house

The goal is to predict the values for SalePrice feature by applying advanced regression models

THE METHOD
Cross Validation: Using 5-fold cross-validation

6 different advanced regression models were used to predict house prices. Models were XGBoost, LightGBM, Gradient Boosting, KNeighborsRegressor, CatBoost

Root Mean Squared Error (RMSE) was used as the metric to evaluate the models success

CONCLUSION
In this notebook, we have:

Data Preparation

Removed the Id column and aligned train/test indices.
Handled missing values, encoded categorical features, and scaled numerical ones.
Modeling & Tuning

Trained and tuned five regressors (GBR, XGB, LGBM, KNN, CatBoost) using 5-fold CV.
Gradient Boosting Regressor achieved the best CV RMSE of 0.12196.
Ensembling

Computed inverse-RMSE weights for blending.
Produced a final stacked prediction with CV RMSE ≈ 0.12121.
Key Takeaways

Proper hyperparameter search can yield significant gains over defaults.
Ensembling complementary models (GBR + CatBoost + XGB + LGBM) further reduces error.
Next Steps

Incorporate advanced feature engineering (interactions, cyclical encodings).
Experiment with stacking or meta-learners.
Validate stability on a hold-out or via time-based splits if available.
Overall, this pipeline delivers robust performance on the “House Prices” dataset and provides a solid foundation for further improvements.
