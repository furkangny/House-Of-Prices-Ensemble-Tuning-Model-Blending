# House-Of-Prices-Ensemble-Tuning-Model-Blending

## The Goal
This notebook demonstrates advanced regression techniques used to predict house prices. Each feature in the dataset represents a different attribute of a house. The objective is to predict the `SalePrice` using various regression models.

## The Method
- **Cross-Validation**: 5-fold cross-validation.  
- **Models**:
  - XGBoost
  - LightGBM
  - Gradient Boosting Regressor (GBR)
  - K-Nearest Neighbors Regressor (KNN)
  - CatBoost  
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

## Data Preparation
1. Removed the `Id` column and aligned train/test indices.  
2. Handled missing values.  
3. Encoded categorical features.  
4. Scaled numerical features.  

## Modeling & Tuning
- Trained and tuned five regressors (GBR, XGB, LGBM, KNN, CatBoost) using 5-fold CV.  
- Gradient Boosting Regressor achieved the best CV RMSE of 0.12196.

## Ensembling
- Computed inverse-RMSE weights for blending.  
- Produced a final stacked prediction with CV RMSE ≈ 0.12121.

## Key Takeaways
- Proper hyperparameter tuning yields significant gains over default settings.  
- Ensembling complementary models (GBR, CatBoost, XGB, LGBM) further reduces error.

## Next Steps
- Incorporate advanced feature engineering (e.g., feature interactions, cyclical encodings).  
- Experiment with stacking or meta-learners.  
- Validate model stability on a hold-out set or via time-based splits, if available.

Overall, this pipeline delivers robust performance on the “House Prices” dataset and provides a solid foundation for further improvements.
