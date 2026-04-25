# B1. Problem Formulation
## (a) Problem Setup
### Target Variable:
The target variable is items_sold, which represents the number of items sold at a store.
### Input Features:
- Store-related: store_size, location_type, competition_density
- Promotion-related: promotion_type
- Time-based: year, month, day_of_week, is_month_end
- Other flags: is_weekend, is_festival
### Type of ML Problem:
This is a supervised regression problem.
### Justification:
The target variable (items_sold) is a continuous numerical value, so regression models such as Linear Regression and Random Forest are appropriate.

## (b) Why use items sold instead of revenue
Using items sold (sales volume) is more reliable than revenue because:

   - Revenue = price × quantity, and can be influenced by pricing changes or discounts
   - Items sold directly reflects customer demand and promotion effectiveness
### Broader Principle:
The target variable should directly represent the business objective and should not be affected by external or confounding factors.

## (c) Alternative Modeling Strategy
Instead of using a single global model across all 50 stores:

Use a store-aware or segmented modeling approach
  - Include store_id as a feature or
  - Build separate models for different store groups (e.g., urban vs rural)
### Justification:
Stores differ in size, location, and customer behavior. Feature importance also shows that store-specific factors significantly influence sales. Therefore, a single global model may fail to capture these differences.


# B2. Data and EDA Strategy
## (a) Data Joining and Dataset Design
   - The data comes from:
     - Transactions
     - Store attributes
     - Promotion details
     - Calendar data
### Joining Strategy:
  Join on store_id for store attributes
  Join on transaction_date for promotions and calendar data
### Final Dataset Grain:
  One row represents one store on one date
### Aggregations before modeling:
  Total items_sold per store per date
  Promotion applied (type or binary)
  Time-based indicators (weekend, festival)

## (b) EDA Strategy
1.Sales over time (month/day_of_week):
  - Identify seasonality and trends
  - Helps create time-based features
2.Sales vs promotion type:
  - Determine which promotions are most effective
3.Sales vs store attributes (store_size, location_type):
  - Understand store-level differences
4.Distribution of items_sold:
  - Detect skewness and outliers
5.Feature relationships (correlation or importance):
  - Identify key drivers of sales

### Impact on modeling:

- Helps in feature engineering (e.g., month, day_of_week)
- Helps in selecting relevant features
- Improves model performance

## (c) Handling Imbalance (80% no promotion)
### Issue:
The model may become biased toward non-promotion cases and ignore promotion effects.
### Solutions:
  - Use balanced sampling techniques
  - Apply class weighting or importance weighting
  - Evaluate model performance separately for promotion vs non-promotion data


# B3. Model Evaluation and Deployment 
## (a) Train-Test Split and Metrics
### Train-Test Split:
Use a time-based split, where:
- Training data = first 80%
- Test data = most recent 20%
### Why not random split:
Random splitting can cause data leakage, where future information is used during training, leading to unrealistic performance.
### Evaluation Metrics:
RMSE (Root Mean Squared Error): penalizes large errors more heavily
MAE (Mean Absolute Error): gives average prediction error
### Interpretation:
Lower RMSE and MAE indicate better model performance in predicting items sold.


## b) Explaining Different Recommendations
The model gives different promotion recommendations for the same store in different months because:

Time-based features like month and day_of_week capture seasonality
Store characteristics like store_size and location_type influence demand
Feature importance shows that these variables significantly impact predictions
Therefore, the model adapts to changing demand patterns across time and store conditions, leading to different recommendations.

## (c) Deployment Process
### Model Saving:
Save the trained pipeline (including preprocessing and model) using tools like pickle or joblib.
### Monthly Prediction Pipeline:
Load new monthly data
Apply the same preprocessing (feature engineering, encoding, scaling)
Generate predictions for items_sold
### Recommendation Generation:
Compare predicted sales across different promotions
Select the promotion with the highest predicted sales
### Monitoring:
Track RMSE and MAE over time
Monitor data drift and changes in feature distribution
Retrain the model when performance degrades
