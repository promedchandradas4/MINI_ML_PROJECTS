#import necessary libraries
import numpy as np
import pandas as pd
from scipy.stats import randint
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



# Task 1: Data Loading

df = pd.read_csv('stock_data.csv')

# Optional profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df, title="Stock Data Profiling Report")
profile.to_file("stock_data_profiling_report.html")


# Task 2: Data Preprocessing

df = df.drop(columns=['Unnamed: 0'], errors='ignore')

X = df.drop('Stock_2', axis=1)
y = df['Stock_2']

numeric_features = X.select_dtypes(include=np.number).columns.tolist()

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numeric_features)
    ],
    remainder='drop'
)

print("Preprocessor ready with numeric features:", numeric_features)


# Task 3: Pipeline Creation

knn = KNeighborsRegressor(n_neighbors=50)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', knn)
])


# Data Splitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Task 5: Model Training

pipeline.fit(X_train, y_train)


# Task 6: Cross-Validation

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f"RMSE per fold: {cv_rmse}")
print(f"Average RMSE : {np.mean(cv_rmse):.4f}")
print(f"Standard deviation : {np.std(cv_rmse):.4f}")


# Task 7: Hyperparameter Tuning
knn_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor())
])

param_dist = {
    'model__n_neighbors': randint(5, 100),
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2],
    'model__leaf_size': randint(20, 50)
}

random_search = RandomizedSearchCV(
    estimator=knn_pipe,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("Best KNN Params:", random_search.best_params_)
print("Best CV R2 Score:", random_search.best_score_)

y_pred = random_search.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")


# Task 8 & 9: Best Model Evaluation

best_model_pipeline = random_search.best_estimator_
y_pred_best_model = best_model_pipeline.predict(X_test)

print("\n--- Best Model Performance on Test Set ---")
print("Best Params:", random_search.best_params_)
print(f"R2 Score: {r2_score(y_test, y_pred_best_model):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best_model)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_best_model):.4f}")


# Task 10: Save the Model Pipeline
with open("stockPrice_knn_pipeline.pkl", "wb") as f:
    pickle.dump(best_model_pipeline, f)

print("✅ KNN pipeline saved as stockPrice_knn_pipeline.pkl")
