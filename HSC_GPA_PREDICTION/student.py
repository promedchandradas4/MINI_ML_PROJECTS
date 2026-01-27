# Load dataset, optimize Random Forest with RandomizedSearchCV, evaluate and save
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint
import pickle

# ensure dataset is loaded
df = pd.read_csv('bangladesh_student_performance.csv')
X = df.drop('hsc_result', axis=1)
y = df['hsc_result']

# build preprocessor (same as earlier)
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor_opt = ColumnTransformer(transformers=[('num', num_pipeline, numeric_features), ('cat', cat_pipeline, categorical_features)])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

pipeline = Pipeline([('preprocessor', preprocessor_opt), ('model', RandomForestRegressor(random_state=42))])

# parameter distribution for randomized search
param_dist = {
    'model__n_estimators': randint(100, 500),
    'model__max_depth': [None, 10, 20, 30],
    'model__min_samples_split': randint(2, 10),
    'model__max_features': ['auto', 'sqrt', 0.5],
    'model__bootstrap': [True, False]
}

rand_search = RandomizedSearchCV(pipeline, param_dist, n_iter=30, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=1, random_state=42)
rand_search.fit(X_train, y_train)

best = rand_search.best_estimator_
print('Best params:', rand_search.best_params_)

# evaluate on test
y_pred = best.predict(X_test)
rmse_opt = mean_squared_error(y_test, y_pred)
r2_opt = r2_score(y_test, y_pred)
print('Optimized RF RMSE:', rmse_opt)
print('Optimized RF R2:', r2_opt)

# save optimized pipeline
with open('rf_optimized.pkl', 'wb') as f:
    pickle.dump(best, f)
print('Saved optimized model to rf_optimized.pkl')