import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import r2_score
import lightgbm as lgbm
import shap

# Load the dataset
print("Loading the Battery_RUL.csv dataset...")
df = pd.read_csv('Battery_RUL.csv')
print(f"Dataset shape: {df.shape}")

# Separate features and target variable
print("Separating features and target variable...")
target = 'RUL'
X = df.drop(columns=[target])
y = df[target]

# Check the distribution of the target variable
print("Analyzing the distribution of the target (RUL) variable...")
print(f"Target (RUL) distribution:")
print(f"Count: {len(y)}")
print(f"Mean: {y.mean()}")
print(f"Std: {y.std()}")
print(f"Min: {y.min()}")
print(f"25%: {y.quantile(0.25)}")
print(f"50%: {y.quantile(0.5)}")
print(f"75%: {y.quantile(0.75)}")
print(f"Max: {y.max()}")

# Function to calculate CV R-squared
def get_cv_r2(X, y, model):
    """Calculate 5-fold cross-validation R-squared score."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    return r2_scores.mean()

# Initialize results dictionary
results = {}

# Method 1: Random Forest feature importance
print("\n1. Running Random Forest feature importance...")
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
# Get feature importances and sort in descending order
rf_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
# Select top 6 features based on importance
top6_rf = rf_importances.index[:6].tolist()
# Calculate cross-validation R² score with all features
rf_cv_score = get_cv_r2(X, y, rf)

# Reduce dataset - remove top feature and rerun feature selection
print("   Creating reduced dataset without the top feature...")
X_reduced = X.drop(columns=[top6_rf[0]])
rf_reduced = RandomForestRegressor(random_state=42)
rf_reduced.fit(X_reduced, y)
rf_reduced_importances = pd.Series(rf_reduced.feature_importances_, index=X_reduced.columns).sort_values(ascending=False)
# Select top 5 features from reduced dataset
top5_rf_reduced = rf_reduced_importances.index[:5].tolist()
rf_cv_score_reduced = get_cv_r2(X_reduced, y, rf_reduced)

# Store results
results['Random Forest'] = {
    'CV Accuracy': rf_cv_score, 
    'Top 6': top6_rf,
    'Top 5 (reduced)': top5_rf_reduced
}

# Method 2: RF-SHAP
print("\n2. Running Random Forest with SHAP feature importance...")
rf_for_shap = RandomForestRegressor(random_state=42)
rf_for_shap.fit(X, y)

# Use a sample of 100 random instances for SHAP analysis to improve efficiency
print("   Selecting 100 random instances for SHAP analysis...")
np.random.seed(42)
sample_indices = np.random.choice(X.shape[0], size=min(100, X.shape[0]), replace=False)
X_sample = X.iloc[sample_indices]

# Calculate SHAP values
explainer = shap.TreeExplainer(rf_for_shap)
shap_values = explainer.shap_values(X_sample)

# Calculate mean absolute SHAP values per feature and sort
rf_shap_importances = pd.DataFrame(np.abs(shap_values).mean(0), index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
top6_rf_shap = rf_shap_importances.index[:6].tolist()

# Reduce dataset - remove top feature and rerun
print("   Creating reduced dataset without the top feature...")
X_reduced = X.drop(columns=[top6_rf_shap[0]])
rf_shap_reduced = RandomForestRegressor(random_state=42)
rf_shap_reduced.fit(X_reduced, y)

# Use the same sample indices but for reduced dataset
X_reduced_sample = X_reduced.iloc[sample_indices]
explainer_reduced = shap.TreeExplainer(rf_shap_reduced)
shap_values_reduced = explainer_reduced.shap_values(X_reduced_sample)

# Get feature importances for reduced dataset
rf_shap_importances_reduced = pd.DataFrame(np.abs(shap_values_reduced).mean(0), index=X_reduced.columns, columns=['importance']).sort_values('importance', ascending=False)
top5_rf_shap_reduced = rf_shap_importances_reduced.index[:5].tolist()
rf_shap_cv_score_reduced = get_cv_r2(X_reduced, y, rf_shap_reduced)

# Store results
results['RF-SHAP'] = {
    'CV Accuracy': rf_cv_score,  # Using the same RF model accuracy
    'Top 6': top6_rf_shap,
    'Top 5 (reduced)': top5_rf_shap_reduced
}

# Method 3: LightGBM
print("\n3. Running LightGBM feature importance...")
lgbm_model = lgbm.LGBMRegressor(random_state=42)
lgbm_model.fit(X, y)
# Get built-in feature importances and sort
lgbm_importances = pd.Series(lgbm_model.feature_importances_, index=X.columns).sort_values(ascending=False)
top6_lgbm = lgbm_importances.index[:6].tolist()
lgbm_cv_score = get_cv_r2(X, y, lgbm_model)

# Reduce dataset - remove top feature
print("   Creating reduced dataset without the top feature...")
X_reduced = X.drop(columns=[top6_lgbm[0]])
lgbm_reduced = lgbm.LGBMRegressor(random_state=42)
lgbm_reduced.fit(X_reduced, y)
lgbm_reduced_importances = pd.Series(lgbm_reduced.feature_importances_, index=X_reduced.columns).sort_values(ascending=False)
top5_lgbm_reduced = lgbm_reduced_importances.index[:5].tolist()
lgbm_cv_score_reduced = get_cv_r2(X_reduced, y, lgbm_reduced)

# Store results
results['LightGBM'] = {
    'CV Accuracy': lgbm_cv_score, 
    'Top 6': top6_lgbm,
    'Top 5 (reduced)': top5_lgbm_reduced
}

# Method 4: LGBM-SHAP
print("\n4. Running LightGBM with SHAP feature importance...")
lgbm_for_shap = lgbm.LGBMRegressor(random_state=42)
lgbm_for_shap.fit(X, y)

# Use a sample of 100 random instances for SHAP analysis
print("   Selecting 100 random instances for SHAP analysis...")
# We reuse the same sample indices as in RF-SHAP for consistency
X_sample = X.iloc[sample_indices]

# Calculate SHAP values
explainer = shap.TreeExplainer(lgbm_for_shap)
shap_values = explainer.shap_values(X_sample)

# Calculate mean absolute SHAP values per feature and sort
lgbm_shap_importances = pd.DataFrame(np.abs(shap_values).mean(0), index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
top6_lgbm_shap = lgbm_shap_importances.index[:6].tolist()

# Reduce dataset - remove top feature
print("   Creating reduced dataset without the top feature...")
X_reduced = X.drop(columns=[top6_lgbm_shap[0]])
lgbm_shap_reduced = lgbm.LGBMRegressor(random_state=42)
lgbm_shap_reduced.fit(X_reduced, y)

# Use the same sample indices but for reduced dataset
X_reduced_sample = X_reduced.iloc[sample_indices]
explainer_reduced = shap.TreeExplainer(lgbm_shap_reduced)
shap_values_reduced = explainer_reduced.shap_values(X_reduced_sample)

# Get feature importances for reduced dataset
lgbm_shap_importances_reduced = pd.DataFrame(np.abs(shap_values_reduced).mean(0), index=X_reduced.columns, columns=['importance']).sort_values('importance', ascending=False)
top5_lgbm_shap_reduced = lgbm_shap_importances_reduced.index[:5].tolist()
lgbm_shap_cv_score_reduced = get_cv_r2(X_reduced, y, lgbm_shap_reduced)

# Store results
results['LGBM-SHAP'] = {
    'CV Accuracy': lgbm_cv_score,  # Using the same LGBM model accuracy
    'Top 6': top6_lgbm_shap,
    'Top 5 (reduced)': top5_lgbm_shap_reduced
}

# Method 5: Feature Agglomeration
print("\n5. Running Feature Agglomeration for feature selection...")
# No transformation or scaling - use pure Feature Agglomeration with 6 clusters
print("   Using 6 clusters for feature agglomeration")
fa = FeatureAgglomeration(n_clusters=6)
fa.fit(X)

# Get the cluster assignments for each feature
cluster_assignments = {X.columns[i]: fa.labels_[i] for i in range(len(X.columns))}

# Calculate variance of each feature without normalization
feature_variances = X.var()

# For each feature, create a score that combines its variance and uniqueness
feature_scores = {}
for feature, cluster in cluster_assignments.items():
    # Higher variance = more important feature
    feature_scores[feature] = feature_variances[feature]
    
# Sort features by their scores
fa_importances = pd.Series(feature_scores).sort_values(ascending=False)

# Get top 6 features across all clusters
top6_fa = fa_importances.index[:6].tolist()

# Evaluate with Random Forest using the top features
print("   Evaluating top 6 features with Random Forest...")
rf_fa = RandomForestRegressor(random_state=42)
fa_cv_score = get_cv_r2(X[top6_fa], y, rf_fa)

# Reduce dataset - remove top feature
print("   Creating reduced dataset without the top feature...")
X_reduced = X.drop(columns=[top6_fa[0]])

# Re-run feature agglomeration on reduced dataset with 6 clusters
print("   Using 6 clusters for the reduced dataset")
fa_reduced = FeatureAgglomeration(n_clusters=6)
fa_reduced.fit(X_reduced)

# Get the cluster assignments for each feature in reduced dataset
cluster_assignments_reduced = {X_reduced.columns[i]: fa_reduced.labels_[i] for i in range(len(X_reduced.columns))}

# Calculate variance of each feature in the reduced dataset
feature_variances_reduced = X_reduced.var()

# For each feature, create a score that combines its variance and uniqueness
feature_scores_reduced = {}
for feature, cluster in cluster_assignments_reduced.items():
    # Higher variance = more important feature
    feature_scores_reduced[feature] = feature_variances_reduced[feature]
    
# Sort features by their scores
fa_importances_reduced = pd.Series(feature_scores_reduced).sort_values(ascending=False)

# Get top 5 features across all clusters
top5_fa_reduced = fa_importances_reduced.index[:5].tolist()

# Evaluate with Random Forest using the top features
rf_fa_reduced = RandomForestRegressor(random_state=42)
fa_reduced_cv_score = get_cv_r2(X_reduced[top5_fa_reduced], y, rf_fa_reduced)

# Store results
results['Feature Agglomeration'] = {
    'CV Accuracy': fa_cv_score, 
    'Top 6': top6_fa,
    'Top 5 (reduced)': top5_fa_reduced
}

# Method 6: Highly Variable Gene Selection (adapted for general features)
print("\n6. Running Highly Variable Feature Selection...")
# Calculate variance for each feature to find the most variable ones
feature_variances = X.var().sort_values(ascending=False)
top6_hvgs = feature_variances.index[:6].tolist()

# Evaluate with Random Forest using the top variable features
print("   Evaluating top variable features with Random Forest...")
rf_hvgs = RandomForestRegressor(random_state=42)
hvgs_cv_score = get_cv_r2(X[top6_hvgs], y, rf_hvgs)

# Reduce dataset - remove top feature
print("   Creating reduced dataset without the top variable feature...")
X_reduced = X.drop(columns=[top6_hvgs[0]])
feature_variances_reduced = X_reduced.var().sort_values(ascending=False)
top5_hvgs_reduced = feature_variances_reduced.index[:5].tolist()

# Evaluate reduced dataset
rf_hvgs_reduced = RandomForestRegressor(random_state=42)
hvgs_reduced_cv_score = get_cv_r2(X_reduced[top5_hvgs_reduced], y, rf_hvgs_reduced)

# Store results
results['HVGS'] = {
    'CV Accuracy': hvgs_cv_score, 
    'Top 6': top6_hvgs,
    'Top 5 (reduced)': top5_hvgs_reduced
}

# Method 7: Spearman Correlation
print("\n7. Running Spearman Correlation for feature selection...")
# Calculate absolute Spearman correlation between each feature and the target
spearman_corr = X.apply(lambda col: col.corr(y, method='spearman')).abs().sort_values(ascending=False)
top6_spearman = spearman_corr.index[:6].tolist()

# Evaluate with Random Forest using the top correlated features
print("   Evaluating top correlated features with Random Forest...")
rf_spearman = RandomForestRegressor(random_state=42)
spearman_cv_score = get_cv_r2(X[top6_spearman], y, rf_spearman)

# Reduce dataset - remove top feature
print("   Creating reduced dataset without the top correlated feature...")
X_reduced = X.drop(columns=[top6_spearman[0]])
spearman_corr_reduced = X_reduced.apply(lambda col: col.corr(y, method='spearman')).abs().sort_values(ascending=False)
top5_spearman_reduced = spearman_corr_reduced.index[:5].tolist()

# Evaluate reduced dataset
rf_spearman_reduced = RandomForestRegressor(random_state=42)
spearman_reduced_cv_score = get_cv_r2(X_reduced[top5_spearman_reduced], y, rf_spearman_reduced)

# Store results
results['Spearman'] = {
    'CV Accuracy': spearman_cv_score, 
    'Top 6': top6_spearman,
    'Top 5 (reduced)': top5_spearman_reduced
}

# Prepare results for CSV output
print("\nPreparing results for output...")
output_data = []
for method, data in results.items():
    top6_str = ', '.join(data['Top 6'])
    top5_str = ', '.join(data['Top 5 (reduced)'])
    output_data.append({
        'Method': method,
        'CV Accuracy': round(data['CV Accuracy'], 4),
        'Top 6 Features': top6_str,
        'Top 5 Features (reduced)': top5_str
    })

results_df = pd.DataFrame(output_data)

# Save results to CSV
results_df.to_csv('result.csv', index=False)
print("Results saved to result.csv")
