import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.cluster import FeatureAgglomeration
from sklearn.metrics import r2_score
import lightgbm as lgbm
import shap

# -------------------------------------------------------------
# 1. Load Dataset & Drop the First Column
# -------------------------------------------------------------
print("Loading the Battery_RUL.csv dataset...")
df = pd.read_csv('Battery_RUL.csv')
print(f"Original dataset shape: {df.shape}")

# Drop the first column
first_col = df.columns[0]
print(f"Dropping the first column: '{first_col}'")
df = df.drop(columns=[first_col])
print(f"Dataset shape after dropping first column: {df.shape}")

# Separate features and target variable
target = 'RUL'
X = df.drop(columns=[target])
y = df[target]

print(f"Features available ({X.shape[1]} features): {list(X.columns)}")

# Check target distribution (3-digit precision)
print("\nTarget (RUL) distribution:")
print(f"Count: {len(y)}")
print(f"Mean: {y.mean():.3f}")
print(f"Std: {y.std():.3f}")
print(f"Min: {y.min():.3f}")
print(f"25%: {y.quantile(0.25):.3f}")
print(f"50%: {y.quantile(0.5):.3f}")
print(f"75%: {y.quantile(0.75):.3f}")
print(f"Max: {y.max():.3f}")

# -------------------------------------------------------------
# 2. Feature Selection Functions (Parametrized by k)
# -------------------------------------------------------------
def select_features_rf(X_in, y_in, k=5):
    k = min(k, X_in.shape[1])
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_in, y_in)
    importances = pd.Series(model.feature_importances_, index=X_in.columns)
    return importances.sort_values(ascending=False).index[:k].tolist()

def select_features_rf_shap(X_in, y_in, k=5):
    k = min(k, X_in.shape[1])
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(X_in, y_in)
    np.random.seed(42)
    sample_size = min(100, X_in.shape[0])
    sample_indices = np.random.choice(X_in.shape[0], size=sample_size, replace=False)
    X_sample = X_in.iloc[sample_indices]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    importances = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_in.columns)
    return importances.sort_values(ascending=False).index[:k].tolist()

def select_features_lgbm(X_in, y_in, k=5):
    k = min(k, X_in.shape[1])
    model = lgbm.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_in, y_in)
    importances = pd.Series(model.feature_importances_, index=X_in.columns)
    return importances.sort_values(ascending=False).index[:k].tolist()

def select_features_lgbm_shap(X_in, y_in, k=5):
    k = min(k, X_in.shape[1])
    model = lgbm.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    model.fit(X_in, y_in)
    np.random.seed(42)
    sample_size = min(100, X_in.shape[0])
    sample_indices = np.random.choice(X_in.shape[0], size=sample_size, replace=False)
    X_sample = X_in.iloc[sample_indices]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    importances = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_in.columns)
    return importances.sort_values(ascending=False).index[:k].tolist()

def select_features_fa(X_in, y_in=None, k=5):
    k = min(k, X_in.shape[1])
    fa = FeatureAgglomeration(n_clusters=k)
    fa.fit(X_in)
    cluster_assignments = {X_in.columns[i]: fa.labels_[i] for i in range(len(X_in.columns))}
    feature_variances = X_in.var()
    feature_scores = {feat: feature_variances[feat] for feat in cluster_assignments}
    return pd.Series(feature_scores).sort_values(ascending=False).index[:k].tolist()

def select_features_hvgs(X_in, y_in=None, k=5):
    k = min(k, X_in.shape[1])
    return X_in.var().sort_values(ascending=False).index[:k].tolist()

def select_features_spearman(X_in, y_in, k=5):
    k = min(k, X_in.shape[1])
    spearman_corr = X_in.apply(lambda col: col.corr(y_in, method='spearman')).abs()
    return spearman_corr.sort_values(ascending=False).index[:k].tolist()

# -------------------------------------------------------------
# 3. Leak-Free 10-Fold Cross-Validation Function
# -------------------------------------------------------------
def get_leak_free_cv10_r2_and_ci(X_data, y_data, selector_func, model_factory, k=5):
    """
    Performs 10-Fold CV where feature selection is performed strictly on X_train
    to eliminate data leakage. Evaluates on X_val and returns 95% CI (3 digits).
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X_data):
        X_train, X_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_train, y_val = y_data.iloc[train_idx], y_data.iloc[val_idx]
        
        # In-fold feature selection (zero leakage)
        selected_features = selector_func(X_train, y_train, k=k)
        
        # Fit model on training fold using selected features
        fold_model = model_factory()
        fold_model.fit(X_train[selected_features], y_train)
        
        # Predict on validation fold
        y_pred = fold_model.predict(X_val[selected_features])
        scores.append(r2_score(y_val, y_pred))
        
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_err = stats.sem(scores)
    n = len(scores)
    
    # 95% CI using Student's t-distribution (df = 9)
    ci_margin = stats.t.ppf(0.975, df=n - 1) * std_err if std_err > 0 else 0.0
    ci_lower = mean_score - ci_margin
    ci_upper = mean_score + ci_margin
    
    return f"{mean_score:.3f} ± {ci_margin:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

# -------------------------------------------------------------
# 4. Run Evaluation Across All 7 Methods
# -------------------------------------------------------------
results = {}

methods = {
    'Random Forest': {
        'selector': select_features_rf,
        'model_factory': lambda: RandomForestRegressor(random_state=42, n_jobs=-1)
    },
    'RF-SHAP': {
        'selector': select_features_rf_shap,
        'model_factory': lambda: RandomForestRegressor(random_state=42, n_jobs=-1)
    },
    'LightGBM': {
        'selector': select_features_lgbm,
        'model_factory': lambda: lgbm.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    },
    'LGBM-SHAP': {
        'selector': select_features_lgbm_shap,
        'model_factory': lambda: lgbm.LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    },
    'Feature Agglomeration': {
        'selector': select_features_fa,
        'model_factory': lambda: RandomForestRegressor(random_state=42, n_jobs=-1)
    },
    'HVGS': {
        'selector': select_features_hvgs,
        'model_factory': lambda: RandomForestRegressor(random_state=42, n_jobs=-1)
    },
    'Spearman': {
        'selector': select_features_spearman,
        'model_factory': lambda: RandomForestRegressor(random_state=42, n_jobs=-1)
    }
}

for name, config in methods.items():
    print(f"\nRunning {name} with leak-free 10-fold CV (Top 5 features)...")
    selector = config['selector']
    model_factory = config['model_factory']
    
    # 1. Leak-free 10-fold CV
    cv_ci = get_leak_free_cv10_r2_and_ci(X, y, selector, model_factory, k=5)
    
    # 2. Extract Top 5 features from full dataset
    top5 = selector(X, y, k=5)
    
    # 3. Drop top feature and extract Top 4 features
    X_reduced = X.drop(columns=[top5[0]])
    top4_reduced = selector(X_reduced, y, k=4)
    
    results[name] = {
        'CV10 95%CI': cv_ci,
        'Top 5': top5,
        'Top 4': top4_reduced
    }

# -------------------------------------------------------------
# 5. Format Output Table & Export
# -------------------------------------------------------------
print("\nPreparing results for output...")
output_data = []
for method, data in results.items():
    output_data.append({
        'method': method,
        'CV10 95%CI': data['CV10 95%CI'],
        'top5 features': ', '.join(data['Top 5']),
        'top4 features': ', '.join(data['Top 4'])
    })

results_df = pd.DataFrame(output_data)

# Print Final Summary Table to terminal
print("\nFinal Results Table:")
print(results_df.to_string(index=False))

# Save with utf-8-sig to prevent 'ﾂｱ' mojibake in Excel
results_df.to_csv('result.csv', index=False, encoding='utf-8-sig')
print("\nResults successfully saved to result.csv (encoded in utf-8-sig)")