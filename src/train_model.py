import pandas as pd

df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# check how many matches per season, so we pick a sensible cutoff for the test set
print(df["season"].value_counts().sort_index())
train_df = df[~df["season"].isin(["2025", "2026"])].copy()
test_df = df[df["season"].isin(["2025", "2026"])].copy()

print(f"Train: {train_df.shape[0]} matches ({train_df['season'].min()} to {train_df['season'].max()})")
print(f"Test: {test_df.shape[0]} matches (seasons: {sorted(test_df['season'].unique())})")
print(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
# columns we'll actually feed into the model
categorical_cols = ["team1", "team2", "venue", "toss_winner", "toss_decision", "team1_home_away", "team2_home_away"]
numeric_cols = ["team1_last5_win_pct", "team2_last5_win_pct", "team1_h2h_win_pct", "team2_h2h_win_pct"]

target_col = "team1_won"

# combine train+test temporarily just for consistent one-hot columns, then split back apart
combined = pd.concat([train_df, test_df], keys=["train", "test"])
combined_encoded = pd.get_dummies(combined, columns=categorical_cols)

X_train = combined_encoded.loc["train"][numeric_cols + [c for c in combined_encoded.columns if c not in df.columns]]
X_test = combined_encoded.loc["test"][numeric_cols + [c for c in combined_encoded.columns if c not in df.columns]]
y_train = train_df[target_col]
y_test = test_df[target_col]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(X_train.columns.tolist()[:15])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

print("=== Logistic Regression (Baseline) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# simplified feature set: drop raw team/venue identity, keep only engineered signal
categorical_cols_simple = ["toss_decision", "team1_home_away", "team2_home_away"]
numeric_cols = ["team1_last5_win_pct", "team2_last5_win_pct", "team1_h2h_win_pct", "team2_h2h_win_pct"]

combined_simple = pd.get_dummies(combined, columns=categorical_cols_simple)
X_train_simple = combined_simple.loc["train"][numeric_cols + [c for c in combined_simple.columns if any(c.startswith(p) for p in categorical_cols_simple)]]
X_test_simple = combined_simple.loc["test"][numeric_cols + [c for c in combined_simple.columns if any(c.startswith(p) for p in categorical_cols_simple)]]

print(f"Simplified X_train shape: {X_train_simple.shape}")

log_reg_simple = LogisticRegression(max_iter=1000)
log_reg_simple.fit(X_train_simple, y_train)
y_pred_simple = log_reg_simple.predict(X_test_simple)
y_proba_simple = log_reg_simple.predict_proba(X_test_simple)[:, 1]

print("=== Logistic Regression (Simplified Features) ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_simple):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_simple):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_simple):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_simple):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_simple):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_simple))
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
rf.fit(X_train_simple, y_train)
rf_pred = rf.predict(X_test_simple)
rf_proba = rf.predict_proba(X_test_simple)[:, 1]

print("=== Random Forest (Simplified Features) ===")
print(f"Accuracy:  {accuracy_score(y_test, rf_pred):.3f}")
print(f"Precision: {precision_score(y_test, rf_pred):.3f}")
print(f"Recall:    {recall_score(y_test, rf_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, rf_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, rf_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# XGBoost
xgb = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric='logloss', random_state=42)
xgb.fit(X_train_simple, y_train)
xgb_pred = xgb.predict(X_test_simple)
xgb_proba = xgb.predict_proba(X_test_simple)[:, 1]

print("=== XGBoost (Simplified Features) ===")
print(f"Accuracy:  {accuracy_score(y_test, xgb_pred):.3f}")
print(f"Precision: {precision_score(y_test, xgb_pred):.3f}")
print(f"Recall:    {recall_score(y_test, xgb_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, xgb_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, xgb_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbose=-1)
lgbm.fit(X_train_simple, y_train)
lgbm_pred = lgbm.predict(X_test_simple)
lgbm_proba = lgbm.predict_proba(X_test_simple)[:, 1]

print("=== LightGBM (Simplified Features) ===")
print(f"Accuracy:  {accuracy_score(y_test, lgbm_pred):.3f}")
print(f"Precision: {precision_score(y_test, lgbm_pred):.3f}")
print(f"Recall:    {recall_score(y_test, lgbm_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, lgbm_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, lgbm_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, lgbm_pred))
from sklearn.model_selection import GridSearchCV

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=5,
    scoring="roc_auc",
)
grid_search.fit(X_train_simple, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV ROC-AUC:", grid_search.best_score_)

best_log_reg = grid_search.best_estimator_
tuned_pred = best_log_reg.predict(X_test_simple)
tuned_proba = best_log_reg.predict_proba(X_test_simple)[:, 1]

print("=== Logistic Regression (Tuned) ===")
print(f"Accuracy:  {accuracy_score(y_test, tuned_pred):.3f}")
print(f"Precision: {precision_score(y_test, tuned_pred):.3f}")
print(f"Recall:    {recall_score(y_test, tuned_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, tuned_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, tuned_proba):.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, tuned_pred))