import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
print("Import 완료")


data_path = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/from_saved_clean/data/clean_data.csv')
output_dir = data_path.parent / "ML model"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
print(f"shape: {df.shape}")


TARGET = "log_transfer_fee"

numeric_features = [
    "age_at_transfer",
    "age_squared",
    "U21_dummy",
    "O30_dummy",
    "UEFA_coeff_from",
    "league_level_diff",
    "contract_years_remaining",
    "covid_dummy",
    "season_proxy_flag"
]

categorical_features = [
    "sub_position",
    "from_league",
    "to_league",
    "transfer_season"
]

all_features = numeric_features + categorical_features
missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
print("없는 컬럼:", missing_cols)


df_model = df[all_features + [TARGET]].dropna(subset=[TARGET])

X = df_model[all_features]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

baseline_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(
    baseline_model, X_train, y_train,
    cv=cv, scoring="r2"
)
cv_rmse = -cross_val_score(
    baseline_model, X_train, y_train,
    cv=cv, scoring="neg_root_mean_squared_error"
)

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)

print(f"CV R²    : {cv_r2.mean():.4f} (sd={cv_r2.std():.4f})")
print(f"CV RMSE  : {cv_rmse.mean():.4f} (sd={cv_rmse.std():.4f})")
print(f"Test R²  : {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


feature_names = baseline_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = baseline_model.named_steps["model"].coef_

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

print("Top 15 positive:")
print(coef_table.head(15).to_string(index=False))
print("\nTop 15 negative:")
print(coef_table.tail(15).sort_values("coefficient").to_string(index=False))


baseline_summary = pd.DataFrame({
    "metric": [
        "train_n", "test_n", "cv_r2_mean", "cv_r2_sd",
        "cv_rmse_mean", "cv_rmse_sd", "test_r2", "test_rmse"
    ],
    "value": [
        len(X_train), len(X_test),
        cv_r2.mean(), cv_r2.std(),
        cv_rmse.mean(), cv_rmse.std(),
        test_r2, test_rmse
    ]
})

baseline_summary.to_csv(output_dir / "model1_metrics.csv", index=False)
coef_table.to_csv(output_dir / "model1_coefficients.csv", index=False)
print("저장 완료")


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
print("Import 완료")


data_path = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/from_saved_clean/data/clean_data.csv')
output_dir = data_path.parent / "ML model"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
print(f"shape: {df.shape}")


TARGET = "log_transfer_fee"

baseline_numeric_features = [
    "age_at_transfer",
    "age_squared",
    "U21_dummy",
    "O30_dummy",
    "UEFA_coeff_from",
    "league_level_diff",
    "contract_years_remaining",
    "covid_dummy",
    "season_proxy_flag"
]

traditional_numeric_features = [
    "Gls_per90",
    "Ast_per90",
    "Mins_Per_90_Playing"
]

categorical_features = [
    "sub_position",
    "from_league",
    "to_league",
    "transfer_season"
]

numeric_features = baseline_numeric_features + traditional_numeric_features
all_features = numeric_features + categorical_features

missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
print("없는 컬럼:", missing_cols)


df_model = df[all_features + [TARGET]].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model = df_model.dropna(subset=["from_league"])

print("Model 2 dataset shape:", df_model.shape)
print("\nMissing values:")
print(df_model[all_features].isna().sum())

X = df_model[all_features]
y = df_model[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

traditional_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(
    traditional_model, X_train, y_train,
    cv=cv, scoring="r2"
)
cv_rmse = -cross_val_score(
    traditional_model, X_train, y_train,
    cv=cv, scoring="neg_root_mean_squared_error"
)

traditional_model.fit(X_train, y_train)
y_pred = traditional_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)

print(f"CV R²    : {cv_r2.mean():.4f} (sd={cv_r2.std():.4f})")
print(f"CV RMSE  : {cv_rmse.mean():.4f} (sd={cv_rmse.std():.4f})")
print(f"Test R²  : {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


feature_names = traditional_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = traditional_model.named_steps["model"].coef_

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

print("Top 15 positive:")
print(coef_table.head(15).to_string(index=False))

print("\nTop 15 negative:")
print(coef_table.tail(15).sort_values("coefficient").to_string(index=False))


traditional_summary = pd.DataFrame({
    "metric": [
        "train_n", "test_n", "cv_r2_mean", "cv_r2_sd",
        "cv_rmse_mean", "cv_rmse_sd", "test_r2", "test_rmse"
    ],
    "value": [
        len(X_train), len(X_test),
        cv_r2.mean(), cv_r2.std(),
        cv_rmse.mean(), cv_rmse.std(),
        test_r2, test_rmse
    ]
})

traditional_summary.to_csv(output_dir / "model2_metrics.csv", index=False)
coef_table.to_csv(output_dir / "model2_coefficients.csv", index=False)

print("저장 완료")


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings

warnings.filterwarnings("ignore")
print("Import 완료")


data_path = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/from_saved_clean/data/clean_data.csv')
output_dir = data_path.parent / "ML model"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
print(f"shape: {df.shape}")


TARGET = "log_transfer_fee"

baseline_numeric_features = [
    "age_at_transfer",
    "age_squared",
    "U21_dummy",
    "O30_dummy",
    "UEFA_coeff_from",
    "league_level_diff",
    "contract_years_remaining",
    "covid_dummy",
    "season_proxy_flag"
]

advanced_numeric_features = [
    "xG_Per",
    "Succ_Take_per90",
    "PrgC_per90",
    "Cmp_percent_Total",
    "KP_per90",
    "Won_percent_Aerial",
    "Recov_per90"
]

categorical_features = [
    "sub_position",
    "from_league",
    "to_league",
    "transfer_season"
]

numeric_features = baseline_numeric_features + advanced_numeric_features
all_features = numeric_features + categorical_features

missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
print("없는 컬럼:", missing_cols)


df_model = df[all_features + [TARGET]].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model = df_model.dropna(subset=["from_league"])
df_model = df_model.replace([np.inf, -np.inf], np.nan)

inf_check = df_model[numeric_features].apply(lambda x: np.isinf(x).sum())
print("inf 개수:")
print(inf_check[inf_check > 0])

print("\nModel 3 dataset shape:", df_model.shape)
print("\nMissing values:")
print(df_model[all_features].isna().sum())

X = df_model[all_features]
y = df_model[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

advanced_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression()),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

cv_r2 = cross_val_score(
    advanced_model, X_train, y_train,
    cv=cv, scoring="r2"
)
cv_rmse = -cross_val_score(
    advanced_model, X_train, y_train,
    cv=cv, scoring="neg_root_mean_squared_error"
)

advanced_model.fit(X_train, y_train)
y_pred = advanced_model.predict(X_test)

test_r2 = r2_score(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)

print(f"CV R²    : {cv_r2.mean():.4f} (sd={cv_r2.std():.4f})")
print(f"CV RMSE  : {cv_rmse.mean():.4f} (sd={cv_rmse.std():.4f})")
print(f"Test R²  : {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")


feature_names = advanced_model.named_steps["preprocessor"].get_feature_names_out()
coefficients = advanced_model.named_steps["model"].coef_

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients
}).sort_values("coefficient", ascending=False)

print("Top 15 positive:")
print(coef_table.head(15).to_string(index=False))

print("\nTop 15 negative:")
print(coef_table.tail(15).sort_values("coefficient").to_string(index=False))


advanced_summary = pd.DataFrame({
    "metric": [
        "train_n", "test_n", "cv_r2_mean", "cv_r2_sd",
        "cv_rmse_mean", "cv_rmse_sd", "test_r2", "test_rmse"
    ],
    "value": [
        len(X_train), len(X_test),
        cv_r2.mean(), cv_r2.std(),
        cv_rmse.mean(), cv_rmse.std(),
        test_r2, test_rmse
    ]
})

advanced_summary.to_csv(output_dir / "model3_metrics.csv", index=False)
coef_table.to_csv(output_dir / "model3_coefficients.csv", index=False)

print("저장 완료")


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import root_mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings("ignore")
print("Import 완료")


data_path = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/from_saved_clean/data/clean_data.csv')
output_dir = data_path.parent / "ML model"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
print(f"shape: {df.shape}")


TARGET = "log_transfer_fee"

baseline_numeric_features = [
    "age_at_transfer",
    "age_squared",
    "U21_dummy",
    "O30_dummy",
    "UEFA_coeff_from",
    "league_level_diff",
    "contract_years_remaining",
    "covid_dummy",
    "season_proxy_flag"
]

traditional_numeric_features = [
    "Gls_per90",
    "Ast_per90",
    "Mins_Per_90_Playing"
]

advanced_numeric_features = [
    "xG_Per",
    "Succ_Take_per90",
    "PrgC_per90",
    "Cmp_percent_Total",
    "KP_per90",
    "Won_percent_Aerial",
    "Recov_per90"
]

candidate_numeric_features = traditional_numeric_features + advanced_numeric_features

categorical_features = [
    "sub_position",
    "from_league",
    "to_league",
    "transfer_season"
]

all_features = baseline_numeric_features + candidate_numeric_features + categorical_features

missing_cols = [c for c in all_features + [TARGET] if c not in df.columns]
print("없는 컬럼:", missing_cols)


df_model = df[all_features + [TARGET]].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model = df_model.dropna(subset=["from_league"])
df_model = df_model.replace([np.inf, -np.inf], np.nan)

print("Model 4 dataset shape:", df_model.shape)
print("\nMissing values:")
print(df_model[all_features].isna().sum())

X = df_model[all_features]
y = df_model[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


numeric_all_features = baseline_numeric_features + candidate_numeric_features

X_train_prep = X_train.copy()
X_test_prep = X_test.copy()

for col in numeric_all_features:
    X_train_prep[col] = pd.to_numeric(X_train_prep[col], errors="coerce")
    X_test_prep[col] = pd.to_numeric(X_test_prep[col], errors="coerce")

    median_val = X_train_prep[col].median()
    X_train_prep[col] = X_train_prep[col].fillna(median_val)
    X_test_prep[col] = X_test_prep[col].fillna(median_val)

for col in categorical_features:
    mode_val = X_train_prep[col].mode(dropna=True)[0]
    X_train_prep[col] = X_train_prep[col].fillna(mode_val)
    X_test_prep[col] = X_test_prep[col].fillna(mode_val)

def build_matrix(df_input, numeric_cols, categorical_cols):
    x_num = df_input[numeric_cols].reset_index(drop=True)

    x_cat = pd.get_dummies(
        df_input[categorical_cols].reset_index(drop=True),
        columns=categorical_cols,
        drop_first=True,
        dtype=float
    )

    return pd.concat([x_num, x_cat], axis=1)

def fit_ols(train_df, y_series, numeric_cols, categorical_cols):
    X_mat = build_matrix(train_df, numeric_cols, categorical_cols)
    X_sm = sm.add_constant(X_mat, has_constant="add")
    model = sm.OLS(y_series.reset_index(drop=True), X_sm).fit()
    return model, X_mat

def compute_vif(df_numeric):
    if df_numeric.shape[1] == 0:
        return pd.Series(dtype=float)
    if df_numeric.shape[1] == 1:
        return pd.Series({df_numeric.columns[0]: 1.0})

    X_vif = sm.add_constant(df_numeric.astype(float), has_constant="add")
    vif_values = pd.Series(
        [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])],
        index=df_numeric.columns
    ).sort_values(ascending=False)

    return vif_values

print("Preprocessing helper 준비 완료")


selected_candidates = candidate_numeric_features.copy()
vif_history = []

while len(selected_candidates) > 1:
    vif_table = compute_vif(X_train_prep[selected_candidates])
    max_vif = vif_table.max()

    if max_vif <= 5:
        break

    drop_var = vif_table.idxmax()
    vif_history.append({
        "step": len(vif_history) + 1,
        "dropped_variable": drop_var,
        "vif": float(max_vif)
    })

    selected_candidates.remove(drop_var)
    print(f"VIF 제거: {drop_var} (VIF = {max_vif:.2f})")

candidate_vif_after_filter = compute_vif(X_train_prep[selected_candidates])

if len(vif_history) == 0:
    print("VIF 제거 없음")

print("\nVIF 이후 남은 candidate variables:")
print(selected_candidates)

print("\nFinal candidate VIF:")
print(candidate_vif_after_filter)


current_candidates = selected_candidates.copy()
selection_history = []

current_model, _ = fit_ols(
    X_train_prep,
    y_train,
    baseline_numeric_features + current_candidates,
    categorical_features
)

current_aic = current_model.aic
current_bic = current_model.bic

print(f"Start AIC: {current_aic:.2f}")
print(f"Start BIC: {current_bic:.2f}")

while len(current_candidates) > 0:
    trial_results = []

    for var in current_candidates:
        trial_candidates = [v for v in current_candidates if v != var]

        trial_model, _ = fit_ols(
            X_train_prep,
            y_train,
            baseline_numeric_features + trial_candidates,
            categorical_features
        )

        trial_results.append({
            "dropped_variable": var,
            "aic": trial_model.aic,
            "bic": trial_model.bic
        })

    trial_df = pd.DataFrame(trial_results).sort_values(["bic", "aic"]).reset_index(drop=True)

    better_df = trial_df[
        (trial_df["aic"] < current_aic) &
        (trial_df["bic"] < current_bic)
    ]

    if better_df.empty:
        break

    best_row = better_df.iloc[0]
    drop_var = best_row["dropped_variable"]

    current_candidates.remove(drop_var)
    current_aic = best_row["aic"]
    current_bic = best_row["bic"]

    selection_history.append({
        "step": len(selection_history) + 1,
        "dropped_variable": drop_var,
        "aic_after": current_aic,
        "bic_after": current_bic
    })

    print(f"AIC/BIC 제거: {drop_var} -> AIC {current_aic:.2f}, BIC {current_bic:.2f}")

if len(selection_history) == 0:
    print("AIC/BIC 추가 제거 없음")

print("\n최종 selected candidate variables:")
print(current_candidates)


final_numeric_features = baseline_numeric_features + current_candidates

X_train_final = build_matrix(X_train_prep, final_numeric_features, categorical_features)
X_test_final = build_matrix(X_test_prep, final_numeric_features, categorical_features)

X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

X_train_sm = sm.add_constant(X_train_final, has_constant="add")
X_test_sm = sm.add_constant(X_test_final, has_constant="add")

final_model = sm.OLS(y_train.reset_index(drop=True), X_train_sm).fit()
y_pred = final_model.predict(X_test_sm)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
linreg = LinearRegression()

cv_r2 = cross_val_score(
    linreg,
    X_train_final,
    y_train,
    cv=cv,
    scoring="r2"
)

cv_rmse = -cross_val_score(
    linreg,
    X_train_final,
    y_train,
    cv=cv,
    scoring="neg_root_mean_squared_error"
)

test_r2 = r2_score(y_test, y_pred)
test_rmse = root_mean_squared_error(y_test, y_pred)

final_candidate_vif = compute_vif(X_train_prep[current_candidates])
max_candidate_vif = final_candidate_vif.max() if len(final_candidate_vif) > 0 else np.nan

print("Final candidate variables:")
print(current_candidates)

print(f"\nTrain AIC   : {final_model.aic:.2f}")
print(f"Train BIC   : {final_model.bic:.2f}")
print(f"Max VIF     : {max_candidate_vif:.2f}")
print(f"CV R²       : {cv_r2.mean():.4f} (sd={cv_r2.std():.4f})")
print(f"CV RMSE     : {cv_rmse.mean():.4f} (sd={cv_rmse.std():.4f})")
print(f"Test R²     : {test_r2:.4f}")
print(f"Test RMSE   : {test_rmse:.4f}")


coef_table = final_model.params.reset_index()
coef_table.columns = ["feature", "coefficient"]

coef_table_no_const = coef_table[coef_table["feature"] != "const"].copy()
coef_table_no_const = coef_table_no_const.sort_values("coefficient", ascending=False)

print("Top 15 positive:")
print(coef_table_no_const.head(15).to_string(index=False))

print("\nTop 15 negative:")
print(coef_table_no_const.tail(15).sort_values("coefficient").to_string(index=False))


model4_metrics = pd.DataFrame({
    "metric": [
        "train_n",
        "test_n",
        "selected_candidate_count",
        "train_aic",
        "train_bic",
        "max_candidate_vif",
        "cv_r2_mean",
        "cv_r2_sd",
        "cv_rmse_mean",
        "cv_rmse_sd",
        "test_r2",
        "test_rmse"
    ],
    "value": [
        len(X_train),
        len(X_test),
        len(current_candidates),
        final_model.aic,
        final_model.bic,
        max_candidate_vif,
        cv_r2.mean(),
        cv_r2.std(),
        cv_rmse.mean(),
        cv_rmse.std(),
        test_r2,
        test_rmse
    ]
})

selected_candidates_df = pd.DataFrame({
    "selected_candidate_variable": current_candidates
})

final_candidate_vif_df = final_candidate_vif.reset_index()
final_candidate_vif_df.columns = ["feature", "vif"]

vif_history_df = pd.DataFrame(vif_history)
selection_history_df = pd.DataFrame(selection_history)

test_predictions_df = pd.DataFrame({
    "actual": y_test.reset_index(drop=True),
    "predicted": y_pred
})

model4_metrics.to_csv(output_dir / "model4_metrics.csv", index=False)
coef_table.to_csv(output_dir / "model4_coefficients.csv", index=False)
selected_candidates_df.to_csv(output_dir / "model4_selected_candidates.csv", index=False)
final_candidate_vif_df.to_csv(output_dir / "model4_candidate_vif.csv", index=False)
vif_history_df.to_csv(output_dir / "model4_vif_history.csv", index=False)
selection_history_df.to_csv(output_dir / "model4_aic_bic_history.csv", index=False)
test_predictions_df.to_csv(output_dir / "model4_test_predictions.csv", index=False)

print("저장 완료")


import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor

import warnings
warnings.filterwarnings("ignore")

print("Import 완료")


data_dir = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/from_saved_clean/data')
ml_dir = data_dir / "ML model"

clean_path = data_dir / "clean_data.csv"
pred_path = ml_dir / "model4_test_predictions.csv"
selected_path = ml_dir / "model4_selected_candidates.csv"

df = pd.read_csv(clean_path)
pred_df = pd.read_csv(pred_path)
selected_df = pd.read_csv(selected_path)

selected_candidates = selected_df["selected_candidate_variable"].dropna().tolist()

print(f"clean_data shape: {df.shape}")
print(f"model4_test_predictions shape: {pred_df.shape}")
print("selected candidates:")
print(selected_candidates)


TARGET = "log_transfer_fee"

baseline_numeric_features = [
    "age_at_transfer",
    "age_squared",
    "U21_dummy",
    "O30_dummy",
    "UEFA_coeff_from",
    "league_level_diff",
    "contract_years_remaining",
    "covid_dummy",
    "season_proxy_flag"
]

categorical_features = [
    "sub_position",
    "from_league",
    "to_league",
    "transfer_season"
]

final_numeric_features = baseline_numeric_features + selected_candidates
all_features = final_numeric_features + categorical_features

id_cols = [
    "player_name",
    "transfer_season",
    "position",
    "sub_position",
    "from_club_name",
    "to_club_name_full",
    "from_league",
    "to_league",
    "transfer_fee_raw",
    "market_value_in_eur"
]

keep_cols = list(dict.fromkeys(id_cols + all_features + [TARGET]))

df_model = df[keep_cols].copy()
df_model = df_model.dropna(subset=[TARGET])
df_model = df_model.dropna(subset=["from_league"])
df_model = df_model.replace([np.inf, -np.inf], np.nan)

print(f"filtered df_model shape: {df_model.shape}")

X = df_model[all_features]
y = df_model[TARGET]
meta = df_model[id_cols].copy()

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X, y, meta, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}")
print(f"Test:  {X_test.shape}")


if len(pred_df) != len(y_test):
    raise ValueError("saved predictions row count != recreated test set row count")

if not np.allclose(
    y_test.reset_index(drop=True).to_numpy(),
    pred_df["actual"].to_numpy(),
    atol=1e-12
):
    raise ValueError("Model 4 test split 재현 실패: actual values do not match saved predictions")

print("Model 4 test split 재현 확인 완료")


results_df = meta_test.reset_index(drop=True).copy()

results_df["actual_log_fee"] = pred_df["actual"].reset_index(drop=True)
results_df["predicted_log_fee"] = pred_df["predicted"].reset_index(drop=True)

results_df["residual"] = results_df["actual_log_fee"] - results_df["predicted_log_fee"]
results_df["pct_gap"] = np.exp(results_df["residual"]) - 1
results_df["pct_gap_percent"] = results_df["pct_gap"] * 100

results_df["actual_fee_eur"] = np.exp(results_df["actual_log_fee"])
results_df["predicted_fee_eur"] = np.exp(results_df["predicted_log_fee"])
results_df["fee_gap_eur"] = results_df["actual_fee_eur"] - results_df["predicted_fee_eur"]

results_df["signal_relative_to_model"] = np.where(
    results_df["residual"] >= 0,
    "above_model_prediction",
    "below_model_prediction"
)

results_df["pricing_label"] = np.where(
    results_df["residual"] >= 0,
    "model_implied_overpriced",
    "model_implied_underpriced"
)

results_df = results_df.sort_values("residual", ascending=False).reset_index(drop=True)

print("results preview:")
print(results_df.head(10).to_string(index=False))


top_n = 20

top_overpriced = results_df.nlargest(top_n, "residual").copy()
top_underpriced = results_df.nsmallest(top_n, "residual").copy()

print("Top 10 above model prediction:")
print(
    top_overpriced[
        ["player_name", "transfer_season", "to_club_name_full", "residual", "pct_gap_percent"]
    ].head(10).to_string(index=False)
)

print("\nTop 10 below model prediction:")
print(
    top_underpriced[
        ["player_name", "transfer_season", "to_club_name_full", "residual", "pct_gap_percent"]
    ].head(10).to_string(index=False)
)


X_train_prep = X_train.copy()
X_test_prep = X_test.copy()

for col in final_numeric_features:
    X_train_prep[col] = pd.to_numeric(X_train_prep[col], errors="coerce")
    X_test_prep[col] = pd.to_numeric(X_test_prep[col], errors="coerce")

    median_val = X_train_prep[col].median()
    X_train_prep[col] = X_train_prep[col].fillna(median_val)
    X_test_prep[col] = X_test_prep[col].fillna(median_val)

for col in categorical_features:
    mode_val = X_train_prep[col].mode(dropna=True)[0]
    X_train_prep[col] = X_train_prep[col].fillna(mode_val)
    X_test_prep[col] = X_test_prep[col].fillna(mode_val)

def build_matrix(df_input, numeric_cols, categorical_cols):
    x_num = df_input[numeric_cols].reset_index(drop=True)

    x_cat = pd.get_dummies(
        df_input[categorical_cols].reset_index(drop=True),
        columns=categorical_cols,
        drop_first=True,
        dtype=float
    )

    return pd.concat([x_num, x_cat], axis=1)

X_train_final = build_matrix(X_train_prep, final_numeric_features, categorical_features)
X_test_final = build_matrix(X_test_prep, final_numeric_features, categorical_features)
X_test_final = X_test_final.reindex(columns=X_train_final.columns, fill_value=0)

print(f"X_train_final shape: {X_train_final.shape}")
print(f"X_test_final shape:  {X_test_final.shape}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)

lin_std = LinearRegression()
lin_std.fit(X_train_scaled, y_train)

std_coef_df = pd.DataFrame({
    "feature": X_train_final.columns,
    "standardized_coefficient": lin_std.coef_,
    "abs_standardized_coefficient": np.abs(lin_std.coef_)
}).sort_values("abs_standardized_coefficient", ascending=False)

print("Top 15 standardized coefficient importance:")
print(std_coef_df.head(15).to_string(index=False))


rf_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_final, y_train)

rf_perm = permutation_importance(
    rf_model,
    X_test_final,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2",
    n_jobs=-1
)

rf_importance_df = pd.DataFrame({
    "feature": X_test_final.columns,
    "rf_perm_importance_mean": rf_perm.importances_mean,
    "rf_perm_importance_sd": rf_perm.importances_std
}).sort_values("rf_perm_importance_mean", ascending=False)

print("Top 15 RF permutation importance:")
print(rf_importance_df.head(15).to_string(index=False))


xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_final, y_train)

xgb_perm = permutation_importance(
    xgb_model,
    X_test_final,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2",
    n_jobs=-1
)

xgb_importance_df = pd.DataFrame({
    "feature": X_test_final.columns,
    "xgb_perm_importance_mean": xgb_perm.importances_mean,
    "xgb_perm_importance_sd": xgb_perm.importances_std
}).sort_values("xgb_perm_importance_mean", ascending=False)

print("Top 15 XGB permutation importance:")
print(xgb_importance_df.head(15).to_string(index=False))


importance_summary = std_coef_df[
    ["feature", "standardized_coefficient", "abs_standardized_coefficient"]
].merge(
    rf_importance_df,
    on="feature",
    how="outer"
).merge(
    xgb_importance_df,
    on="feature",
    how="outer"
)

importance_summary = importance_summary.sort_values(
    "abs_standardized_coefficient",
    ascending=False
)

results_df.to_csv(ml_dir / "results.csv", index=False)
top_overpriced.to_csv(ml_dir / "model4_top_overpriced_candidates.csv", index=False)
top_underpriced.to_csv(ml_dir / "model4_top_underpriced_candidates.csv", index=False)

std_coef_df.to_csv(ml_dir / "model4_standardized_coef_importance.csv", index=False)
rf_importance_df.to_csv(ml_dir / "model4_rf_permutation_importance.csv", index=False)
xgb_importance_df.to_csv(ml_dir / "model4_xgb_permutation_importance.csv", index=False)
importance_summary.to_csv(ml_dir / "model4_feature_importance_summary.csv", index=False)

print("저장 완료")


import statsmodels.api as sm

# Model 4와 같은 train-fit 방식으로 전체 1407명 예측값 생성
X_train_sm = sm.add_constant(X_train_final, has_constant="add")
X_test_sm = sm.add_constant(X_test_final, has_constant="add")

ols_all_view = sm.OLS(y_train.reset_index(drop=True), X_train_sm).fit()

train_pred = ols_all_view.predict(X_train_sm)
test_pred = ols_all_view.predict(X_test_sm)

def make_results_block(meta_df, actual_y, pred_y, split_name):
    out = meta_df.reset_index(drop=True).copy()
    out["split"] = split_name
    out["actual_log_fee"] = actual_y.reset_index(drop=True)
    out["predicted_log_fee"] = np.array(pred_y)
    out["residual"] = out["actual_log_fee"] - out["predicted_log_fee"]
    out["pct_gap"] = np.exp(out["residual"]) - 1
    out["pct_gap_percent"] = out["pct_gap"] * 100
    out["actual_fee_eur"] = np.exp(out["actual_log_fee"])
    out["predicted_fee_eur"] = np.exp(out["predicted_log_fee"])
    out["fee_gap_eur"] = out["actual_fee_eur"] - out["predicted_fee_eur"]
    out["signal_relative_to_model"] = np.where(
        out["residual"] >= 0,
        "above_model_prediction",
        "below_model_prediction"
    )
    out["pricing_label"] = np.where(
        out["residual"] >= 0,
        "model_implied_overpriced",
        "model_implied_underpriced"
    )
    return out

results_train_1125 = make_results_block(meta_train, y_train, train_pred, "train")
results_test_282 = make_results_block(meta_test, y_test, test_pred, "test")

results_all_1407 = pd.concat(
    [results_train_1125, results_test_282],
    axis=0,
    ignore_index=True
)

results_all_1407 = results_all_1407.sort_values("residual", ascending=False).reset_index(drop=True)

results_train_1125.to_csv(ml_dir / "results_train_1125.csv", index=False)
results_test_282.to_csv(ml_dir / "results_test_282.csv", index=False)
results_all_1407.to_csv(ml_dir / "results_all_1407.csv", index=False)

print(results_all_1407.shape)
print(results_all_1407["split"].value_counts())
print("저장 완료")



# Audit-only exports. No fitting or selection changes.
import json
from pathlib import Path
audit_dir = (Path(__file__).resolve().parents[1]/"data/work"/'audit/reproduction/audit_evidence')
X_train_sm.to_csv(audit_dir / "model4_training_design.csv", index=False)
X_test_sm.to_csv(audit_dir / "model4_test_design.csv", index=False)
pd.DataFrame({"original_row": X_train.index}).to_csv(audit_dir / "train_indices.csv", index=False)
pd.DataFrame({"original_row": X_test.index}).to_csv(audit_dir / "test_indices.csv", index=False)
A = X_train_sm.to_numpy(dtype=float)
scaled = A / np.where(np.linalg.norm(A, axis=0) == 0, 1, np.linalg.norm(A, axis=0))
info = {"shape_with_intercept": list(A.shape), "rank_raw": int(np.linalg.matrix_rank(A)), "rank_column_scaled": int(np.linalg.matrix_rank(scaled)), "singular_values_scaled": np.linalg.svd(scaled, compute_uv=False).tolist()}
(audit_dir / "design_rank.json").write_text(json.dumps(info, indent=2))
