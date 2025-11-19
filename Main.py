
# numpy is used for numerical computing functions.
# pandas is used to crete dataframes to obtain spreadsheet and sql data files. Built ontop of Numpy.
# SKlearn is a machine elarning toolkit that has ready to use algorithms.
    # standard library in python.


import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# optional XGBoost
have_xgb = True
try:
    from xgboost import XGBClassifier
except Exception:
    have_xgb = False
    print("[INFO] xgboost not found. (pip install xgboost) — we’ll skip it for now.")


# loading in the dataset:
iris = load_iris()
# X is the set of data without headers, confirmed above
X = pd.DataFrame(
    iris.data,
    columns=[c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names]
)

# Y are the headers, part of a series.
y = pd.Series(iris.target, name="species_id")
class_names = iris.target_names
feature_names = iris.feature_names

print("Features:", feature_names)
print("Classes:", list(class_names))
pd.concat([X, y], axis=1).head()


# Split into train and testing sets using train_testsplit from sklearn
# X is the data, y is the result, random_state is to always use the same split.
# good to keep the same state for exact results everytime.
# stratify is to keep the split across different outcomes.
# in this case, 0.25 will be evenly split among all features.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
X_train.shape, X_test.shape

# step 3:
# models will be kept in a dictionary.
# Easy to access and will contain all data once called upon.
# 2 different models: log regression which is linear and random forrest, nonlinear.

models = {}

# 3a) Logistic Regression in a Pipeline so scaling happens inside CV/inference too.
models["LogisticRegression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, multi_class="multinomial", random_state=42))
])

# 3b) Random Forest (tree-based)
models["RandomForest"] = RandomForestClassifier(
    n_estimators=300, random_state=42
)

# 3c) XGBoost (optional)
if have_xgb:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        max_depth=4,
        objective="multi:softprob",
        num_class=len(class_names),
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=-1,
        tree_method="hist"
    )

list(models.keys())



