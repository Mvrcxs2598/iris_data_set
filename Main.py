
# numpy is used for numerical computing functions.
# pandas is used to crete dataframes to obtain spreadsheet and sql data files. Built ontop of Numpy.
# SKlearn is a machine elarning toolkit that has ready to use algorithms.
    # standard library in python.

"""
Goal of this python script is to visaulize two different types of ML models on the same data.
Using logistic regression and Randomforrest.

"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 #
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


# train_test_split is used to create 4 sub groups of data.
# 2 are the features and 2 are the results split into training and testing
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
# pipeline is used like: raw data → scaler → model → prediction
# Important to use pipeline since it keeps training/testing sets apart from each other.
# Avoids data leakage and keeps the preprocessing always the same.

# LogR uses a formula similar to that below:
# w1 * x1 + w2 * x2 + b = 0     → the decision boundary (line)
# Learns weights

models["LogisticRegression"] = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, multi_class="multinomial", random_state=42))
])

#Random Forest (tree-base) Uses a formula by creating multiple(up to thousands) of trees that are split by a single feature.
# This focuses on feature importance.
# Learns thresholds and tree structure
models["RandomForest"] = RandomForestClassifier(
    n_estimators=300, random_state=42
)


# 2 models created above, now to create accuracies and predictions:
accuracies = {}
pred_scores = {}

for name, model in models.items():
    # model.fit is creating a model based on the selected model on the training data.
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    # now to store the results in a dictionary to compare later.
    pred_scores[name] = y_pred
    accuracies[name] = accuracy_score(y_test,y_pred)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracies[name]:.4f}")
    # can uncomment the line to get precision/recall/F1 per class, but extra info.
    # print(classification_report(y_test, y_pred, target_names=class_names))


# Step 3b:
# since we are storing everything in dictionaries, using .items() to get list of keys.
# now to dive deeper into how often each model guesses the wrong classification.
for name, y_pred in pred_scores.items():
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names
    )
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()



# Step 4a:
# Collecting important features from POV of each model type.
importances_dict = {}

if "LogisticRegression" in models:
    lr_pipeline = models["LogisticRegression"]
    lr_clf = lr_pipeline.named_steps["clf"]
    lr_coefs = np.abs(lr_clf.coef_).mean(axis=0)  # average across classes
    importances_dict["LogisticRegression"] = lr_coefs

# For Random Forest: built-in feature_importances_
if "RandomForest" in models:
    rf = models["RandomForest"]
    importances_dict["RandomForest"] = rf.feature_importances_

print(importances_dict)



# step 4b:
# We have the important features now from obtaining them from the Coeffs.
# plot the coeffs based on the features in a visual way.
def plot_importances(importances_dict, feature_names):
    model_names = list(importances_dict.keys())
    n_features = len(feature_names)
    x = np.arange(n_features)
    width = 0.8 / max(1, len(model_names))  # bar width per model

    plt.figure(figsize=(10, 5))
    for i, model_name in enumerate(model_names):
        imp = np.array(importances_dict[model_name], dtype=float)
        s = imp.sum()
        if s > 0:
            imp = imp / s  # normalize so bars are comparable
        plt.bar(x + i * width, imp, width=width, label=model_name, alpha=0.85)

    plt.xticks(x + width * (len(model_names) - 1) / 2, feature_names, rotation=20)
    plt.ylabel("Normalized Importance")
    plt.title("Feature Importances per Model")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_importances(importances_dict, feature_names)


df_plot = X.copy()
df_plot["species"] = y.map(lambda i: class_names[i])

pd.plotting.scatter_matrix(
    df_plot[["sepal_length", "sepal_width", "petal_length", "petal_width"]],
    figsize=(9, 9),
    diagonal='hist'
)
plt.suptitle("Iris Feature Scatter Matrix", y=1.02)
plt.tight_layout()
plt.show()


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7, 5))
for i, cls in enumerate(class_names):
    idx = (y.values == i)
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=cls, alpha=0.8)

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("Iris PCA (2D) by Species")
plt.legend()
plt.tight_layout()
plt.show()
