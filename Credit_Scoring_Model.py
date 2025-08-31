import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

csv_path = "UCI_Credit_Card.csv"
df = pd.read_csv(csv_path)
df.columns = [c.strip().lower().replace(".", "_") for c in df.columns]
if "id" in df.columns:
    df = df.drop(columns=["id"])
target_candidates = ["default_payment_next_month", "default_payment_next_month".replace("_", "."), "default.payment.next.month"]
target_col = None
for cand in target_candidates:
    if cand in df.columns:
        target_col = cand
        break
if target_col is None:
    raise ValueError("Target column not found. Expected default.payment.next.month or default_payment_next_month")

pay_cols = [c for c in df.columns if c.startswith("pay_")]
for c in pay_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["late_payments_6m"] = (df[pay_cols] >= 1).sum(axis=1)
if "pay_0" in df.columns:
    df["recent_delinquency_severity"] = df["pay_0"].clip(lower=0)
else:
    df["recent_delinquency_severity"] = df[pay_cols].clip(lower=0).max(axis=1)
bill_cols = [c for c in df.columns if c.startswith("bill_amt")]
payamt_cols = [c for c in df.columns if c.startswith("pay_amt")]
if "limit_bal" in df.columns:
    for c in bill_cols:
        u = df[c] / (df["limit_bal"].replace(0, np.nan))
        df[f"{c}_util"] = u.fillna(0).clip(lower=0)
    util_cols = [f"{c}_util" for c in bill_cols]
    df["avg_util_6m"] = df[util_cols].mean(axis=1)
    df["max_util_6m"] = df[util_cols].max(axis=1)
df["avg_bill_6m"] = df[bill_cols].mean(axis=1)
df["sum_pay_6m"] = df[payamt_cols].sum(axis=1)
sum_pos_bills = df[bill_cols].clip(lower=0).sum(axis=1).replace(0, np.nan)
df["pay_ratio_6m"] = (df["sum_pay_6m"] / sum_pos_bills).fillna(0).clip(0, 5)
cat_cols = [c for c in ["sex", "education", "marriage"] if c in df.columns]

y = df[target_col].astype(int)
base_features = []
for c in ["limit_bal", "age", "avg_util_6m", "max_util_6m", "late_payments_6m", "recent_delinquency_severity", "avg_bill_6m", "sum_pay_6m", "pay_ratio_6m"]:
    if c in df.columns:
        base_features.append(c)
base_features += pay_cols + bill_cols + payamt_cols + cat_cols
base_features = list(dict.fromkeys([c for c in base_features if c in df.columns]))
X = df[base_features].copy()

if cat_cols:
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(df[cat_cols])
    X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names_out(cat_cols), index=X.index)
    X = pd.concat([X.drop(columns=cat_cols, errors="ignore"), X_cat], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
num_cols = X.columns
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[num_cols])
X_test_scaled = scaler.transform(X_test[num_cols])

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}
fitted = {}
for name, clf in models.items():
    if name == "LogisticRegression":
        clf.fit(X_train_scaled, y_train)
    else:
        clf.fit(X_train, y_train)
    fitted[name] = clf

def evaluate(clf, name):
    X_te = X_test_scaled if name == "LogisticRegression" else X_test
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }, y_pred, y_proba

results = {}
for name, clf in fitted.items():
    metrics, y_pred, y_proba = evaluate(clf, name)
    results[name] = {"metrics": metrics, "y_pred": y_pred, "y_proba": y_proba}

for name, res in results.items():
    m = res["metrics"]
    print(f"{name:18s} | Acc {m['Accuracy']:.3f}  Prec {m['Precision']:.3f}  Rec {m['Recall']:.3f}  F1 {m['F1']:.3f}  ROC-AUC {m['ROC_AUC']:.3f}")

plt.figure(figsize=(7,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    auc = roc_auc_score(y_test, res["y_proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"k--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

best_model = max(results.items(), key=lambda kv: kv[9]["metrics"]["ROC_AUC"])
print(f"\nBest model by ROC-AUC: {best_model}")
name = best_model
clf = fitted[name]
X_te = X_test_scaled if name == "LogisticRegression" else X_test
print("\nClassification report:")
print(classification_report(y_test, clf.predict(X_te)))
