
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PREPROCESSED_CSV = "suitable_job_roles_2000_preprocessed.csv"
CLEANED_XLSX = "suitable_job_roles_2000_cleaned.xlsx"
LABEL_ENCODER_FILE = "label_encoder_jobrole.joblib"

df_clean = pd.read_excel(CLEANED_XLSX)
print("Loaded cleaned human-readable data:", df_clean.shape)

plt.figure(figsize=(10,5))
deg_counts = df_clean['Degree'].value_counts().sort_values(ascending=False)
deg_counts.plot(kind='bar')
plt.title("Highest Qualification Distribution (Degree)")
plt.xlabel("Degree")
plt.ylabel("Count")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("qualification_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(10,5))

if 'Experience_Years' in df_clean.columns:
    exp_counts = df_clean['Experience_Years'].value_counts().sort_index()
    exp_counts.plot(kind='bar')
    plt.title("Years of Experience Distribution")
    plt.xlabel("Years of Experience")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("experience_years_distribution.png", dpi=150)
    plt.show()
else:
    print("Column 'Experience_Years' not found in cleaned Excel.")

df = pd.read_csv(PREPROCESSED_CSV)
print("Loaded preprocessed data for ML:", df.shape)

if "JobRole_Label" not in df.columns:
    raise ValueError("Expected column 'JobRole_Label' in preprocessed CSV")

X = df.drop("JobRole_Label", axis=1)
y = df["JobRole_Label"]

label_encoder = joblib.load(LABEL_ENCODER_FILE)
class_names = list(label_encoder.classes_)
print("Job role classes:", class_names)

plt.figure(figsize=(10,5))
counts = y.value_counts().sort_index()
plt.bar(range(len(counts)), counts.values)
plt.xticks(range(len(counts)), [class_names[i] if i < len(class_names) else str(i) for i in counts.index], rotation=45, ha="right")
plt.title("Job Role Distribution")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("job_role_distribution.png", dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

#Random Forest 
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions & metrics
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nRandom Forest accuracy: {acc:.4f}\n")
print("Classification report:\n")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("confusion_matrix.csv")

feat_importances = rf.feature_importances_
feat_names = X.columns.to_list()

fi_df = pd.DataFrame({
    "feature": feat_names,
    "importance": feat_importances
}).sort_values("importance", ascending=False)

TOP_N = 25
top_fi = fi_df.head(TOP_N)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_fi)), top_fi["importance"].values[::-1])
plt.yticks(range(len(top_fi)), top_fi["feature"].values[::-1])
plt.xlabel("Feature importance")
plt.title(f"Top {TOP_N} Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importances_top25.png", dpi=150)
plt.show()
