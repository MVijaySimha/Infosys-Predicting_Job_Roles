
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_excel("attendance_chart.xlsx")

df["Attendance_Percentage"] = (df["No. of Days Attended"] / df["Total Duration"]) * 100
df["Attendance_Category"] = df["Attendance_Percentage"].apply(lambda x: 1 if x >= 75 else 0)  # 1=High, 0=Low

X = df[["Total Duration", "No. of Days Attended"]]
y = df["Attendance_Category"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Accuracy:", round(accuracy_score(y_test, y_pred_log), 3))

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
print("Decision Tree Accuracy:", round(accuracy_score(y_test, y_pred_tree), 3))

plt.figure(figsize=(6, 4))
plt.scatter(df["No. of Days Attended"], df["Attendance_Percentage"], 
            c=df["Attendance_Category"], cmap="coolwarm", s=100, marker='x')
plt.title("Attendance Classification (High vs Low)")
plt.xlabel("Days Attended")
plt.ylabel("Attendance Percentage")
plt.colorbar(label="High(1) / Low(0) Attendance")
plt.show()
