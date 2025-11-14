import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_excel("attendance_chart.xlsx")

df["Attendance_Percentage"] = (df["No. of Days Attended"] / df["Total Duration"]) * 100

X = df[["No. of Days Attended"]]
y = df["Attendance_Percentage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

y_pred = reg_model.predict(X_test)

print("R² Score:", round(r2_score(y_test, y_pred), 3))

plt.figure(figsize=(6, 4))
plt.scatter(df["No. of Days Attended"], df["Attendance_Percentage"], color="blue", label="Actual")
plt.plot(df["No. of Days Attended"], reg_model.predict(X), color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression: Attendance Prediction")
plt.xlabel("Days Attended")
plt.ylabel("Attendance Percentage")
plt.legend()
plt.show()
