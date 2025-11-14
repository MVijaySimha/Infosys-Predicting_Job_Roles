import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_excel("attendance_chart.xlsx")

df["Attendance_Percentage"] = (df["No. of Days Attended"] / df["Total Duration"]) * 100

X = df[["Total Duration", "No. of Days Attended"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("Cluster Distribution:\n", df["Cluster"].value_counts())
print("\nCluster Centers:\n", kmeans.cluster_centers_)
print("\nSample Data with Cluster Labels:\n", df[["Student", "No. of Days Attended", "Attendance_Percentage", "Cluster"]])

plt.figure(figsize=(6, 4))
plt.scatter(df["No. of Days Attended"], df["Attendance_Percentage"], 
            c=df["Cluster"], cmap="viridis", s=100, marker='x')
plt.title("KMeans Clustering on Attendance Data")
plt.xlabel("Days Attended")
plt.ylabel("Attendance Percentage")
plt.colorbar(label="Cluster")
plt.show()
