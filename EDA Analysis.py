import pandas as pd
import numpy as np

students = [
    'Danne', 'Bob', 'Charlie', 'David', 'priya', 'prasad', 'George', 'pandu', 'Ivan', 'Jane',
    'Kevin', 'Laura', 'Mike', 'Dhatri', 'Rose']
total_days = 40

attendance_probabilities = {
    'Danne': 0.85,
    'Bob': 0.54,
    'Charlie': 0.76,
    'David': 0.31,
    'priya': 0.87,
    'prasad': 0.89,
    'George': 0.82,
    'pandu': 0.70,
    'Ivan': 0.85,
    'Jane': 0.80,
    'Kevin': 0.60,
    'Laura': 0.78,
    'Mike': 0.82,
    'Dhatri': 0.91,
    'Rose': 0.62,
}
np.random.seed(42)
attendance_records = []

for student in students:
    present_prob = attendance_probabilities.get(student, 0.85)
    for day in range(1, total_days + 1):
        status = np.random.choice(['Present', 'Absent'], p=[present_prob, 1 - present_prob])
        attendance_records.append({
            'Student': student,
            'Day': day,
            'Status': status
        })
df_attendance = pd.DataFrame(attendance_records)
summary = df_attendance.groupby('Student')['Status'].value_counts().unstack(fill_value=0)
summary['Total_Days'] = total_days
summary['Attendance_%'] = (summary['Present'] / total_days) * 100
summary['Attendance_%'] = summary['Attendance_%'].round(2)
summary = summary.reset_index()

print("📊 Attendance Summary (40 Days):")
print(summary)