import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Data
df = pd.read_csv("student_data.csv")
df.head()

# Shape of dataset
print(f"The Rows are {df.shape[0]} and the Columns are {df.shape[1]}")

#Display the Data Information

print("Dataset Information:")
df.info()

#Describe the dataset
print("Describe the All the Datasets.")
df.describe(include='all')

#Fined the Null values
print("Find the Missing Values")
df.isnull().sum()

df.info()

#Fillup the Missing Values for Gender

df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

#Fillup the missing Values for the Remaining Columns.
l1 = ['age','previous_grades','attendance_percentage','study_hours_per_week','assignments_submitted','exam_scores']

for i in l1:
    df[i] = df[i].fillna(df[i].mean())

#Check the missing Values is Fillup or not

print("Check any missing Values are there or not.")
df.isnull().sum()

#Distribution of Exam Scores with Scaled Data

plt.figure(figsize=(15,8))
plt.hist(df['exam_scores'],bins=20,color="red",edgecolor='black',label="Distribution of Exam Scores")
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score")
plt.ylabel("Number of Student")
plt.grid(True)
plt.savefig("Distribution_of_Exam_Scores",dpi=400,bbox_inches='tight')
plt.show()

#Average Exam Scores vs Age (Pass vs Fail)

plt.figure(figsize=(15,8))

pass_data = df[df['student_success']==1].groupby('age')['exam_scores'].mean()
fail_data = df[df['student_success']==0].groupby('age')['exam_scores'].mean()

plt.plot(pass_data.index,pass_data.values,marker='o',color='green',label='Pass')
plt.plot(fail_data.index,fail_data.values,marker='o',color='red',label='Fail')
plt.title("Average Exam Scores vs Age (Pass vs Fail)")
plt.xlabel("Age")
plt.ylabel("Average Exam Score")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig("Average_Exam_Scores_vs_Age_(Pass vs Fail).png",dpi=400,bbox_inches='tight')
plt.show()

#Lets Convert our Object Columns into the Numaric
#Using The LabelEncoder
print("We are Encode the Gender Columns using the Label Encoding Technique.")
encoder = LabelEncoder()

df['gender'] = encoder.fit_transform(df['gender'])
df['student_success'] = encoder.fit_transform(df['student_success'])  #Pass=1, Fail=0

print("Lets see our encoded Data..... ")
df.head()

print("Let see the Datatypes of the all Data Columns.")
df.info()

#Lets do the FetureScaling for the Numaric Columns

features = ['age', 'gender', 'previous_grades', 'attendance_percentage', 'study_hours_per_week', 'assignments_submitted', 'exam_scores']

scaler = StandardScaler()

df_scaled = df.copy()

df_scaled[features] = scaler.fit_transform(df_scaled[features])

#Lets see the data after Scaling
print("Data After Scaling:")
print(df_scaled.head())

#Lets train the model

X = df_scaled[features]   #This is our Feture Column
y = df_scaled['student_success']  #This is our Target Column

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(Y_test, y_pred))

#Lets Find the Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("The Confusion Matrix is:")
print(conf_matrix)

#Lets Draw the Charts for Confusion Matrix

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("Confusion_Matrix.png", dpi=800, bbox_inches='tight')
plt.show()

#User Input for the Prediction

print("-----------------Lets Predict The Result For Student-----------------")

try:
    age = float(input("Enter the Age of the Student: "))
    gender = float(input("Enter the Gender of the Student (Female=0, Male=1): "))
    previous_grades = float(input("Enter the previous grades of the Student: "))
    attendance_percentage = float(input("Enter the attendance percentage of the Student: "))
    study_hours_per_week = float(input("Enter the study hours per week of the Student: "))
    assignments_submitted = float(input("Enter the assignments submitted marks of the Student (out of 20): "))
    exam_scores = float(input("Enter the Last exam scores of the Student: "))

    # Ensure the same column order as training
    user_input_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'previous_grades': previous_grades,
        'attendance_percentage': attendance_percentage,
        'study_hours_per_week': study_hours_per_week,
        'assignments_submitted': assignments_submitted,
        'exam_scores': exam_scores
    }])[features]

    # Scale user input
    user_input_scaled = scaler.transform(user_input_df)

    # Predict
    prediction = model.predict(user_input_scaled)[0]
    result = "Pass" if prediction == 1 else "Fail"

    print(f"Prediction Based on Input = {result}")

except Exception as e:
    print("An error occurred:", e)