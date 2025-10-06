import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("Student_Performance.csv")
df = pd.DataFrame(data)
# print(df)

le = LabelEncoder()
encoder = le.fit_transform(df["Extracurricular Activities"])
df["Extracurricular Activities"] = encoder


x = df[["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"]]
y = df["Performance Index"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train.values,y_train.values)
print(df)

a = int(input(" Hours Studied (1-9):"))
b = int(input("Previous Scores(1-100)"))
c = int(input("Extracurricular Activities Yes (1) or No (0) :"))
d = int(input("Sleep Hours (1-8):"))
e = int(input("Sample Question Papers Practiced (1-5): "))



prediction = model.predict([[a,b,c,d,e]])
print("prediction for new data",prediction)

joblib.dump(model,"student_prediction")
