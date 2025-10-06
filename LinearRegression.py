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


x = df[["Hours Studied", "Previous Scores", 
        "Extracurricular Activities", 
        "Sleep Hours", 
        "Sample Question Papers Practiced"]]
y = df["Performance Index"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train.values,y_train.values)

prediction = model.predict([[8,89,1,7,6]])
print(prediction)
joblib.dump(model,"student_prediction")
