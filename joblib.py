import joblib

model = joblib.load("student_prediction")

prediction = model.predict([[8,89,1,7,6]])

print(prediction)
