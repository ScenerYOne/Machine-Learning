import uvicorn
from fastapi import FastAPI, Body
from joblib import load
import numpy as np

app = FastAPI(title="Wine Quality Prediction API",
              description="API for predicting the quality/type of wine",
              version="1.0")

# เริ่มต้น FastAPI
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)


# Endpoint เริ่มต้น
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Wine Quality Prediction API!"}


# Endpoint สำหรับการทำนาย
@app.post('/prediction', tags=["predictions"])
async def get_prediction(Malic_acid: float, Ash: float, Alcalinity_of_ash: float, Magnesium: float,
                         Total_phenols: float, Flavanoids: float, Nonflavanoid_phenols: float,
                         Proanthocyanins: float, Color_intensity: float, OD280: float, OD31: float,
                         Proline: float):
    # ข้อมูลฟีเจอร์ที่ได้รับจากผู้ใช้
    conditions = [Malic_acid, Ash, Alcalinity_of_ash, Magnesium, Total_phenols, Flavanoids, Nonflavanoid_phenols,
                  Proanthocyanins, Color_intensity, OD280, OD31, Proline]

    # พิมพ์ข้อมูลที่ส่งเข้ามาเพื่อทำนาย
    print(f"Data for prediction: {conditions}")

    # โหลดโมเดลที่ดีที่สุด
    model = load('D:/Machine Learning/Machine-Learning-Course-main/ML-Based Wine Type/Machine-Learning-Course-main/ML-Based Wine Type/models/best_model.pkl')

    # โหลดคอลัมน์จากไฟล์ columns.pkl
    columns = load('D:/Machine Learning/Machine-Learning-Course-main/ML-Based Wine Type/Machine-Learning-Course-main/ML-Based Wine Type/models/columns.pkl')

    # ตรวจสอบว่าจำนวนฟีเจอร์ที่ส่งมาและจำนวนคอลัมน์ตรงกันหรือไม่
    if len(conditions) != len(columns):
        return {"error": "Number of features provided does not match the expected number."}

    # map the conditions to the model columns
    data = []
    for i in range(len(columns)):
        print(columns[i], conditions[i])  # ตรวจสอบลำดับข้อมูล
        data.append(conditions[i])

    # ทำนายผลลัพธ์
    prediction = model.predict([conditions]).tolist()

    # แสดงผลลัพธ์เฉพาะปริมาณแอลกอฮอล์ที่ทำนายได้
    result = f"ไวน์นี้มีแอลกอฮอล์ชนิดที่ {prediction[0]}"
    print(result)
    return {"prediction": result}








