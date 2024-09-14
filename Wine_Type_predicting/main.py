from fastapi import FastAPI
import joblib
import numpy as np

# โหลดโมเดลที่ฝึกไว้
model = joblib.load('wine_quality_model.pkl')

# สร้างแอปพลิเคชัน FastAPI
app = FastAPI()

# ฟังก์ชันสำหรับแปลงประเภทไวน์เป็นตัวเลข
def encode_wine_type(wine_type: str):
    if wine_type.lower() == 'white wine':
        return 0
    elif wine_type.lower() == 'red wine':
        return 1
    else:
        return -1

# สร้าง API ที่รับค่าผ่าน Query Parameters
@app.get("/predict_quality/")
async def predict_quality(pH: float, wine_type: str):
    # แปลง wine_type เป็นตัวเลข
    wine_type_encoded = encode_wine_type(wine_type)

    # ตรวจสอบว่าประเภทของไวน์ถูกต้องหรือไม่
    if wine_type_encoded == -1:
        return {"error": "Invalid wine type. Please choose either 'White Wine' or 'Red Wine'."}

    # สร้าง input สำหรับโมเดล
    input_features = np.array([[pH, wine_type_encoded]])

    # ทำการทำนายคุณภาพไวน์
    predicted_quality = model.predict(input_features)

    # ส่งผลลัพธ์การทำนายกลับไป
    return {
        "pH": pH,
        "wine_type": wine_type,
        "predicted_quality": int(predicted_quality[0])
    }
