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


# ฟังก์ชันสำหรับประเมินความเหมาะสมของค่า pH
def evaluate_ph(ph: float, wine_type: str):
    if wine_type.lower() == 'white wine':
        if 3.0 <= ph <= 3.3:
            return "เหมาะสม"
        elif ph < 3.0:
            return "ต่ำเกินไป อาจมีรสเปรี้ยวมากเกินไป"
        else:
            return "สูงเกินไป เสี่ยงต่อการเสื่อมสภาพเร็วขึ้น"
    elif wine_type.lower() == 'red wine':
        if 3.3 <= ph <= 3.8:  # ขยายช่วง pH ของไวน์แดงให้ครอบคลุมถึง 3.8
            return "เหมาะสม"
        elif ph < 3.3:
            return "ต่ำเกินไป อาจมีรสเปรี้ยวมากเกินไป"
        else:
            return "สูงเกินไป เสี่ยงต่อการเสื่อมสภาพเร็วขึ้น"
    else:
        return "ไม่สามารถประเมินได้ เนื่องจากประเภทไวน์ไม่ถูกต้อง"


# ฟังก์ชันสำหรับแนะนำประเภทไวน์ที่เหมาะสมกับค่า pH
def suggest_correct_wine_type(ph: float):
    if 3.0 <= ph <= 3.3:
        return 'White Wine'
    elif 3.3 <= ph <= 3.8:  # ขยายช่วง pH ของไวน์แดงให้ครอบคลุมถึง 3.8
        return 'Red Wine'
    else:
        return 'Unknown'


# สร้าง API ที่รับค่าผ่าน Query Parameters
@app.get("/predict_quality/")
async def predict_quality(pH: float, wine_type: str):
    # แปลง wine_type เป็นตัวเลข
    wine_type_encoded = encode_wine_type(wine_type)

    # ตรวจสอบว่าประเภทของไวน์ถูกต้องหรือไม่
    if wine_type_encoded == -1:
        return {"error": "Invalid wine type. Please choose either 'White Wine' or 'Red Wine'."}

    # ประเมินความเหมาะสมของค่า pH กับประเภทไวน์ที่ระบุ
    ph_evaluation = evaluate_ph(pH, wine_type)

    # แนะนำประเภทไวน์ที่เหมาะสมกับค่า pH
    correct_wine_type = suggest_correct_wine_type(pH)

    # หากค่า pH ไม่ตรงกับชนิดของไวน์ที่ระบุ ให้แสดงผลว่าควรเป็นไวน์ประเภทใด
    if ph_evaluation != "เหมาะสม" and correct_wine_type != wine_type:
        return {
            "pH": pH,
            "wine_type": wine_type,
            "error": f"ค่า pH {pH} ไม่เหมาะสมกับ {wine_type}, ควรเป็น {correct_wine_type}",
            "suggested_wine_type": correct_wine_type
        }

    # กรณีที่ pH และชนิดไวน์ตรงกัน
    return {
        "pH": pH,
        "wine_type": wine_type,
        "ph_evaluation": "เหมาะสม"
    }
