from fastapi import FastAPI, Query
import pandas as pd
import joblib

app = FastAPI()

# โหลดโมเดลที่ฝึกแล้ว
model = joblib.load("best.pkl")
imputer = joblib.load("imputer.pkl")
columns = joblib.load("columns.pkl")

# สร้าง endpoint สำหรับทำนาย
@app.get("/predict")
def predict_passenger(
    Operating_Airline: str = Query(..., description="ชื่อสายการบิน"),
    GEO_Region: str = Query(..., description="ภูมิภาค"),
    Price_Category_Code: str = Query(..., description="หมวดหมู่ราคา"),
    Year: int = Query(..., description="ปี"),
    Month: int = Query(..., description="เดือน")
):
    input_data = pd.DataFrame([{
        'Operating Airline': Operating_Airline,
        'GEO Region': GEO_Region,
        'Price Category Code': Price_Category_Code,
        'Year': Year,
        'Month': Month
    }])

    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    missing_cols = list(set(columns) - set(input_data_encoded.columns))

    if missing_cols:
        missing_cols_df = pd.DataFrame(0, index=input_data_encoded.index, columns=missing_cols)
        input_data_encoded = pd.concat([input_data_encoded, missing_cols_df], axis=1)

    input_data_encoded = input_data_encoded[columns]
    input_data_encoded = imputer.transform(input_data_encoded)

    prediction = model.predict(input_data_encoded)

    return {"prediction จำนวนผู้โดยสารเฉลี่ยต่อเดือน": prediction[0]}
