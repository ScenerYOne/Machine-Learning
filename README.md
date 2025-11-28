# 🥗 AppTestModel - Hydroponic Vegetable Detection

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-Ultralytics-blueviolet)

A web-based tool for testing YOLO models specifically designed for **hydroponic vegetable** images.
**AppTestModel** allows users to upload a custom `.pt` model file and test images to instantly visualize detection results with bounding boxes and class names directly in the browser.

## ✨ Features

- **Model Upload:** Upload your custom trained YOLO (`.pt`) model via the web interface.
- **Instant Inference:** Upload an image and get detection results immediately.
- **Visual Feedback:** Displays the processed image with drawn bounding boxes, confidence scores, and class labels.
- **FastAPI Backend:** High-performance backend for efficient model handling and image processing.
- **React Frontend:** Clean, responsive, and user-friendly interface.

## 📂 Project Structure

```text
AppTestModel/
├── uploaded_models/      # Storage for uploaded .pt models
├── server.py             # Backend entry point (FastAPI + Ultralytics)
├── app_testmodel/        # Frontend source code (React + Vite)
│   ├── src/
│   │   ├── App.jsx       # Main application logic
│   │   └── App.css       # Styling
│   ├── vite.config.js
│   └── package.json
└── README.md
