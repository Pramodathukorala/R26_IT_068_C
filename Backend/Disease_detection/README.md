# Poultry Disease AI Backend

## Project Overview

This project is an AI-assisted poultry disease screening backend developed using FastAPI and TensorFlow/Keras.

The system accepts poultry images and performs AI-based disease screening using two specialized image streams:

- FECAL image analysis
- BODY image analysis

An additional AI input validation model verifies whether the uploaded image is a valid BODY image, FECAL image, or an unsupported OTHER image before disease prediction is performed.

---

## Supported Disease Classes

### FECAL Model

The fecal image model supports:

- Healthy
- Coccidiosis
- Newcastle

### BODY Model

The body image model supports:

- Healthy
- Fowlpox

---

## AI Models

The backend uses three trained AI models:

1. Fecal Disease Classification Model
2. Body Disease Classification Model
3. Input Type Validation Model

The input validator classifies uploaded images into:

- BODY
- FECAL
- OTHER

This prevents unrelated images and incorrect image types from being passed directly to the disease classification models.

---

## System Workflow

The AI processing pipeline is:

```text
Image Upload
    ↓
Image Preprocessing
    ↓
Image Quality Check
    ↓
Input Type Validation
    ↓
BODY / FECAL / OTHER
    ↓
Correct Disease Model
    ↓
Disease Prediction
    ↓
Confidence Threshold Check
    ↓
Risk Level
    ↓
Recommendation
    ↓
Grad-CAM Heatmap
    ↓
API Response