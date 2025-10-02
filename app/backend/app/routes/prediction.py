@router.post("/predict", response_model=dict)
async def predict_diabetes(request: DiabetesInput):
    try:
        patient_data = request.dict()

        # 1️⃣ ML Model Prediction
        ml_output, bmi_category = diabetes_model.predict(patient_data)

        # 2️⃣ LLM Interpretation
        llm_response = insulyn_llm.generate_advice(
            patient_data, ml_output, bmi_category
        )

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "ml_output": ml_output,
            "llm_advice": llm_response,
            "bmi_category": bmi_category,
        }

    except Exception as e:
        import traceback
        logger.error(f"Prediction endpoint error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Error generating prediction")
