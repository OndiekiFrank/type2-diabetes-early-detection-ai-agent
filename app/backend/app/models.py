from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ===============================
# Diabetes Prediction Models
# ===============================
class DiabetesInput(BaseModel):
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: float = Field(..., ge=0, le=200, description="Glucose level in mg/dL")
    blood_pressure: float = Field(..., ge=0, le=122, description="Blood pressure in mmHg")
    skin_thickness: float = Field(..., ge=0, le=99, description="Skin thickness in mm")
    insulin: float = Field(..., ge=0, le=846, description="Insulin level in mu U/ml")
    weight: float = Field(..., ge=20, le=200, description="Weight in kg")
    height: float = Field(..., ge=0.5, le=2.5, description="Height in meters")
    diabetes_pedigree_function: float = Field(..., ge=0.08, le=2.42, description="Diabetes pedigree function")
    age: int = Field(..., ge=21, le=81, description="Age in years")

    @validator("height")
    def validate_height(cls, v):
        if v <= 0:
            raise ValueError("Height must be positive")
        return v

    @property
    def bmi(self) -> float:
        """Calculate BMI from weight and height"""
        return round(self.weight / (self.height ** 2), 1)


class MLModelOutput(BaseModel):
    risk_label: str = Field(..., description='"high risk" or "low risk"')
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of diabetes")
    feature_importances: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores from the ML model"
    )
    calculated_bmi: float = Field(..., description="BMI calculated from weight and height")


# ===============================
# LLM Advice (Groq/OpenAI Response)
# ===============================
class LLMAdviceResponse(BaseModel):
    risk_summary: str
    clinical_interpretation: List[str]
    recommendations: Dict[str, str]
    prevention_tips: List[str]
    monitoring_plan: List[str]
    clinician_message: str
    feature_explanation: str
    safety_note: str

    @validator("recommendations", pre=True)
    def validate_recommendations(cls, v):
        """Ensure all recommendation values are strings, converting lists if needed"""
        if isinstance(v, dict):
            for key, value in v.items():
                if isinstance(value, list):
                    # Convert list to bullet-point string
                    v[key] = "\n".join([f"• {item}" if not item.startswith("•") else item for item in value])
                elif not isinstance(value, str):
                    v[key] = str(value)
        return v


class CombinedResponse(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ml_output: MLModelOutput
    llm_advice: LLMAdviceResponse
    bmi_category: str = Field(..., description="Underweight, Normal, Overweight, or Obese")


# ===============================
# Chatbot Models - FIXED
# ===============================
class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User's message")
    conversation_context: Optional[str] = Field(
        None, description="Optional context about previous conversation"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    suggestions: List[str] = Field(default_factory=list, description="Optional suggestions for next steps")


# ===============================
# Diet Planning Models
# ===============================
class DietPlanRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, description="User's age")
    weight: float = Field(..., ge=30, le=200, description="Weight in kg")
    height: float = Field(..., ge=1.0, le=2.5, description="Height in meters")
    dietary_preferences: Optional[str] = Field(None, description="e.g., vegetarian, low-carb, high-protein")
    health_conditions: Optional[str] = Field(None, description="e.g., hypertension, high cholesterol")
    diabetes_risk: Optional[str] = Field(None, description="low/medium/high risk")


class DietPlanResponse(BaseModel):
    daily_calorie_target: int
    meal_plan: Dict[str, List[str]]
    foods_to_include: List[str]
    foods_to_avoid: List[str]
    portion_guidance: List[str]
    timing_recommendations: List[str]


# ===============================
# Health Metrics Model
# ===============================
class HealthMetrics(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    services: Dict[str, bool]
    metrics: Dict[str, Any]


# ===============================
# Error Response Models
# ===============================
class ValidationError(BaseModel):
    loc: List[str]
    msg: str
    type: str


class HTTPValidationError(BaseModel):
    detail: List[ValidationError]


# ===============================
# Error Response for Chat
# ===============================
class ErrorResponse(BaseModel):
    detail: str