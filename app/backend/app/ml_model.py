import pickle
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Union
from app.config import settings
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DiabetesPredictionModel:
    def __init__(self):
        self.model = None
        self.model_metadata = {}
        self.feature_names = [
            "pregnancies", "glucose", "blood_pressure", "skin_thickness",
            "insulin", "bmi", "diabetes_pedigree_function", "age"
        ]
        self.model_loaded = False

    def load_model(self, model_path: str) -> bool:
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            if model_path.endswith(".pkl"):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            elif model_path.endswith(".joblib"):
                self.model = joblib.load(model_path)
            else:
                logger.error("Unsupported model format. Use .pkl or .joblib")
                return False

            self.model_metadata = {
                "model_type": type(self.model).__name__,
                "features": self.feature_names,
                "model_path": model_path,
                "loaded_at": np.datetime64("now"),
            }
            self.model_loaded = True
            logger.info(f"Diabetes model loaded successfully from {model_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            self.model_loaded = False
            return False

    def predict(self, features: Union[Dict[str, float], Any]) -> Tuple[str, float, Dict[str, float]]:
        if not self.model_loaded or not self.model:
            raise Exception("Model not loaded. Call load_model() first.")

        try:
            if hasattr(features, "dict"):
                features = features.dict()

            # Compute BMI if not present
            if "bmi" not in features:
                features["bmi"] = round(features["weight"] / (features["height"] ** 2), 1)

            feature_array = np.array([[ 
                features["pregnancies"],
                features["glucose"],
                features["blood_pressure"],
                features["skin_thickness"],
                features["insulin"],
                features["bmi"],
                features["diabetes_pedigree_function"],
                features["age"]
            ]])

            if hasattr(self.model, "predict_proba"):
                probability = self.model.predict_proba(feature_array)[0, 1]
            else:
                probability = float(self.model.predict(feature_array)[0])

            risk_label = "high risk" if probability >= 0.5 else "low risk"
            feature_importances = self._calculate_feature_importance(features, probability)

            return risk_label, probability, feature_importances

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise Exception(f"Prediction failed: {e}")

    def _calculate_feature_importance(self, features: Dict[str, float], probability: float) -> Dict[str, float]:
        try:
            normal_ranges = {
                "pregnancies": 0, "glucose": 100, "blood_pressure": 80,
                "skin_thickness": 20, "insulin": 80, "bmi": 25,
                "diabetes_pedigree_function": 0.5, "age": 35
            }
            importances = {}
            for feature, value in features.items():
                normal_value = normal_ranges.get(feature)
                if normal_value is not None:
                    deviation = abs(value - normal_value) / normal_value if normal_value > 0 else abs(value - normal_value)
                    importances[feature] = min(deviation, 2.0)

            total = sum(importances.values())
            if total > 0:
                importances = {k: v / total for k, v in importances.items()}
            return importances
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {feature: 0.125 for feature in self.feature_names}

    def get_bmi_category(self, bmi: float) -> str:
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 25:
            return "Normal weight"
        elif 25 <= bmi < 30:
            return "Overweight"
        return "Obese"

    def validate_features(self, features: Dict[str, float]) -> bool:
        try:
            for feature in self.feature_names:
                if feature not in features:
                    logger.error(f"Missing feature: {feature}")
                    return False

            if not (0 <= features["pregnancies"] <= 20): return False
            if not (0 <= features["glucose"] <= 200): return False
            if not (0 <= features["blood_pressure"] <= 122): return False
            if not (0 <= features["skin_thickness"] <= 99): return False
            if not (0 <= features["insulin"] <= 846): return False
            if not (0 <= features["bmi"] <= 50): return False
            if not (0.08 <= features["diabetes_pedigree_function"] <= 2.42): return False
            if not (21 <= features["age"] <= 81): return False

            return True
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False


# Global model instance
diabetes_model = DiabetesPredictionModel()

def initialize_model() -> bool:
    try:
        success = diabetes_model.load_model(settings.MODEL_PATH)
        if success:
            logger.info("Diabetes model initialized successfully")
        else:
            logger.error("Diabetes model initialization failed")
        return success
    except Exception as e:
        logger.error(f"Diabetes model initialization error: {e}")
        return False
