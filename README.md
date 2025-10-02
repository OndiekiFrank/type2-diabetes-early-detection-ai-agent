# TYPE2-DIABETES-EARLY-DETECTION-AI-AGENT

![Profile Photo](images/Profile%20Photo.jpg)

## Background

Type 2 Diabetes is a growing global health concern that can lead to severe complications if undiagnosed or untreated.  
Early prediction enables:

- **Timely medical interventions**
- **Lifestyle adjustments**
- **Cost reduction for healthcare systems**

This project leverages real-world health survey data to **predict diabetes risk** and identify key factors that drive the disease.

---

## Project Objectives

1. **Analyze** lifestyle and health-related features to uncover risk patterns.
2. **Build** machine learning models to classify individuals as diabetic or non-diabetic.
3. **Evaluate** models for accuracy, interpretability, and real-world applicability.
4. **Recommend** preventive measures based on findings.

---

## Repository Structure

type2-diabetes-ai-agent/
│
├─ 📁 data/
│   ├─ raw/                     # (Optional) Original/raw datasets
│   └─ processed/                # Cleaned or feature-engineered datasets
│
├─ 📁 notebooks/
│   └─ type2_diabetes_ai_agent.ipynb   # Main Jupyter Notebook
│
├─ 📁 images/
│   ├─ profile_photo.jpg               # Main repo image (first impression)
│   ├─ class_distribution.png
│   ├─ correlation_heatmap_numerical.png
│   ├─ pair_plot_key_features_diabetes_outcome.png
│   ├─ box_plots.png
│   ├─ confusion_matrix_final_test.png
│
├─ 📁 src/
│   ├─ __init__.py
│   ├─ data_preprocessing.py          
│   ├─ model_training.py               
│   └─ utils.py                        
│
├─ .gitignore
├─ README.md                           
└─ LICENSE                              

---

## Data Understanding

- **Source:** Public health survey dataset (cleaned for analysis).
- **Target Variable:** Presence of Type 2 Diabetes (binary classification).
- **Features:** Demographics (age, gender), lifestyle (physical activity, smoking), and clinical metrics (BMI, cholesterol check, blood pressure).

**Key Data Insights:**

- Imbalanced target distribution: Fewer positive diabetes cases.
- Strong correlations between **BMI**, **Age**, **Cholesterol Check**, and diabetes status.
- Lifestyle habits like **physical inactivity** and **smoking** show moderate association.

![Class Distribution](images/Distribution%20of%20Diabetes%20class.png)


---

## Methodology

| Stage | Approach |
|------|----------|
| **Data Cleaning** | Handling missing values, encoding categorical variables, standardizing numerical features. |
| **EDA** | Correlation heatmaps, boxplots, and hypothesis testing (ANOVA, chi-square). |
| **Balancing** | **SMOTE** applied to address class imbalance. |
| **Modeling** | Logistic Regression, Random Forest, XGBoost, KNN, Deep Neural Network (Keras/TensorFlow). |
| **Evaluation** | Accuracy, F1-score, ROC-AUC, confusion matrices. |
| **Tuning** | Hyperparameter optimization (GridSearchCV, Keras Tuner). |

### Correlation Heatmap of Numerical Features

Shows relationships among continuous variables and highlights strong predictors.

![Correlation Heatmap of Numerical Features](images/Correlation%20Heatmap%20of%20Numerical%20Features.png)


### Pair Plot of Key Features by Diabetes Outcome

Visualizes pairwise relationships among the top predictive features, colored by diabetes outcome.

![Pair Plot of Key Features by Diabetes Outcome](images/Pair%20Plot%20of%20Key%20Features%20by%20Diabetes%20Outcome.png)

---

## Machine Learning Models

| Model                   | Key Strengths | Performance Summary |
|--------------------------|---------------|----------------------|
| Logistic Regression      | Interpretability | Accuracy 0.83, F1 0.81 |
| Random Forest            | Handles non-linear relationships | Accuracy 0.87, F1 0.84 |
| **XGBoost** (Best)       | High predictive power | **Accuracy 0.89, F1 0.87** |
| KNN                      | Simple, non-parametric | Accuracy 0.79, F1 0.77 |
| Deep Neural Network      | Captures complex patterns | Accuracy 0.85, F1 0.82 |

### Box Plots of Key Numerical Features

Box plots reveal the distribution, spread, and potential outliers for key numerical variables.

![Box Plots](images/box%20plots.png)

---

## Key Findings

- **BMI, Age, and Cholesterol Check** are the strongest predictors of Type 2 Diabetes.
- **Physical activity** and **smoking habits** are important lifestyle factors.
- Addressing class imbalance (SMOTE) improved model performance across all algorithms.
- **XGBoost** emerged as the most accurate and stable model.

### Confusion Matrix – Final Test

The confusion matrix summarizes the model’s classification performance,  
highlighting true positives, false positives, true negatives, and false negatives.

![Confusion Matrix – Final Test](images/Confusion%20Matrix%20-%20Final%20Test.png)

---

## Insights

 -**Non-Diabetic**: Most individuals are in the healthy range, but lifestyle habits
 determine long-term protection.
 -**Pre-Diabetic**: This group is at a critical turning point — small lifestyle
 changes can prevent progression to diabetes.
 -**Diabetic**: Require active management of diet, exercise, and medication to
 avoid complications.

## Recommendations

1. **Public Health Campaigns** 
 - Encourage regular **cholesterol checks** and annual diabetes screening for high-risk age groups.
   - Promote **physical activity programs** to help maintain healthy BMI.

2. **Lifestyle Interventions** 
   - Educate communities on **smoking cessation** and its link to diabetes.

3. **Clinical Integration** 
   - Deploy the XGBoost model as a decision-support tool in primary care for early detection.

## Conclusion

 1. **Non-Diabetic** → focus on prevention and maintaining healthy habits.
 2. **Pre-Diabetic** → take urgent lifestyle actions to reverse/prevent diabetes.
 3. **Diabetic** → combine lifestyle changes with medical management to control
 the condition and prevent complications. 

