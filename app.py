import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Dog Health Predictor", page_icon="dog", layout="wide")

TOP_FEATURES = ['Age', 'Weight (lbs)', 'Daily Activity Level', 'Diet', 'Medications', 'Seizures', 'Annual Vet Visits', 'Spay/Neuter Status']

@st.cache_resource
def train_model():
    df = pd.read_csv('synthetic_dog_breed_health_data.csv')
    df.replace('', np.nan, inplace=True)
    df = df.drop(columns=['ID', 'Synthetic'], errors='ignore')

    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df['Healthy'] = df['Healthy'].map({'Yes': 1, 'No': 0})

    X = df[TOP_FEATURES]
    y = df['Healthy']

    numeric_features = ['Age', 'Weight (lbs)', 'Annual Vet Visits']
    ordinal_features = ['Daily Activity Level']
    categorical_features = ['Diet', 'Medications', 'Seizures', 'Spay/Neuter Status']

    daily_activity_categories = ['Low', 'Moderate', 'Active', 'Very Active']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
        ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(categories=[daily_activity_categories]))]), ordinal_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(X_processed, y)

    return model, preprocessor, df

model, preprocessor, df = train_model()

st.title("Dog Health Predictor")
st.write("Fill in the parameters to predict health status.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Возраст", 0, 20, 5)
    weight = st.slider("Вес (lbs)", 0, 150, 30)
    annual_vet = st.slider("Посещения ветеринара", 0.0, 10.0, 1.0)
    daily_activity = st.selectbox("Уровень активности", ['Low', 'Moderate', 'Active', 'Very Active'])

with col2:
    diet = st.selectbox("Диета", df['Diet'].dropna().unique())
    medications = st.selectbox("Употребление лекарств", df['Medications'].dropna().unique())
    seizures = st.selectbox("Травмы/Заболевания", df['Seizures'].dropna().unique())
    spay_neuter = st.selectbox("Стерилизован", df['Spay/Neuter Status'].dropna().unique())

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame({
        'Age': [age], 'Weight (lbs)': [weight], 'Annual Vet Visits': [annual_vet],
        'Daily Activity Level': [daily_activity], 'Diet': [diet],
        'Medications': [medications], 'Seizures': [seizures],
        'Spay/Neuter Status': [spay_neuter]
    })

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    st.divider()
    st.subheader("Result")
    if prediction == 1:
        st.success("**Healthy**")
    else:
        st.error("**Sick**")

    st.metric("Healthy", f"{probability[1]*100:.1f}%")
    st.metric("Sick", f"{probability[0]*100:.1f}%")
