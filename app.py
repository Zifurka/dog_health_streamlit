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

    X = df.drop('Healthy', axis=1)
    y = df['Healthy']

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    ordinal_features = ['Breed Size', 'Daily Activity Level', 'Owner Activity Level']
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_features = [col for col in categorical_features if col not in ordinal_features]

    size_categories = ['Small', 'Medium', 'Large']
    daily_activity_categories = ['Low', 'Moderate', 'Active', 'Very Active']
    owner_activity_categories = ['Low', 'Moderate', 'Active', 'Very Active']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
        ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(categories=[size_categories, daily_activity_categories, owner_activity_categories]))]), ordinal_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(X_processed, y)

    return model, preprocessor, df

model, preprocessor, df = train_model()

st.title("Dog Breed Health Predictor")
st.write("Fill in the dog's parameters to predict its health status.")

col1, col2, col3 = st.columns(3)

with col1:
    breed = st.selectbox("Breed", sorted(df['Breed'].dropna().unique()))
    sex = st.selectbox("Sex", df['Sex'].dropna().unique())
    age = st.slider("Age (years)", 0, 20, 5)
    weight = st.slider("Weight (lbs)", 0, 150, 30)

with col2:
    breed_size = st.selectbox("Breed Size", ['Small', 'Medium', 'Large'])
    daily_activity = st.selectbox("Daily Activity Level", ['Low', 'Moderate', 'Active', 'Very Active'])
    diet = st.selectbox("Diet", df['Diet'].dropna().unique())
    food_brand = st.selectbox("Food Brand", df['Food Brand'].dropna().unique())

with col3:
    spay_neuter = st.selectbox("Spay/Neuter Status", df['Spay/Neuter Status'].dropna().unique())
    other_pets = st.selectbox("Other Pets in Household", df['Other Pets in Household'].dropna().unique())
    medications = st.selectbox("Medications", df['Medications'].dropna().unique())
    seizures = st.selectbox("Seizures", df['Seizures'].dropna().unique())

walk_distance = st.slider("Daily Walk Distance (miles)", 0.0, 10.0, 2.0)
hours_sleep = st.slider("Hours of Sleep", 0.0, 20.0, 10.0)
play_time = st.slider("Play Time (hrs)", 0.0, 10.0, 2.0)
owner_activity = st.selectbox("Owner Activity Level", ['Low', 'Moderate', 'Active', 'Very Active'])
annual_vet = st.slider("Annual Vet Visits", 0.0, 10.0, 1.0)
avg_temp = st.slider("Average Temperature (F)", 0.0, 120.0, 70.0)

if st.button("Predict", type="primary"):
    input_data = pd.DataFrame({
        'Breed': [breed], 'Breed Size': [breed_size], 'Sex': [sex],
        'Age': [age], 'Weight (lbs)': [weight],
        'Spay/Neuter Status': [spay_neuter], 'Daily Activity Level': [daily_activity],
        'Diet': [diet], 'Food Brand': [food_brand],
        'Daily Walk Distance (miles)': [walk_distance],
        'Other Pets in Household': [other_pets], 'Medications': [medications],
        'Seizures': [seizures], 'Hours of Sleep': [hours_sleep],
        'Play Time (hrs)': [play_time], 'Owner Activity Level': [owner_activity],
        'Annual Vet Visits': [annual_vet], 'Average Temperature (F)': [avg_temp]
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
