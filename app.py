import streamlit as st
import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Предсказание здоровья собак", page_icon="dog", layout="wide")

NUMERIC_FEATURES = ['Age', 'Weight (lbs)', 'Daily Walk Distance (miles)', 'Hours of Sleep', 'Play Time (hrs)', 'Annual Vet Visits', 'Average Temperature (F)']
ORDINAL_FEATURES = ['Breed Size', 'Daily Activity Level', 'Owner Activity Level']
CATEGORICAL_FEATURES = ['Breed', 'Sex', 'Spay/Neuter Status', 'Diet', 'Food Brand', 'Other Pets in Household', 'Medications', 'Seizures']

SIZE_CATEGORIES = ['Small', 'Medium', 'Large']
ACTIVITY_CATEGORIES = ['Low', 'Moderate', 'Active', 'Very Active']


@st.cache_resource
def load_model():
    df = pd.read_csv('synthetic_dog_breed_health_data.csv')
    df.replace('', np.nan, inplace=True)
    df = df.drop(columns=['ID', 'Synthetic'], errors='ignore')

    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df['Healthy'] = df['Healthy'].map({'Yes': 1, 'No': 0})

    X = df[NUMERIC_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES]
    y = df['Healthy']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), NUMERIC_FEATURES),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL_FEATURES),
        ('ord', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(categories=[SIZE_CATEGORIES, ACTIVITY_CATEGORIES, ACTIVITY_CATEGORIES]))
        ]), ORDINAL_FEATURES)
    ])

    X_processed = preprocessor.fit_transform(X)
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(X_processed, y)

    return model, preprocessor, df


model, preprocessor, df = load_model()

st.title("Предсказание здоровья собаки")
st.write("Заполните параметры для предсказания состояния здоровья.")

col1, col2 = st.columns(2)

with col1:
    breed = st.selectbox("Порода", sorted(df['Breed'].dropna().unique()), key="breed")
    sex = st.selectbox("Пол", df['Sex'].dropna().unique(), key="sex")
    age = st.slider("Возраст (лет)", 0, 20, 5, key="age")
    weight = st.slider("Вес (фунты)", 0, 150, 30, key="weight")
    breed_size = st.selectbox("Размер породы", SIZE_CATEGORIES, key="breed_size")
    spay_neuter = st.selectbox("Стерилизация/Кастрация", df['Spay/Neuter Status'].dropna().unique(), key="spay")

with col2:
    daily_activity = st.selectbox("Активность в день", ACTIVITY_CATEGORIES, key="daily_act")
    owner_activity = st.selectbox("Активность владельца", ACTIVITY_CATEGORIES, key="owner_act")
    diet = st.selectbox("Рацион", df['Diet'].dropna().unique(), key="diet")
    food_brand = st.selectbox("Бренд корма", df['Food Brand'].dropna().unique(), key="food")
    medications = st.selectbox("Лекарства", df['Medications'].dropna().unique(), key="med")
    seizures = st.selectbox("Судороги", df['Seizures'].dropna().unique(), key="seiz")
    other_pets = st.selectbox("Другие питомцы", df['Other Pets in Household'].dropna().unique(), key="pets")

walk_distance = st.slider("Прогулки в день (мили)", 0.0, 10.0, 2.0, key="walk")
hours_sleep = st.slider("Часы сна", 0.0, 20.0, 10.0, key="sleep")
play_time = st.slider("Игровое время (часы)", 0.0, 10.0, 2.0, key="play")
annual_vet = st.slider("Визитов к ветеринару в год", 0.0, 10.0, 1.0, key="vet")
avg_temp = st.slider("Средняя температура (F)", 0.0, 120.0, 70.0, key="temp")

if st.button("Предсказать", type="primary", key="predict_btn"):
    input_data = pd.DataFrame({
        'Age': [age], 'Weight (lbs)': [weight],
        'Daily Walk Distance (miles)': [walk_distance], 'Hours of Sleep': [hours_sleep],
        'Play Time (hrs)': [play_time], 'Annual Vet Visits': [annual_vet],
        'Average Temperature (F)': [avg_temp],
        'Breed Size': [breed_size], 'Daily Activity Level': [daily_activity],
        'Owner Activity Level': [owner_activity],
        'Breed': [breed], 'Sex': [sex], 'Spay/Neuter Status': [spay_neuter],
        'Diet': [diet], 'Food Brand': [food_brand],
        'Other Pets in Household': [other_pets], 'Medications': [medications],
        'Seizures': [seizures]
    })

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    st.divider()
    st.subheader("Результат")
    if prediction == 1:
        st.success("**Здорова**")
    else:
        st.error("**Больна**")

    c1, c2 = st.columns(2)
    c1.metric("Здорова", f"{probability[1]*100:.1f}%")
    c2.metric("Больна", f"{probability[0]*100:.1f}%")
