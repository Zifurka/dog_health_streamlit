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

st.set_page_config(page_title="Предсказание здоровья собак", page_icon="dog", layout="wide")

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

st.title("Предсказание здоровья собаки")
st.write("Заполните параметры для предсказания состояния здоровья.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Возраст (лет)", 0, 20, 5)
    weight = st.slider("Вес (фунты)", 0, 150, 30)
    annual_vet = st.slider("Визитов к ветеринару в год", 0.0, 10.0, 1.0)
    daily_activity = st.slider("Уровень активности в день", 0, 10, 5)

with col2:
    diet = st.selectbox("Рацион питания", df['Diet'].dropna().unique())
    medications = st.selectbox("Лекарства", df['Medications'].dropna().unique())
    seizures = st.selectbox("Судороги", df['Seizures'].dropna().unique())
    spay_neuter = st.selectbox("Стерилизация/Кастрация", df['Spay/Neuter Status'].dropna().unique())

if st.button("Предсказать", type="primary"):
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
    st.subheader("Результат")
    if prediction == 1:
        st.success("**Здорова**")
    else:
        st.error("**Больна**")

    st.metric("Здорова", f"{probability[1]*100:.1f}%")
    st.metric("Больна", f"{probability[0]*100:.1f}%")
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

st.set_page_config(page_title="Предсказание здоровья собак", page_icon="dog", layout="wide")

TOP_FEATURES = ['Age', 'Weight (lbs)', 'Daily Activity Level', 'Diet', 'Medications', 'Seizures', 'Annual Vet Visits', 'Spay/Neuter Status']

# Категории для Daily Activity Level
DAILY_ACTIVITY_CATEGORIES = ['Low', 'Moderate', 'Active', 'Very Active']

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

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features),
        ('ord', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ordinal', OrdinalEncoder(categories=[DAILY_ACTIVITY_CATEGORIES]))]), ordinal_features)
    ])

    X_processed = preprocessor.fit_transform(X)
    model = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')
    model.fit(X_processed, y)

    return model, preprocessor, df

model, preprocessor, df = train_model()

st.title("Предсказание здоровья собаки")
st.write("Заполните параметры для предсказания состояния здоровья.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Возраст (лет)", 0, 20, 5)
    weight = st.slider("Вес (фунты)", 0, 150, 30)
    annual_vet = st.slider("Визитов к ветеринару в год", 0.0, 10.0, 1.0)
    daily_activity = st.selectbox("Уровень активности", DAILY_ACTIVITY_CATEGORIES)

with col2:
    diet = st.selectbox("Рацион питания", sorted(df['Diet'].dropna().unique()))
    medications = st.selectbox("Лекарства", sorted(df['Medications'].dropna().unique()))
    seizures = st.selectbox("Судороги", sorted(df['Seizures'].dropna().unique()))
    spay_neuter = st.selectbox("Стерилизация/Кастрация", sorted(df['Spay/Neuter Status'].dropna().unique()))

if st.button("Предсказать", type="primary"):
    input_data = pd.DataFrame({
        'Age': [age], 
        'Weight (lbs)': [weight], 
        'Annual Vet Visits': [annual_vet],
        'Daily Activity Level': [daily_activity], 
        'Diet': [diet],
        'Medications': [medications], 
        'Seizures': [seizures],
        'Spay/Neuter Status': [spay_neuter]
    })

    # Преобразуем все строковые значения в строки
    for col in ['Diet', 'Medications', 'Seizures', 'Spay/Neuter Status', 'Daily Activity Level']:
        input_data[col] = input_data[col].astype(str)

    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0]

    st.divider()
    st.subheader("Результат")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        if prediction == 1:
            st.success("**Здорова**")
        else:
            st.error("**Больна**")
    
    with col_res2:
        st.metric("Вероятность здоровья", f"{probability[1]*100:.1f}%")
    
    with col_res3:
        st.metric("Вероятность болезни", f"{probability[0]*100:.1f}%")
    
    st.divider()
    st.caption("Модель обучена на синтетических данных и служит только для демонстрации работы приложения.")
