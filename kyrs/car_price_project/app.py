import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

# Заголовок приложения
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction App")
st.markdown("Predict the selling price of used cars based on their features")

# Боковая панель для загрузки модели и данных
st.sidebar.header("Model & Data")
st.sidebar.markdown("---")

# Загрузка сохраненных артефактов модели
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('model_artifacts/car_price_model.pkl')
        with open('model_artifacts/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        scaler = joblib.load('model_artifacts/scaler.pkl')
        with open('model_artifacts/feature_names.pkl', 'rb') as f:
            features = pickle.load(f)
        return model, label_encoders, scaler, features
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None, None

model, label_encoders, scaler, features = load_model_artifacts()

if model is not None:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.error("❌ Failed to load model artifacts")

st.sidebar.markdown("---")

# Функция для извлечения числовых значений
def extract_numeric_advanced(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = str(value).strip().lower()
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            return float(numbers[0])
    try:
        return float(value)
    except:
        return np.nan

# Функция для предобработки данных
def preprocess_input(data_dict, label_encoders):
    """Преобразует входные данные в формат для модели"""
    df_input = pd.DataFrame([data_dict])
    
    # Извлечение марки из названия
    df_input['brand'] = df_input['name'].str.split().str[0]
    
    # Группировка редких брендов
    known_brands = list(label_encoders['brand'].classes_)
    df_input['brand'] = df_input['brand'].apply(
        lambda x: x if x in known_brands else 'Other'
    )
    
    # Обработка числовых полей
    df_input['mileage'] = df_input['mileage'].apply(extract_numeric_advanced)
    df_input['engine'] = df_input['engine'].apply(extract_numeric_advanced)
    
    # Создание признака возраста
    current_year = datetime.now().year
    df_input['car_age'] = current_year - df_input['year']
    
    # Создание интерактивных признаков
    df_input['age_mileage_ratio'] = df_input['car_age'] / (df_input['mileage'] + 1)
    
    # Кодирование категориальных переменных
    owner_mapping = {
        'First Owner': 0,
        'Second Owner': 1,
        'Third Owner': 2,
        'Fourth & Above Owner': 3
    }
    df_input['owner_encoded'] = df_input['owner'].map(owner_mapping)
    df_input['owner_encoded'] = df_input['owner_encoded'].fillna(1)  # медианное значение
    
    # Частотное кодирование для fuel
    fuel_freq = {
        'Petrol': 0.598,
        'Diesel': 0.394,
        'CNG': 0.005,
        'LPG': 0.003,
        'Electric': 0.001
    }
    df_input['fuel_freq'] = df_input['fuel'].map(fuel_freq)
    df_input['fuel_freq'] = df_input['fuel_freq'].fillna(0.598)
    
    # Label Encoding для других категориальных признаков
    for col in ['seller_type', 'transmission', 'brand']:
        le = label_encoders[col]
        # Для неизвестных значений используем наиболее частый класс
        known_classes = set(le.classes_)
        df_input[col] = df_input[col].apply(lambda x: x if x in known_classes else le.classes_[0])
        df_input[col + '_encoded'] = le.transform(df_input[col])
    
    # Выбор финальных признаков (без price_per_km, так как это целевая переменная)
    final_features = [
        'year', 'km_driven', 'mileage', 'engine', 'car_age',
        'owner_encoded', 'fuel_freq',
        'seller_type_encoded', 'transmission_encoded', 'brand_encoded',
        'age_mileage_ratio'
    ]
    
    # Создание DataFrame с финальными признаками
    X = df_input[final_features].copy()
    
    # Масштабирование числовых признаков
    numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'car_age', 'age_mileage_ratio']
    X_scaled = X.copy()
    X_scaled[numeric_features] = scaler.transform(X[numeric_features])
    
    return X_scaled

# Основные вкладки приложения
tab1, tab2, tab3 = st.tabs(["📊 Batch Prediction", "✍️ Single Prediction", "📈 Model Info"])

with tab1:
    st.header("Batch Prediction from CSV")
    st.markdown("Upload a CSV file with car data for batch predictions")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.write("### Preview of uploaded data")
        st.dataframe(df_uploaded.head(), use_container_width=True)
        
        if st.button("🚀 Predict Prices", type="primary"):
            try:
                predictions = []
                progress_bar = st.progress(0)
                
                for idx, row in df_uploaded.iterrows():
                    # Подготовка данных
                    X_processed = preprocess_input(row.to_dict(), label_encoders)
                    
                    # Предсказание
                    pred = model.predict(X_processed)[0]
                    predictions.append(pred)
                    
                    # Обновление прогресс-бара
                    progress_bar.progress((idx + 1) / len(df_uploaded))
                
                # Добавление предсказаний в DataFrame
                df_result = df_uploaded.copy()
                df_result['predicted_price'] = predictions
                df_result['predicted_price'] = df_result['predicted_price'].round(2)
                
                st.success(f"✅ Successfully predicted prices for {len(df_uploaded)} cars!")
                
                # Отображение результатов
                st.write("### Prediction Results")
                st.dataframe(df_result, use_container_width=True)
                
                # Кнопка для скачивания результатов
                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name='car_predictions.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

with tab2:
    st.header("Single Car Prediction")
    st.markdown("Enter details for a single car to get a price prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Car Name (e.g., 'Hyundai i20')", "Hyundai i20")
        year = st.number_input("Year", min_value=1990, max_value=2024, value=2015)
        km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
        fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
        seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
        
    with col2:
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
        mileage = st.text_input("Mileage (e.g., '18.0 kmpl')", "18.0 kmpl")
        engine = st.text_input("Engine (e.g., '1197 CC')", "1197 CC")
    
    if st.button("🔮 Predict Price", type="primary"):
        if model is None:
            st.error("Model not loaded. Please check model artifacts.")
        else:
            try:
                # Сбор данных
                input_data = {
                    'name': name,
                    'year': year,
                    'km_driven': km_driven,
                    'fuel': fuel,
                    'seller_type': seller_type,
                    'transmission': transmission,
                    'owner': owner,
                    'mileage': mileage,
                    'engine': engine
                }
                
                # Предобработка
                X_processed = preprocess_input(input_data, label_encoders)
                
                # Предсказание
                prediction = model.predict(X_processed)[0]
                
                # Отображение результата
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                col_pred1, col_pred2 = st.columns([1, 2])
                
                with col_pred1:
                    st.metric(
                        label="Predicted Selling Price",
                        value=f"₹{prediction:,.0f}",
                        delta=None
                    )
                
                with col_pred2:
                    st.info("""
                    **Note:** This is an estimated price based on the model's training data. 
                    Actual market price may vary based on condition, location, and other factors.
                    """)
                
                # Дополнительная информация
                with st.expander("📋 Input Details"):
                    st.json(input_data)
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

with tab3:
    st.header("Model Information")
    
    if model is not None and features is not None:
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.subheader("Model Details")
            st.write(f"**Model Type:** Gradient Boosting Regressor")
            st.write(f"**Number of Features:** {len(features)}")
            st.write(f"**Training Date:** Model artifacts timestamp")
            
        with col_info2:
            st.subheader("Feature Importance")
            # Вычисление важности признаков
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.dataframe(importance_df.head(10), use_container_width=True)
        
        st.subheader("Required Features")
        st.write("The model expects the following features:")
        for i, feature in enumerate(features, 1):
            st.write(f"{i}. {feature}")
        
        st.markdown("---")
        st.subheader("How to Use")
        st.markdown("""
        1. **Single Prediction:** Fill in all fields in the 'Single Prediction' tab
        2. **Batch Prediction:** Upload a CSV file with the following columns:
           - `name`: Car name with brand
           - `year`: Manufacturing year
           - `km_driven`: Kilometers driven
           - `fuel`: Fuel type
           - `seller_type`: Type of seller
           - `transmission`: Transmission type
           - `owner`: Number of previous owners
           - `mileage`: Mileage with units
           - `engine`: Engine capacity with units
        """)
    else:
        st.warning("Model information not available")

# Футер
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Car Price Prediction App • Built with Streamlit and Scikit-learn</p>
    </div>
    """,
    unsafe_allow_html=True
)