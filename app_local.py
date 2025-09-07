
import streamlit as st
import pandas as pd
import pickle
import io
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
import requests
import os
from datetime import datetime
import hashlib
import string
from io import BytesIO

# --- KONFIG GITHUB (aktualizuj TAG przy nowej wersji) ---
GITHUB_USER = "dabmarcin"
GITHUB_REPO = "used_cars_v2"
TAG         = "v1.0.0"  # Release jest OPUBLIKOWANY (nie draft)

# >>> WSZYSTKIE 3 pliki z Release assets <<<
MODEL_URL  = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{TAG}/model_samochody.pkl"
SCALER_URL = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{TAG}/scaler_samochody.pkl"
CSV_URL    = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{TAG}/used_cars_clean_v3.csv"

# --- KONFIG HASHY (pełne 64-znakowe, bez zmian w przyszłości poza podbiciem) ---
MODEL_SHA256  = "a7c0b369747911bd4af1c587b93d5f680146364be24aacc491ae2cd8837968e9"
SCALER_SHA256 = "7de8c9dbb493095a7cc33bb0b9258fe4a5128ef221b34f563edc9cdc0a14a86e"
CSV_SHA256    = "9b982d6c315a79aa85ec5ed82de7571f51ec2f6e8b9895c7435a5caa5ba284b5"

# Wyczyść cache po zmianie wersji/tagu, żeby uniknąć starych artefaktów
try:
    st.cache_data.clear()
    st.cache_resource.clear()
except Exception:
    pass

def _normalize_sha256(expected: str | None):
    """Akceptuje 'sha256:<hex>' lub samo <hex>; waliduje długość i znaki."""
    if expected is None:
        return None
    s = expected.strip().lower()
    if s.startswith("sha256:"):
        s = s.split(":", 1)[1]
    hexdigits = set(string.hexdigits.lower())
    if len(s) != 64 or any(ch not in hexdigits for ch in s):
        raise ValueError(f"Niepoprawny SHA-256: {expected}")
    return s

def _check_sha256(content: bytes, expected: str | None, name: str):
    exp = _normalize_sha256(expected)
    if exp is None:
        return
    h = hashlib.sha256(content).hexdigest()
    if h != exp:
        raise ValueError(f"Plik {name} ma inną sumę SHA-256! Oczekiwano {exp}, otrzymano {h}")

@st.cache_resource(show_spinner=False)
def load_remote_pickle(url: str, expected_sha256: str | None = None):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    content = r.content
    _check_sha256(content, expected_sha256, url)
    return pickle.load(BytesIO(content))

@st.cache_data(show_spinner=False)
def load_remote_csv(url: str, expected_sha256: str | None = None):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    content = r.content
    _check_sha256(content, expected_sha256, url)
    return pd.read_csv(BytesIO(content))

# --- Ładowanie modelu i scalera z GitHub ---
model = None
scaler = None

try:
    model = load_remote_pickle(MODEL_URL, expected_sha256=MODEL_SHA256)
    st.success("Model loaded from GitHub.")
except Exception as e:
    st.error(f"Error loading model from GitHub: {e}")

try:
    scaler = load_remote_pickle(SCALER_URL, expected_sha256=SCALER_SHA256)
    st.success("Scaler loaded from GitHub.")
except Exception as e:
    st.warning(f"Warning: error loading scaler from GitHub: {e}")
    scaler = None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_exchange_rate(target_currency):
    """
    Fetch exchange rate for target currency from USD.
    Cached for 1 hour to improve performance.
    """
    base_currency = "USD"
    api_url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
    
    try:
        # Add timeout to prevent hanging
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rates = data['rates']
        
        if target_currency in rates:
            return rates[target_currency]
        else:
            st.warning(f"Exchange rate not found for currency: {target_currency}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timeout: Exchange rate service is taking too long to respond.")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Connection error: Unable to connect to exchange rate service.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching exchange rate: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"API response format error: {e}")
        return None

if model is not None:
    try:
        # --- Ładowanie danych treningowych z GitHub ---
        df_train = load_remote_csv(CSV_URL, expected_sha256=CSV_SHA256)
        st.success("Training data loaded from GitHub.")
        df_train.dropna(subset=['brand', 'model_year', 'milage', 'fuel_type', 'transmission', 'price', 'engine_hp', 'engine_cylinders', 'ext_col'], inplace=True)

        # --- Opcje dla dropdownów ---
        brands = sorted(df_train['brand'].unique().tolist())
        years = sorted(df_train['model_year'].unique().astype(int).tolist(), reverse=True)
        fuel_types = sorted(df_train['fuel_type'].unique().tolist())
        transmissions = sorted(df_train['transmission'].unique().tolist())
        colors = sorted(df_train['ext_col'].unique().tolist())
        engine_cylinders_options = sorted(df_train['engine_cylinders'].unique().astype(int).tolist())

        # --- Oblicz przedziały przebiegu ---
        milage_bins = pd.cut(df_train['milage'], bins=5, include_lowest=True, labels=False, duplicates='drop')
        milage_ranges = []
        num_milage_bins = 5
        bins_milage = pd.cut(df_train['milage'], bins=num_milage_bins, include_lowest=True, labels=False, retbins=True)[1]
        for i in range(num_milage_bins):
            lower_bound = int(bins_milage[i])
            upper_bound = int(bins_milage[i+1]) if i < num_milage_bins - 1 else int(df_train['milage'].max())
            milage_ranges.append(f"{lower_bound} - {upper_bound}")

        # --- Oblicz przedziały mocy silnika ---
        engine_hp_bins = pd.cut(df_train['engine_hp'], bins=5, include_lowest=True, labels=False, duplicates='drop')
        engine_hp_ranges = []
        num_hp_bins = 5
        bins_hp = pd.cut(df_train['engine_hp'], bins=num_hp_bins, include_lowest=True, labels=False, retbins=True)[1]
        for i in range(num_hp_bins):
            lower_bound = int(bins_hp[i])
            upper_bound = int(bins_hp[i+1]) if i < num_hp_bins - 1 else int(df_train['engine_hp'].max())
            engine_hp_ranges.append(f"{lower_bound} - {upper_bound}")

    except Exception as e:
        st.error(f"Error loading CSV from GitHub: {e}")
        df_train = None

    if df_train is not None:
        st.title("Car Price Estimator")

        # --- Formularz z dropdownami ---
        with st.form("car_features"):
            brand = st.selectbox("Car Brand", options=brands)
            year = st.selectbox("Production Year", options=years)
            milage_range = st.selectbox("Mileage", options=["Never mind"] + [f"From {r}" for r in milage_ranges], help="Mileage data is in kilometers.")
            engine_hp_range = st.selectbox("Engine Power (HP)", options=["Never mind"] + [f"{r}" for r in engine_hp_ranges], help="Engine power in horsepower.")
            fuel_type = st.selectbox("Fuel Type", options=fuel_types)
            transmission = st.selectbox("Transmission", options=transmissions)
            ext_col = st.selectbox("Color", options=colors)

            submitted = st.form_submit_button("Estimate Price")

            if submitted:
                try:
                    input_data = {
                        'brand': [brand],
                        'model_year': [year],
                        'milage': [None],
                        'engine_hp': [None],
                        'fuel_type': [fuel_type],
                        'transmission': [transmission],
                        'ext_col': [ext_col],
                    }
                    input_df_base = pd.DataFrame(input_data)

                    # Obsługa przebiegu
                    milage_value = 0
                    if milage_range != "Never mind":
                        selected_milage_range = milage_range.replace("From ", "").split(" - ")
                        milage_value = int(selected_milage_range[0])

                    # Obsługa mocy silnika
                    engine_hp_value = 0
                    if engine_hp_range != "Never mind":
                        selected_engine_hp_range = engine_hp_range.split(" - ")
                        engine_hp_value = int(selected_engine_hp_range[0])

                    # --- Inżynieria cech ---
                    current_year = datetime.now().year
                    wiek = current_year - int(input_df_base['model_year'][0])
                    wiek_kwadrat = wiek**2
                    wiek_przebieg = wiek * milage_value

                    # --- Kodowanie kategorialne ---
                    categorical_data = {
                        'brand': [brand],
                        'fuel_type': [fuel_type],
                        'transmission': [transmission],
                        'ext_col': [ext_col]
                    }
                    categorical_df = pd.get_dummies(
                        pd.DataFrame(categorical_data),
                        columns=['brand', 'fuel_type', 'transmission', 'ext_col'],
                        prefix_sep='_',
                        dummy_na=False
                    )

                    # --- Przygotuj DataFrame do predykcji ---
                    numerical_data = {
                        'milage': [milage_value],
                        'wiek': [wiek],
                        'engine_hp': [engine_hp_value],
                        'engine_liters_log': [0],
                        'wiek_kwadrat': [wiek_kwadrat],
                        'wiek_przebieg': [wiek_przebieg]
                    }
                    final_input_numerical = pd.DataFrame(numerical_data)

                    final_input_df = pd.concat([final_input_numerical, categorical_df], axis=1)

                    # --- Dopasuj kolejność kolumn do modelu ---
                    expected_columns = model.feature_names_in_.tolist()
                    for col in expected_columns:
                        if col not in final_input_df.columns:
                            final_input_df[col] = 0
                    final_input_df = final_input_df[expected_columns]

                    # --- Skalowanie cech numerycznych ---
                    numerical_features_to_scale = ['milage', 'wiek', 'wiek_kwadrat', 'wiek_przebieg', 'engine_hp', 'engine_liters_log']
                    numerical_features_present = [col for col in numerical_features_to_scale if col in final_input_df.columns and scaler is not None and hasattr(scaler, 'feature_names_in_') and col in scaler.feature_names_in_]

                    if scaler is not None and numerical_features_present:
                        try:
                            scaled_features = scaler.transform(final_input_df[numerical_features_present])
                            final_input_df[numerical_features_present] = scaled_features
                        except Exception as e:
                            st.error(f"Error during scaling: {e}")
                    elif scaler is not None and not hasattr(scaler, 'feature_names_in_'):
                        st.warning("Scaler does not have feature_names_in_ attribute. Scaling may not work correctly.")
                    elif scaler is None:
                        st.warning("Scaler was not loaded. Predictions may be inaccurate.")

                    # --- Dokonaj predykcji ---
                    predicted_price_usd = model.predict(final_input_df)[0]
                    st.success(f"Predicted car price: {predicted_price_usd:.2f} $")
                    st.session_state['predicted_price_usd'] = predicted_price_usd

                except Exception as e:
                    st.error(f"Error preparing data or making prediction: {e}")

        # --- Wybór waluty do przeliczenia ---
        target_currency = st.selectbox("Select conversion currency", options=["PLN", "GBP", "EUR", "SEK"], key='selected_currency')

        if 'predicted_price_usd' in st.session_state:
            if st.button("Convert"):
                target_currency = st.session_state.get('selected_currency')
                predicted_price_usd = st.session_state.get('predicted_price_usd')
                if target_currency and predicted_price_usd is not None:
                    exchange_rate = get_exchange_rate(target_currency)
                    if exchange_rate is not None:
                        converted_price = predicted_price_usd * exchange_rate
                        st.write(f"Price in {target_currency}: {converted_price:.2f}")
                    else:
                        st.warning("Failed to retrieve exchange rate for selected currency.")
                elif not target_currency:
                    st.warning("Select currency for conversion.")

else:
    st.warning("Model was not loaded.")
