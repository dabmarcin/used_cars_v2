import pandas as pd
from pycaret.classification import setup, compare_models, finalize_model, save_model
import streamlit as st
import boto3
import os

# 📌 Wczytanie danych z plików CSV
files = ["used_cars_clean_v2.csv"]
dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs, ignore_index=True)

# Konwersja kolumny 'price' na typ string
df['price'] = df['price'].astype(str)

# Usunięcie znaku '$' i spacji z kolumny 'price'
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False)

# Konwersja kolumny 'price' na typ liczbowy (float)
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Usunięcie wierszy, gdzie cena jest większa niż 4500000
df_filtered = df[df['price'] <= 450000].copy()

# Usunięcie wierszy z autami ze skrzynią "Transmission w/Dual Shift Mode"
df_filtered = df_filtered[df_filtered['transmission'] != "Transmission w/Dual Shift Mode"].copy()

# Wyświetlenie najmniejszej wartości z kolumny 'price' po filtracji
min_price = df_filtered['price'].min()
print(f"Najmniejsza cena (po filtracji): {min_price}")

# Wyświetlenie największej wartości z kolumny 'price' po filtracji
max_price = df_filtered['price'].max()
print(f"Największa cena (po filtracji): {max_price}")

# Wyświetlenie średniej wartości z kolumny 'price' po filtracji
mean_price = df_filtered['price'].mean()
print(f"Średnia cena (po filtracji): {mean_price}")

# Zapisanie przefiltrowanego DataFrame do nowego pliku CSV
csv_file_path = 'used_cars_clean_v3.csv'
df_filtered.to_csv(csv_file_path, index=False)

print(f"Przefiltrowane dane zostały zapisane do pliku {csv_file_path}")

# Wyświetlenie komunikatu w Streamlit
st.write(f"Dane zostały pomyślnie przygotowane i zapisane do pliku: `{csv_file_path}`")

# Konfiguracja dostępu do DigitalOcean Spaces
try:
    access_key = st.secrets["do_spaces"]["access_key"]
    secret_key = st.secrets["do_spaces"]["secret_key"]
    bucket_name = st.secrets["do_spaces"]["bucket_name"]
    region_name = st.secrets["do_spaces"]["region"]
    endpoint_url = st.secrets["do_spaces"]["endpoint_url"]

    # Inicjalizacja klienta S3 dla DigitalOcean Spaces
    session = boto3.session.Session()
    client = session.client('s3',
                            config=boto3.session.Config(signature_version='s3v4'),
                            endpoint_url=endpoint_url,
                            aws_access_key_id=access_key,
                            aws_secret_access_key=secret_key)

    # Nazwa pliku, który chcemy wysłać na Space
    file_to_upload = csv_file_path

    # Wysyłanie pliku na Space
    client.upload_file(file_to_upload, bucket_name, file_to_upload)

    print(f"Plik {file_to_upload} został pomyślnie wysłany do bucketu '{bucket_name}' na DigitalOcean Spaces.")
    st.write(f"Plik `{file_to_upload}` został również pomyślnie wysłany na DigitalOcean Spaces do bucketu: `{bucket_name}`.")

except KeyError as e:
    print(f"Brak wymaganego klucza w sekcji [do_spaces] pliku .streamlit/secrets.toml: {e}")
    st.error(f"Brak wymaganego klucza w sekcji `[do_spaces]` pliku `.streamlit/secrets.toml`: `{e}`")
except Exception as e:
    print(f"Wystąpił błąd podczas wysyłania pliku na DigitalOcean Spaces: {e}")
    st.error(f"Wystąpił błąd podczas wysyłania pliku na DigitalOcean Spaces: `{e}`")