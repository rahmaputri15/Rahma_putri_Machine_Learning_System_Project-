import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

# 1. Load Data
print("Membaca data kesehatan...")
df = pd.read_csv('healthcare_dataset_cleaned.csv')

# --- PERBAIKAN: Menangani Nilai Kosong (NaN) ---
# Menghapus baris yang kolom targetnya kosong
print(f"Jumlah data awal: {len(df)}")
df = df.dropna(subset=['Test Results_Encoded']) 

# Mengisi nilai kosong di kolom fitur (X) dengan 0 (agar model tidak error)
df = df.fillna(0)
print(f"Jumlah data setelah pembersihan NaN: {len(df)}")

# Pisahkan Fitur (X) dan Target (y)
# Pastikan kolom 'Test Results_Encoded' dan 'Name' ada di dataset
X = df.drop(['Test Results_Encoded', 'Name'], axis=1)
y = df['Test Results_Encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow
mlflow.set_experiment("Eksperimen_Healthcare_Basic")

if __name__ == "__main__":
    # Aktifkan pencatatan otomatis
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        print("Sedang melatih model...")
        
        # Inisialisasi model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Melatih model
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        
        # Output hasil
        akurasi = accuracy_score(y_test, y_pred)
        print(f"---")
        print(f"Selesai!")
        print(f"Akurasi Model: {akurasi:.2f}")
        print(f"---")