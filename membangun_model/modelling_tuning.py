import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import dagshub  # <-- TAMBAHKAN INI

# --- KONFIGURASI DAGSHUB ---
DAGSHUB_USERNAME = "pupuyypyuput15"
DAGSHUB_REPONAME = "MLflow_DagsHub_Rahma_Putri_vscode"

# Inisialisasi koneksi DagsHub ke MLflow
dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPONAME, mlflow=True)

# 1. Load Data
print("Membaca data kesehatan...")
df = pd.read_csv('healthcare_dataset_cleaned.csv')

# Menangani Nilai Kosong (NaN)
df = df.dropna(subset=['Test Results_Encoded'])
df = df.fillna(0)

# Pisahkan Fitur (X) dan Target (y)
X = df.drop(['Test Results_Encoded', 'Name'], axis=1)
y = df['Test Results_Encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Setup MLflow (Nama eksperimen akan muncul di DagsHub)
mlflow.set_experiment("Eksperimen_Healthcare_Tuning")

# 3. Tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10]
}

with mlflow.start_run(run_name="Tuning_RandomForest_DagsHub"):
    print("Mencari parameter terbaik (GridSearchCV)...")
    
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("Mencatat hasil ke DagsHub...")
    mlflow.log_params(grid_search.best_params_)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # Simpan model
    mlflow.sklearn.log_model(best_model, "model_terbaik")
    
    print("-" * 30)
    print("Selesai! Cek dashboard DagsHub kamu sekarang.")
    print(f"Accuracy: {acc:.2f}")
    print("-" * 30)