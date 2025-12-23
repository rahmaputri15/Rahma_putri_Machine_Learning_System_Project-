import requests
import time
import random
import json
import os

MODEL_URL = "http://127.0.0.1:5001/invocations"
METRICS_FILE = "model_metrics.json"

# Initialize metrics storage
metrics_data = {
    'request_count': 0,
    'latency_sum': 0.0,
    'latency_count': 0,
    'error_count': 0
}

def load_metrics():
    """Load existing metrics from file"""
    global metrics_data
    try:
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, 'r') as f:
                metrics_data = json.load(f)
    except Exception as e:
        print(f"Could not load metrics: {e}")

def save_metrics():
    """Save metrics to file"""
    try:
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics_data, f)
    except Exception as e:
        print(f"Could not save metrics: {e}")

def generate_random_payload():
    """Generate random healthcare data"""
    return {
        "dataframe_split": {
            "columns": [
                "Age",
                "Billing Amount",
                "Room Number",
                "Gender_Encoded",
                "Blood Type_Encoded",
                "Medical Condition_Encoded",
                "Date of Admission_Encoded",
                "Doctor_Encoded",
                "Hospital_Encoded",
                "Insurance Provider_Encoded",
                "Admission Type_Encoded",
                "Discharge Date_Encoded",
                "Medication_Encoded",
                "Billing_Amount_Scaled"
            ],
            "data": [[
                float(random.randint(20, 80)),           # Age
                float(random.uniform(100000, 1000000)),  # Billing Amount
                float(random.randint(100, 500)),         # Room Number
                float(random.randint(0, 1)),             # Gender_Encoded
                float(random.randint(0, 7)),             # Blood Type_Encoded
                float(random.randint(0, 5)),             # Medical Condition_Encoded
                float(random.randint(1, 365)),           # Date of Admission_Encoded
                float(random.randint(0, 50)),            # Doctor_Encoded
                float(random.randint(0, 10)),            # Hospital_Encoded
                float(random.randint(0, 5)),             # Insurance Provider_Encoded
                float(random.randint(0, 2)),             # Admission Type_Encoded
                float(random.randint(1, 365)),           # Discharge Date_Encoded
                float(random.randint(0, 20)),            # Medication_Encoded
                float(random.uniform(0, 1))              # Billing_Amount_Scaled
            ]]
        }
    }

def hit_model():
    """Send request to healthcare model and update metrics"""
    payload = generate_random_payload()
    
    start = time.time()
    try:
        response = requests.post(MODEL_URL, json=payload, timeout=5)
        latency = time.time() - start

        # Update metrics
        metrics_data['request_count'] += 1
        metrics_data['latency_sum'] += latency
        metrics_data['latency_count'] += 1
        
        # Save to file
        save_metrics()

        if response.status_code == 200:
            result = response.json()
            print(f"Request #{metrics_data['request_count']} | "
                  f"Status: {response.status_code} | "
                  f"Latency: {latency:.3f}s")
            print(f"   Prediction: {result}")
        else:
            print(f"Request #{metrics_data['request_count']} | "
                  f"Status: {response.status_code} | "
                  f"Error: {response.text[:100]}")
            metrics_data['error_count'] += 1
            save_metrics()

    except requests.exceptions.ConnectionError:
        metrics_data['request_count'] += 1
        metrics_data['error_count'] += 1
        save_metrics()
        print(f"Connection Error: Model tidak bisa diakses di {MODEL_URL}")
        print(f"   Pastikan MLflow server sedang berjalan")
        
    except Exception as e:
        metrics_data['request_count'] += 1
        metrics_data['error_count'] += 1
        save_metrics()
        print(f"Error: {type(e).__name__} - {str(e)[:100]}")

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Healthcare Model Inference Simulator")
    print("=" * 70)
    
    # RESET: Hapus file metrics lama untuk mulai dari 0
    if os.path.exists(METRICS_FILE):
        os.remove(METRICS_FILE)
        print(f"Reset metrics file - starting fresh from 0")
    
    print(f"Model URL: {MODEL_URL}")
    print(f"Metrics File: {METRICS_FILE}")
    print("=" * 70)
    print(f"Current request count: {metrics_data['request_count']}")
    print("=" * 70)
    print("\nPress Ctrl+C to stop\n")
    
    request_num = metrics_data['request_count']
    
    try:
        while True:
            request_num += 1
            print(f"\n{'='*70}")
            print(f"REQUEST #{request_num} - {time.strftime('%H:%M:%S')}")
            print(f"{'='*70}")
            
            hit_model()
            
            wait_time = random.uniform(1, 3)
            print(f"Waiting {wait_time:.1f}s before next request...")
            time.sleep(wait_time)
            
    except KeyboardInterrupt:
        print("\n\nStopping inference simulator...")
        print(f"Final stats:")
        print(f"   Total requests: {metrics_data['request_count']}")
        print(f"   Total errors: {metrics_data['error_count']}")
        if metrics_data['latency_count'] > 0:
            avg_latency = metrics_data['latency_sum'] / metrics_data['latency_count']
            print(f"   Average latency: {avg_latency:.3f}s")