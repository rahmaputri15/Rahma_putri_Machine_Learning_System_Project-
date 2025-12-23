from prometheus_client import start_http_server, Gauge, Counter, Summary
import psutil
import time
import os

# ========== METRICS ==========
# Sistem
cpu_usage = Gauge('system_cpu_usage', 'Persentase penggunaan CPU')
ram_usage = Gauge('system_ram_usage', 'Persentase penggunaan RAM')
disk_usage = Gauge('system_disk_usage', 'Persentase penggunaan Disk')

# Model Traffic (akan diupdate dari inference.py via file)
http_requests_total = Gauge('http_requests_total', 'Total request ke model')
model_latency_sum = Gauge('model_latency_seconds_sum', 'Total latency')
model_latency_count = Gauge('model_latency_seconds_count', 'Total request count')
model_errors_total = Gauge('model_errors_total', 'Total error pada model')

# Network & Memory
network_io_sent = Gauge('network_sent_bytes', 'Byte terkirim')
network_io_recv = Gauge('network_recv_bytes', 'Byte diterima')
python_mem = Gauge('python_memory_usage_bytes', 'Memory script ini')
active_threads = Gauge('active_threads_count', 'Jumlah thread aktif')

def collect_metrics():
    """Collect system metrics dan baca model metrics dari file"""
    import json
    
    while True:
        # System metrics
        cpu_usage.set(psutil.cpu_percent())
        ram_usage.set(psutil.virtual_memory().percent)
        disk_usage.set(psutil.disk_usage('/').percent)
        
        net = psutil.net_io_counters()
        network_io_sent.set(net.bytes_sent)
        network_io_recv.set(net.bytes_recv)
        
        process = psutil.Process(os.getpid())
        python_mem.set(process.memory_info().rss)
        active_threads.set(threading.active_count())
        
        # Baca model metrics dari file
        try:
            if os.path.exists('model_metrics.json'):
                with open('model_metrics.json', 'r') as f:
                    data = json.load(f)
                    http_requests_total.set(data.get('request_count', 0))
                    model_latency_sum.set(data.get('latency_sum', 0))
                    model_latency_count.set(data.get('latency_count', 0))
                    model_errors_total.set(data.get('error_count', 0))
        except Exception as e:
            print(f"Warning: Could not read model metrics: {e}")
        
        time.sleep(5)

if __name__ == '__main__':
    import threading
    
    print("=" * 70)
    print("Starting Prometheus Exporter")
    print("=" * 70)
    
    start_http_server(8000)
    print(f"Prometheus Exporter running at: http://localhost:8000")
    print(f"Metrics endpoint: http://localhost:8000/metrics")
    print("=" * 70)
    print("Waiting for model metrics from inference.py...")
    print("=" * 70)
    
    collect_metrics()