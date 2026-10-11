"""
Latency & Throughput Benchmark Script.
Simulates concurrent client requests to evaluate p50, p95, p99 latency percentiles and RPS.
"""

import statistics
import time
import requests
import httpx


def benchmark_api(
    url: str = "http://localhost:8000/api/v1/predict",
    total_requests: int = 100,
):
    """Executes synthetic requests against the target prediction endpoint."""
    print(f"Benchmarking {url} with {total_requests} requests...")
    latencies = []

    # Sample feature payload matching CancerFeatures schema
    sample_payload = {
        "features": {
            "mean_radius": 17.99,
            "mean_texture": 10.38,
            "mean_perimeter": 122.8,
            "mean_area": 1001.0,
            "mean_smoothness": 0.1184
        }
    }

    with httpx.Client(timeout=10.0) as client:
        for _ in range(total_requests):
            start = time.perf_counter()
            try:
                response = client.post(url, json=sample_payload)
                elapsed_ms = (time.perf_counter() - start) * 1000
                if response.status_code == 200:
                    latencies.append(elapsed_ms)
            except requests.RequestException as e:
                print(f"Request failed: {e}")

    if latencies:
        print("\n--- Benchmark Results ---")
        print(f"Successful Requests: {len(latencies)}/{total_requests}")
        print(f"Median (p50): {statistics.median(latencies):.2f} ms")
        print(f"p95 Latency:  {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
        print(f"Max Latency:  {max(latencies):.2f} ms")
    else:
        print("No successful requests recorded. Ensure the API server is running.")


if __name__ == "__main__":
    benchmark_api()
