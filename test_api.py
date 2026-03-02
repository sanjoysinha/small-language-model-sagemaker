import statistics
import time

import requests

BASE_URL = "http://localhost:8000"

PHISHING_SAMPLE = (
    "Dear valued customer, your Bank of America account has been suspended. "
    "Click the link below immediately to verify your identity and restore access. "
    "Failure to act within 24 hours will result in permanent account closure. "
    "http://bank0famerica-verify.suspicious-domain.com/login"
)

SAFE_SAMPLE = (
    "Hi team, please find attached the Q3 budget report. "
    "Let me know if you have any questions before the Friday review meeting. "
    "Best regards, Sarah"
)


def test_health():
    resp = requests.get(f"{BASE_URL}/health")
    data = resp.json()
    print(f"  Health response: {data}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {data}"
    assert data.get("status") == "healthy", f"Status: {data.get('status')}"
    assert data.get("is_ready") is True, f"is_ready: {data.get('is_ready')}"
    print("[PASS] Health check")


def test_single_prediction():
    # Test phishing detection
    resp = requests.post(
        f"{BASE_URL}/predict",
        json={"email_text": PHISHING_SAMPLE},
    )
    assert resp.status_code == 200
    data = resp.json()
    print(
        f"  Phishing sample -> label={data['label']}, "
        f"prob={data['phishing_probability']:.4f}, "
        f"latency={data['latency_ms']:.1f}ms"
    )

    # Test safe email detection
    resp = requests.post(
        f"{BASE_URL}/predict",
        json={"email_text": SAFE_SAMPLE},
    )
    assert resp.status_code == 200
    data = resp.json()
    print(
        f"  Safe sample -> label={data['label']}, "
        f"prob={data['phishing_probability']:.4f}, "
        f"latency={data['latency_ms']:.1f}ms"
    )
    print("[PASS] Single prediction")


def test_batch_prediction():
    resp = requests.post(
        f"{BASE_URL}/predict/batch",
        json={
            "emails": [
                {"email_text": PHISHING_SAMPLE},
                {"email_text": SAFE_SAMPLE},
                {"email_text": PHISHING_SAMPLE},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 3
    print(
        f"  Batch of 3 emails -> total_latency={data['total_latency_ms']:.1f}ms"
    )
    for i, pred in enumerate(data["predictions"]):
        print(f"    [{i}] {pred['label']} (prob={pred['phishing_probability']:.4f})")
    print("[PASS] Batch prediction")


def test_latency_benchmark():
    latencies = []
    n_requests = 100
    for _ in range(n_requests):
        start = time.perf_counter()
        resp = requests.post(
            f"{BASE_URL}/predict",
            json={"email_text": PHISHING_SAMPLE},
        )
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        assert resp.status_code == 200

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(n_requests * 0.95) - 1]
    p99 = sorted(latencies)[int(n_requests * 0.99) - 1]
    avg = statistics.mean(latencies)
    print(
        f"  {n_requests} requests: avg={avg:.1f}ms, "
        f"p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms"
    )
    print("[PASS] Latency benchmark")


def test_error_handling():
    # Empty text
    resp = requests.post(f"{BASE_URL}/predict", json={"email_text": ""})
    assert resp.status_code == 422
    print("  Empty text -> 422 (rejected)")

    # Missing field
    resp = requests.post(f"{BASE_URL}/predict", json={})
    assert resp.status_code == 422
    print("  Missing field -> 422 (rejected)")

    print("[PASS] Error handling")


if __name__ == "__main__":
    print("=" * 50)
    print("Phishing Detection API - End-to-End Tests")
    print("=" * 50)

    test_health()
    test_single_prediction()
    test_batch_prediction()
    test_latency_benchmark()
    test_error_handling()

    print()
    print("=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
