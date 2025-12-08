#!/usr/bin/env python3
"""
Test script for the deployed ML model
Validates health, readiness, and prediction endpoints
"""

import argparse
import random
import sys
import time

import requests


class ModelTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "User-Agent": "ML-Model-Tester/1.0"}
        )

    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Health check passed: {data}")
            return data.get("status") == "healthy"
        except Exception as e:
            print(f"✗ Health check failed: {e}")
            return False

    def test_readiness(self) -> bool:
        """Test readiness endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/ready", timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Readiness check passed: {data}")
            return data.get("status") == "ready"
        except Exception as e:
            print(f"✗ Readiness check failed: {e}")
            return False

    def test_model_info(self) -> bool:
        """Test model info endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=5)
            response.raise_for_status()
            data = response.json()
            print(f"✓ Model info retrieved: {data}")
            return "model_type" in data
        except Exception as e:
            print(f"✗ Model info failed: {e}")
            return False

    def generate_test_features(self, n_samples: int = 1) -> list[list[float]]:
        """Generate random test features"""
        features = []
        for _ in range(n_samples):
            sample = [random.uniform(-2, 2) for _ in range(20)]
            features.append(sample)
        return features

    def test_prediction_single(self) -> bool:
        """Test single prediction"""
        try:
            features = self.generate_test_features(1)
            payload = {"features": features}

            response = self.session.post(
                f"{self.base_url}/predict", json=payload, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            print(f"✓ Single prediction successful")
            print(f"  Predictions: {data['predictions']}")
            print(
                f"  Probabilities shape: {len(data['probabilities'])}x{len(data['probabilities'][0])}"
            )

            return (
                "predictions" in data
                and "probabilities" in data
                and len(data["predictions"]) == 1
            )
        except Exception as e:
            print(f"✗ Single prediction failed: {e}")
            return False

    def test_prediction_batch(self) -> bool:
        """Test batch prediction"""
        try:
            features = self.generate_test_features(5)
            payload = {"features": features}

            response = self.session.post(
                f"{self.base_url}/predict", json=payload, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            print(f"✓ Batch prediction successful")
            print(f"  Predictions: {data['predictions']}")
            print(f"  Batch size: {len(data['predictions'])}")

            return (
                "predictions" in data
                and "probabilities" in data
                and len(data["predictions"]) == 5
            )
        except Exception as e:
            print(f"✗ Batch prediction failed: {e}")
            return False

    def test_invalid_input(self) -> bool:
        """Test error handling with invalid input"""
        try:
            # Wrong number of features
            payload = {"features": [[1.0, 2.0, 3.0]]}  # Only 3 features instead of 20

            response = self.session.post(
                f"{self.base_url}/predict", json=payload, timeout=10
            )

            # Should return 400 Bad Request
            if response.status_code == 400:
                print("✓ Error handling works correctly for invalid input")
                return True
            else:
                print(f"✗ Expected 400, got {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            return False

    def test_load(self, n_requests: int = 10) -> bool:
        """Test load handling"""
        try:
            print(f"Running load test with {n_requests} requests...")

            features = self.generate_test_features(1)
            payload = {"features": features}

            success_count = 0
            start_time = time.time()

            for i in range(n_requests):
                try:
                    response = self.session.post(
                        f"{self.base_url}/predict", json=payload, timeout=10
                    )
                    if response.status_code == 200:
                        success_count += 1
                except Exception:
                    pass

            end_time = time.time()
            duration = end_time - start_time

            success_rate = success_count / n_requests
            rps = n_requests / duration

            print(f"✓ Load test completed:")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Requests per second: {rps:.2f}")
            print(f"  Duration: {duration:.2f}s")

            return success_rate > 0.8  # 80% success rate threshold

        except Exception as e:
            print(f"✗ Load test failed: {e}")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests"""
        print("=" * 50)
        print("ML Model Deployment Test Suite")
        print("=" * 50)

        tests = [
            ("Health Check", self.test_health),
            ("Readiness Check", self.test_readiness),
            ("Model Info", self.test_model_info),
            ("Single Prediction", self.test_prediction_single),
            ("Batch Prediction", self.test_prediction_batch),
            ("Error Handling", self.test_invalid_input),
            ("Load Test", lambda: self.test_load(20)),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n--- {test_name} ---")
            if test_func():
                passed += 1
            else:
                print(f"FAILED: {test_name}")

        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} tests passed")
        print("=" * 50)

        return passed == total


def main():

    parser = argparse.ArgumentParser(description="Test ML model deployment")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the ML model service",
    )
    parser.add_argument(
        "--wait", type=int, default=0, help="Wait time in seconds before starting tests"
    )

    args = parser.parse_args()

    if args.wait > 0:
        print(f"Waiting {args.wait} seconds before starting tests...")
        time.sleep(args.wait)

    tester = ModelTester(args.url)
    success = tester.run_all_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
