import unittest
import os

from app import app


class CropYieldAppTest(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_history_page(self):
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)

    def test_dashboard_page(self):
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)

    def test_model_performance_page(self):
        response = self.client.get("/model-performance")
        self.assertEqual(response.status_code, 200)

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.exists("crop_yield_model.pkl")
        )

    def test_metrics_file_exists(self):
        self.assertTrue(
            os.path.exists("model_metrics.csv")
        )

    def test_prediction(self):

        response = self.client.post(
            "/",
            data={
                "year": "2026",
                "state": "Uttar Pradesh",
                "crop": "Wheat",
                "season": "Rabi",
                "area": "1000",
                "rainfall": "800",
                "fertilizer": "50000",
                "pesticide": "1000"
            }
        )

        self.assertEqual(response.status_code, 200)

        self.assertIn(
            b"Prediction Result",
            response.data
        )


if __name__ == "__main__":
    unittest.main()