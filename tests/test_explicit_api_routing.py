import unittest

from app.services.heart_agent import HeartMRIAgent


class ExplicitApiRoutingTest(unittest.TestCase):
    def test_routes_report_quick_prompt(self):
        self.assertEqual(
            HeartMRIAgent._get_explicit_api_request(
                "Please generate a cardiac function evaluation report"
            ),
            "Medical Report Generation",
        )

    def test_routes_chinese_report_request(self):
        self.assertEqual(
            HeartMRIAgent._get_explicit_api_request("请生成一份心功能评估报告"),
            "Medical Report Generation",
        )

    def test_routes_explicit_metrics_request(self):
        self.assertEqual(
            HeartMRIAgent._get_explicit_api_request(
                "Please calculate cardiac metrics from these scans"
            ),
            "Cardiac Metrics Calculation",
        )

    def test_leaves_visual_question_to_agent(self):
        self.assertIsNone(
            HeartMRIAgent._get_explicit_api_request(
                "Can you describe the ventricular morphology?"
            )
        )

    def test_infers_explicit_sequence_filenames(self):
        cases = {
            "2CH.zip": "cine 2ch",
            "4CH.zip": "cine 4ch",
            "SAX.zip": "cine sa",
            "LGE.zip": "lge sa",
            "cine_short-axis.nii.gz": "cine sa",
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(
                    HeartMRIAgent._get_filename_sequence(filename), expected
                )

    def test_does_not_guess_ambiguous_filename(self):
        self.assertEqual(
            HeartMRIAgent._get_filename_sequence("patient_001.zip"), "unknown"
        )


if __name__ == "__main__":
    unittest.main()
