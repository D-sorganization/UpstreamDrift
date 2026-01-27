import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.assess_repository import assess_J
from scripts.generate_assessment_summary import extract_score_from_report


class TestAssessmentScripts(unittest.TestCase):

    def test_extract_score_from_report(self):
        # Create a dummy report file
        dummy_report = Path("dummy_report.md")
        dummy_report.write_text(
            "# Assessment: Test\n\n**Grade**: 8.5/10\n", encoding="utf-8"
        )
        try:
            score = extract_score_from_report(dummy_report)
            self.assertEqual(score, 8.5)
        finally:
            if dummy_report.exists():
                dummy_report.unlink()

    def test_assess_J_logic(self):
        # We can't easily mock the file system for the whole function without heavy mocking,
        # but we can verify that the function runs without error and returns a report path.
        # This assumes REPO_ROOT is set correctly in the imported module.

        # We need to mock generate_markdown_report to avoid writing files
        with patch("scripts.assess_repository.generate_markdown_report") as mock_gen:
            mock_gen.return_value = Path("dummy_output.md")

            # We also need to mock grep_count to avoid scanning the whole repo
            with patch("scripts.assess_repository.grep_count") as mock_grep:
                mock_grep.return_value = 1

                result = assess_J()
                self.assertIsInstance(result, Path)
                self.assertEqual(str(result), "dummy_output.md")

                # Verify that generate_markdown_report was called with expected arguments
                # The score should be 7.5 (default) + potentially adjustments
                # In current state, if src/api exists (which it does), score is 7.5.
                # If grep_count finds FastAPI (mocked to 1), it appends "FastAPI usage detected".

                args, _ = mock_gen.call_args
                category_id = args[0]
                score = args[2]
                self.assertEqual(category_id, "J")
                # We expect 7.5 if api or src/api exists.
                # Since we are running in the actual repo, src/api exists.
                self.assertEqual(score, 7.5)


if __name__ == "__main__":
    unittest.main()
