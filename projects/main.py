import argparse
import json
from pathlib import Path

from pipeline.evaluator import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM job-fit pipeline runner")
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Run only one test case by id (e.g., tc_001).",
    )
    parser.add_argument(
        "--show-passed",
        action="store_true",
        help="Print passed cases too.",
    )
    args = parser.parse_args()

    test_file = Path(__file__).parent / "eval" / "test_cases.json"
    with test_file.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    selected_cases = cases
    if args.case_id:
        selected_cases = [c for c in cases if c["id"] == args.case_id]
        if not selected_cases:
            raise ValueError(f"Case id not found: {args.case_id}")

    total = len(selected_cases)
    passed = 0

    for case in selected_cases:
        result = run_pipeline(
            user_input=case["user_input"],
            job_description=case["job_description"],
            resume_text=case["resume"],
        )

        ok = result["status"] == "ok"
        passed += 1 if ok else 0

        if ok and not args.show_passed:
            continue

        print("=" * 80)
        print(f"CASE: {case['id']}")
        print(f"EXPECTED ROUTE: {case['expected_route']}")
        print(f"PIPELINE STATUS: {result['status']}")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    print("=" * 80)
    print(f"SUMMARY: {passed}/{total} cases returned status=ok")


if __name__ == "__main__":
    main()
