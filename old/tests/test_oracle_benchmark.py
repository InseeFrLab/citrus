from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
import requests

from src.bodacc.api import BodaccFetchError, bodacc_api
from src.modele.benchmark import (
    JOIN_KEY,
    compare_predictions,
    summarize_metrics,
)
import src.modele.oracle_benchmark as oracle_module
from src.modele.oracle_benchmark import (
    BodaccLookupResolutionError,
    build_argument_parser,
    load_annotation_dataset,
    resolve_bodacc_announcement_id,
    run_oracle_benchmark,
    select_oracle_annotations,
)


def annotation(
    operation_type="VE",
    *,
    issue=147,
    number=853,
    **overrides,
):
    prefix = f"A2023{issue:04d}"
    row = {
        "id_operation": issue,
        "ref_annonce": f"RCS-A_BX{prefix}",
        "numero_annonce": number,
        "ref_annonce_complet": f"{prefix}{number}",
        "type_op": operation_type,
        "siren_cedante": 123456782,
        "siren_beneficiaire": 732829320,
        "date_creation_op": datetime(2026, 8, 24, 12, 0),
        "date_effet_comptable_op": datetime(2023, 7, 11, 9, 30),
        "date_realisation_juridique_op": None,
        "montant": 155.0 if operation_type == "VE" else None,
    }
    row.update(overrides)
    return row


def extracted(operation_type, **overrides):
    row = {
        "anneeCampagne": 2023,
        "typeOperation": operation_type,
        "sirenCedant": "123456782",
        "raisonSocialeCedant": "CEDANT",
        "sirenBeneficiaire": "732829320",
        "raisonSocialeBeneficiaire": "BENEFICIAIRE",
        "dateEffetComptable": "2023-07-11",
        "dateRealisationJuridique": None,
        "montantNet": 155 if operation_type == "VE" else None,
        "source": "https://www.bodacc.fr/annonce/test",
    }
    row.update(overrides)
    return row


class RecordingSkill:
    def __init__(self, operation_type, *, fail_ids=(), **result_overrides):
        self.operation_type = operation_type
        self.calls = []
        self.fail_ids = set(fail_ids)
        self.result_overrides = result_overrides

    def extract(self, announcement):
        self.calls.append(announcement)
        if announcement["id"] in self.fail_ids:
            raise RuntimeError("synthetic skill failure")
        return extracted(self.operation_type, **self.result_overrides)


def successful_fetch(announcement_id):
    return {"id": announcement_id, "source_fact": "only fetched data"}


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, json_error=None):
        self.payload = payload
        self.status_code = status_code
        self.json_error = json_error

    def json(self):
        if self.json_error is not None:
            raise self.json_error
        return self.payload


class BodaccFetchTest(unittest.TestCase):
    def setUp(self):
        self.client = bodacc_api()
        self.announcement_id = "A20230147853"

    def assert_fetch_code(self, response, code):
        with patch("src.bodacc.api.requests.get", return_value=response):
            with self.assertRaises(BodaccFetchError) as raised:
                self.client.fetch_annonce_json(self.announcement_id)
        self.assertEqual(raised.exception.code, code)

    def test_success_queries_two_and_requires_exact_result_id(self):
        payload = {
            "total_count": 1,
            "results": [{"id": self.announcement_id, "dateparution": "2023-08-02"}],
        }
        with patch(
            "src.bodacc.api.requests.get",
            return_value=FakeResponse(payload),
        ) as request:
            result = self.client.fetch_annonce_json(
                self.announcement_id, timeout=4.0
            )

        self.assertEqual(result["id"], self.announcement_id)
        self.assertEqual(request.call_args.kwargs["params"]["limit"], 2)
        self.assertEqual(
            request.call_args.kwargs["params"]["where"],
            f'id="{self.announcement_id}"',
        )
        self.assertEqual(request.call_args.kwargs["timeout"], 4.0)

    def test_campaign_siren_search_is_validated_scoped_and_sorted(self):
        payload = {
            "total_count": 2,
            "results": [
                {"id": "A202302271164"},
                {"id": "A202302241697"},
            ],
        }
        with patch(
            "src.bodacc.api.requests.get",
            return_value=FakeResponse(payload),
        ) as request:
            result = self.client.search_acte_siren_for_year(
                "380622332", 2023, timeout=4.0
            )

        self.assertEqual(
            [row["id"] for row in result],
            ["A202302241697", "A202302271164"],
        )
        params = request.call_args.kwargs["params"]
        self.assertIn('search(acte, "380622332")', params["where"])
        self.assertIn("date'2023-01-01'", params["where"])
        self.assertIn("date'2024-01-01'", params["where"])
        self.assertEqual(params["limit"], 100)
        self.assertEqual(request.call_args.kwargs["timeout"], 4.0)

        for siren, year in (("123", 2023), ("380622332", 1800)):
            with self.subTest(siren=siren, year=year):
                with self.assertRaises(ValueError):
                    self.client.search_acte_siren_for_year(siren, year)

    def test_network_exception_is_categorized(self):
        with patch(
            "src.bodacc.api.requests.get",
            side_effect=requests.exceptions.ConnectionError("offline"),
        ):
            with self.assertRaises(BodaccFetchError) as raised:
                self.client.fetch_annonce_json(self.announcement_id)
        self.assertEqual(raised.exception.code, "network_exception")

    def test_non_2xx_and_invalid_json_are_categorized(self):
        self.assert_fetch_code(FakeResponse({}, status_code=503), "http_error")
        self.assert_fetch_code(
            FakeResponse(json_error=ValueError("bad json")), "invalid_json"
        )

    def test_zero_and_multiple_results_are_categorized(self):
        for total_count in (0, 1):
            with self.subTest(total_count=total_count):
                self.assert_fetch_code(
                    FakeResponse({"total_count": total_count, "results": []}),
                    "zero_results",
                )
        self.assert_fetch_code(
            FakeResponse(
                {
                    "total_count": 2,
                    "results": [
                        {"id": self.announcement_id},
                        {"id": self.announcement_id},
                    ],
                }
            ),
            "multiple_results",
        )

    def test_malformed_payload_and_mismatched_id_are_categorized(self):
        for payload in (
            [],
            {},
            {"results": ["not-an-object"]},
            {"results": [{"id": "A20230001001"}]},
        ):
            with self.subTest(payload=payload):
                self.assert_fetch_code(
                    FakeResponse(payload), "malformed_payload"
                )


class OracleBenchmarkTest(unittest.TestCase):
    def _write_annotations(self, directory, rows, name="annotations.parquet"):
        path = Path(directory) / name
        pl.DataFrame(rows).write_parquet(path)
        return path

    def _run(self, directory, rows, **overrides):
        path = self._write_annotations(directory, rows)
        output = Path(directory) / "artifacts"
        skills = overrides.pop(
            "skills",
            {
                "VE": RecordingSkill("VE"),
                "LG": RecordingSkill("LG"),
                "TP": RecordingSkill("TP"),
            },
        )
        result = run_oracle_benchmark(
            path,
            output,
            fetch_announcement=overrides.pop(
                "fetch_announcement", successful_fetch
            ),
            skills=skills,
            run_timestamp=datetime(2026, 8, 24, 18, 0, tzinfo=UTC),
            git_commit="test-commit",
            **overrides,
        )
        return result, output, skills

    def test_parquet_loader_excludes_creation_date_entirely(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_annotations(
                directory,
                [annotation(date_creation_op=datetime(1999, 1, 1))],
            )
            loaded = load_annotation_dataset(path)

        self.assertNotIn("date_creation_op", loaded.columns)
        self.assertNotIn("id_operation", loaded.columns)
        self.assertEqual(loaded.height, 1)

    def test_tp_is_supported_but_default_limit_keeps_it_unselected(self):
        rows = [
            annotation("TP", issue=150),
            annotation("LG", issue=149),
            annotation("VE", issue=148),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows)

        self.assertEqual(
            result.selected_annotations["type_op"].to_list(), ["VE", "LG"]
        )
        coverage = result.summary["coverage"]
        self.assertEqual(coverage["tp_rows_available"], 1)
        self.assertEqual(coverage["by_type"]["TP"]["selected_rows"], 0)
        self.assertEqual(coverage["other_types_skipped"], 0)

    def test_sampling_is_deterministic_per_type_and_supports_full(self):
        frame = pl.DataFrame(
            [
                annotation("LG", issue=153),
                annotation("VE", issue=152),
                annotation("VE", issue=151),
                annotation("LG", issue=150),
            ]
        ).select(oracle_module.ANNOTATION_COLUMNS)

        first = select_oracle_annotations(frame, max_ve=1, max_lg=1)
        second = select_oracle_annotations(frame.reverse(), max_ve=1, max_lg=1)
        full = select_oracle_annotations(frame, max_ve=None, max_lg=None)

        self.assertEqual(
            first["ref_annonce_complet"].to_list(),
            second["ref_annonce_complet"].to_list(),
        )
        self.assertEqual(first.height, 2)
        self.assertEqual(full.height, 4)

    def test_tp_sampling_is_deterministic_and_supports_full(self):
        frame = pl.DataFrame(
            [
                annotation("TP", issue=153),
                annotation("TP", issue=151),
                annotation("TP", issue=152),
            ]
        ).select(oracle_module.ANNOTATION_COLUMNS)

        first = select_oracle_annotations(
            frame, max_ve=0, max_lg=0, max_tp=2
        )
        second = select_oracle_annotations(
            frame.reverse(), max_ve=0, max_lg=0, max_tp=2
        )
        full = select_oracle_annotations(
            frame, max_ve=0, max_lg=0, max_tp=None
        )

        self.assertEqual(
            first["ref_annonce_complet"].to_list(),
            second["ref_annonce_complet"].to_list(),
        )
        self.assertEqual(first.height, 2)
        self.assertEqual(full.height, 3)

    def test_tp_only_calls_only_tp_skill(self):
        skills = {
            "VE": RecordingSkill("VE"),
            "LG": RecordingSkill("LG"),
            "TP": RecordingSkill("TP"),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [
                    annotation("VE"),
                    annotation("LG", issue=148),
                    annotation("TP", issue=149),
                ],
                max_ve=0,
                max_lg=0,
                max_tp=50,
                skills=skills,
            )

        self.assertEqual(skills["VE"].calls, [])
        self.assertEqual(skills["LG"].calls, [])
        self.assertEqual(len(skills["TP"].calls), 1)
        self.assertEqual(result.predictions["oracle_type"].to_list(), ["TP"])
        self.assertEqual(
            result.summary["coverage"]["by_type"]["TP"][
                "successful_predictions"
            ],
            1,
        )

    def test_lg_only_zero_ve_never_calls_ve_skill(self):
        skills = {"VE": RecordingSkill("VE"), "LG": RecordingSkill("LG")}
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [annotation("VE"), annotation("LG", issue=148)],
                max_ve=0,
                max_lg=50,
                skills=skills,
            )

        self.assertEqual(skills["VE"].calls, [])
        self.assertEqual(len(skills["LG"].calls), 1)
        self.assertEqual(result.predictions["oracle_type"].to_list(), ["LG"])

    def test_oracle_calls_each_public_boundary_with_fetched_payload_only(self):
        skills = {"VE": RecordingSkill("VE"), "LG": RecordingSkill("LG")}
        rows = [annotation("VE"), annotation("LG", issue=148)]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, skills=skills)

        for operation_type in ("VE", "LG"):
            self.assertEqual(len(skills[operation_type].calls), 1)
            self.assertEqual(
                set(skills[operation_type].calls[0]), {"id", "source_fact"}
            )
            self.assertNotIn("siren_cedante", skills[operation_type].calls[0])
        self.assertEqual(
            result.predictions["typeOperation"].to_list(), ["VE", "LG"]
        )

    def test_reference_mapping_is_direct_and_inconsistency_is_not_guessed(self):
        row = annotation()
        self.assertEqual(
            resolve_bodacc_announcement_id(row), "A20230147853"
        )
        row["ref_annonce_complet"] = "A20230147854"
        with self.assertRaises(BodaccLookupResolutionError):
            resolve_bodacc_announcement_id(row)
        row = annotation(ref_annonce="unexpected-format")
        with self.assertRaises(BodaccLookupResolutionError):
            resolve_bodacc_announcement_id(row)

    def test_unresolved_reference_is_recorded_and_missing_in_benchmark(self):
        rows = [
            annotation(ref_annonce="unresolved"),
            annotation("LG", issue=148),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows)

        coverage = result.summary["coverage"]
        self.assertEqual(coverage["lookup_resolution_failures"], 1)
        self.assertEqual(coverage["successful_predictions"], 1)
        self.assertEqual(coverage["benchmark"]["missing_predictions"], 1)
        self.assertEqual(
            result.errors.filter(
                pl.col("failure_stage") == "lookup_resolution"
            ).height,
            1,
        )

    def test_null_join_key_is_reported_without_aborting_valid_rows(self):
        fetch_calls = []

        def fetch(announcement_id):
            fetch_calls.append(announcement_id)
            return successful_fetch(announcement_id)

        rows = [
            annotation(ref_annonce_complet=None),
            annotation("LG", issue=148),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, rows, fetch_announcement=fetch
            )

        coverage = result.summary["coverage"]
        self.assertEqual(result.selected_annotations.height, 2)
        self.assertEqual(result.benchmark_annotations.height, 1)
        self.assertEqual(coverage["benchmark_eligible_rows"], 1)
        self.assertEqual(coverage["invalid_join_key_failures"], 1)
        self.assertEqual(coverage["successful_predictions"], 1)
        self.assertEqual(coverage["benchmark"]["annotated_rows"], 1)
        self.assertEqual(coverage["benchmark"]["missing_predictions"], 0)
        self.assertEqual(fetch_calls, [annotation("LG", issue=148)[JOIN_KEY]])
        invalid = result.errors.filter(
            pl.col("failure_code") == "invalid_join_key"
        )
        self.assertEqual(invalid.height, 1)
        self.assertIsNone(invalid[JOIN_KEY][0])

    def test_empty_join_key_is_an_observable_invalid_key(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [
                    annotation(ref_annonce_complet="   "),
                    annotation("LG", issue=148),
                ],
            )

        self.assertEqual(result.benchmark_annotations.height, 1)
        self.assertEqual(
            result.summary["coverage"]["invalid_join_key_failures"], 1
        )

    def test_duplicate_join_keys_exclude_every_ambiguous_row(self):
        fetch_calls = []

        def fetch(announcement_id):
            fetch_calls.append(announcement_id)
            return successful_fetch(announcement_id)

        duplicate = annotation()
        rows = [
            duplicate,
            {**duplicate, "id_operation": 999},
            annotation("LG", issue=148),
        ]
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, rows, fetch_announcement=fetch
            )

        coverage = result.summary["coverage"]
        self.assertEqual(result.selected_annotations.height, 3)
        self.assertEqual(result.benchmark_annotations.height, 1)
        self.assertEqual(coverage["duplicate_join_key_failures"], 2)
        self.assertEqual(coverage["lookup_resolution_failures"], 2)
        self.assertEqual(coverage["successful_predictions"], 1)
        self.assertEqual(coverage["benchmark"]["annotated_rows"], 1)
        duplicate_errors = result.errors.filter(
            pl.col("failure_code") == "duplicate_join_key"
        )
        self.assertEqual(duplicate_errors.height, 2)
        self.assertEqual(
            duplicate_errors[JOIN_KEY].to_list(),
            [duplicate[JOIN_KEY], duplicate[JOIN_KEY]],
        )
        self.assertEqual(fetch_calls, [annotation("LG", issue=148)[JOIN_KEY]])

    def test_fetch_failure_is_recorded_and_next_row_continues(self):
        failing_id = annotation()["ref_annonce_complet"]

        def fetch(announcement_id):
            if announcement_id == failing_id:
                raise BodaccFetchError("network_exception", "offline")
            return successful_fetch(announcement_id)

        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [annotation(), annotation("LG", issue=148)],
                fetch_announcement=fetch,
            )

        self.assertEqual(result.summary["coverage"]["bodacc_fetch_failures"], 1)
        self.assertEqual(result.predictions["oracle_type"].to_list(), ["LG"])
        self.assertEqual(result.benchmark_annotations.height, 2)
        self.assertEqual(
            result.summary["coverage"]["benchmark"]["missing_predictions"], 1
        )
        self.assertEqual(
            result.summary["coverage"]["by_type"]["VE"][
                "bodacc_fetch_failures"
            ],
            1,
        )

    def test_skill_exception_is_recorded_and_next_row_continues(self):
        failing_id = annotation()["ref_annonce_complet"]
        skills = {
            "VE": RecordingSkill("VE", fail_ids={failing_id}),
            "LG": RecordingSkill("LG"),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [annotation(), annotation("LG", issue=148)],
                skills=skills,
            )

        self.assertEqual(
            result.summary["coverage"]["skill_execution_failures"], 1
        )
        self.assertEqual(result.predictions["oracle_type"].to_list(), ["LG"])
        self.assertEqual(result.benchmark_annotations.height, 2)
        self.assertEqual(
            result.summary["coverage"]["benchmark"]["missing_predictions"], 1
        )

    def test_success_attaches_key_and_uses_generic_benchmark_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(
                oracle_module,
                "compare_predictions",
                wraps=compare_predictions,
            ) as compare_mock, patch.object(
                oracle_module,
                "summarize_metrics",
                wraps=summarize_metrics,
            ) as summarize_mock:
                result, _, _ = self._run(directory, [annotation()])

        self.assertEqual(
            result.predictions["ref_annonce_complet"].to_list(),
            ["A20230147853"],
        )
        compare_mock.assert_called_once()
        self.assertEqual(
            compare_mock.call_args.kwargs["amount_tolerance"], 0.1
        )
        summarize_mock.assert_called_once_with(result.comparison)

    def test_amount_tolerance_can_override_generic_default(self):
        skills = {
            "VE": RecordingSkill("VE", montantNet=154.8),
            "LG": RecordingSkill("LG"),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [annotation()],
                amount_tolerance=0.25,
                skills=skills,
            )

        self.assertTrue(result.comparison.rows["montantNet_correct"][0])
        self.assertEqual(
            result.summary["metadata"]["amount_tolerance_keur"], 0.25
        )

    def test_field_failures_are_inspectable_in_errors(self):
        skills = {
            "VE": RecordingSkill("VE", sirenCedant=None),
            "LG": RecordingSkill("LG"),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, [annotation()], skills=skills
            )

        benchmark_error = result.errors.filter(
            pl.col("failure_stage") == "benchmark"
        ).row(0, named=True)
        self.assertIn("sirenCedant", benchmark_error["failing_fields"])
        self.assertEqual(
            benchmark_error["expected_sirenCedant"], "123456782"
        )
        self.assertIsNone(benchmark_error["predicted_sirenCedant"])
        self.assertFalse(benchmark_error["sirenCedant_correct"])

    def test_outputs_and_machine_readable_summary_are_written(self):
        rows = [annotation(), annotation("LG", issue=148), annotation("TP", issue=149)]
        with tempfile.TemporaryDirectory() as directory:
            result, output, _ = self._run(directory, rows)
            for filename in (
                "predictions.parquet",
                "comparison.parquet",
                "errors.parquet",
                "summary.json",
            ):
                self.assertTrue((output / filename).is_file())
            persisted = json.loads((output / "summary.json").read_text())
            persisted_predictions = pl.read_parquet(
                output / "predictions.parquet"
            )

        self.assertEqual(persisted["metadata"]["selection_mode"], "reference_type_oracle")
        self.assertEqual(persisted["coverage"]["by_type"]["VE"]["successful_predictions"], 1)
        self.assertEqual(persisted["coverage"]["by_type"]["LG"]["successful_predictions"], 1)
        self.assertEqual(persisted_predictions.height, result.predictions.height)
        self.assertNotIn("date_creation_op", persisted_predictions.columns)

    def test_cli_accepts_lg_only_full_and_tp_limits(self):
        parser = build_argument_parser()
        args = parser.parse_args(
            [
                "--annotations",
                "input.parquet",
                "--output-dir",
                "artifacts",
                "--max-ve",
                "0",
                "--max-lg",
                "all",
                "--max-tp",
                "50",
            ]
        )
        self.assertEqual(args.max_ve, 0)
        self.assertIsNone(args.max_lg)
        self.assertEqual(args.max_tp, 50)


if __name__ == "__main__":
    unittest.main()
