from datetime import UTC, datetime
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from src.bodacc.api import BodaccFetchError
from src.modele.benchmark import JOIN_KEY
import src.modele.routing_benchmark as routing_benchmark_module
from src.modele.routing_benchmark import (
    FINAL_TYPES,
    PREDICTION_SCHEMA,
    build_argument_parser,
    load_routing_annotations,
    reference_family_for_type,
    run_routing_benchmark,
    select_routing_annotations,
    summarize_routing_metrics,
)
from src.routing import (
    FamilyRouter,
    RoutingFamily,
    RoutingLLMError,
    RoutingOutputError,
    RoutingResult,
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
        JOIN_KEY: f"{prefix}{number}",
        "type_op": operation_type,
        "siren_cedante": "ANNOTATED_CEDANT_SENTINEL",
        "siren_beneficiaire": "ANNOTATED_BENEFICIARY_SENTINEL",
        "date_creation_op": "ANNOTATED_CREATION_DATE_SENTINEL",
        "date_effet_comptable_op": "ANNOTATED_EFFECT_DATE_SENTINEL",
        "montant": "ANNOTATED_AMOUNT_SENTINEL",
    }
    row.update(overrides)
    return row


def fetched_announcement(announcement_id):
    return {
        "id": announcement_id,
        "registre": "RCS-A",
        "acte": {
            "vente": {
                "descriptif": f"Cession synthétique {announcement_id}"
            }
        },
        "type_op": "FETCHED_REFERENCE_SENTINEL",
        "reference_family": "FETCHED_EXPECTED_SENTINEL",
        "unrelated": "FETCHED_RAW_SENTINEL",
    }


class RecordingRouter:
    def __init__(self, families=None, failures=None):
        self.families = families or {}
        self.failures = failures or {}
        self.calls = []

    def route(self, announcement):
        self.calls.append(announcement)
        announcement_id = announcement["id"]
        if announcement_id in self.failures:
            raise self.failures[announcement_id]
        family = self.families.get(announcement_id, RoutingFamily.UNKNOWN)
        return RoutingResult(
            family=family,
            evidence=(f"indice {announcement_id}",),
            reason=f"raison {announcement_id}",
        )


class RecordingAsk:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def json_response(family):
    return json.dumps(
        {
            "family": family,
            "evidence": ["indice source"],
            "reason": "Justification source concise.",
        }
    )


class RoutingBenchmarkTest(unittest.TestCase):
    def _write_annotations(self, directory, rows, name="annotations.parquet"):
        path = Path(directory) / name
        pl.DataFrame(rows).write_parquet(path)
        return path

    def _run(self, directory, rows, **overrides):
        path = self._write_annotations(directory, rows)
        output = Path(directory) / "artifacts"
        router = overrides.pop("router", RecordingRouter())
        result = run_routing_benchmark(
            path,
            output,
            fetch_announcement=overrides.pop(
                "fetch_announcement", fetched_announcement
            ),
            router=router,
            run_timestamp=datetime(2026, 8, 27, 15, 0, tzinfo=UTC),
            git_commit="test-commit",
            model_name="fake-offline-model",
            **overrides,
        )
        return result, output, router

    def test_exact_reference_mapping_and_unsupported_types(self):
        expected = {
            "VE": RoutingFamily.VE,
            "LG": RoutingFamily.LG,
            "TP": RoutingFamily.TP,
            "FU": RoutingFamily.FUSION_FAMILY,
            "AB": RoutingFamily.FUSION_FAMILY,
            "SP": RoutingFamily.FUSION_FAMILY,
            "ST": RoutingFamily.FUSION_FAMILY,
            "AP": RoutingFamily.FUSION_FAMILY,
        }
        self.assertEqual(
            {kind: reference_family_for_type(kind) for kind in FINAL_TYPES},
            expected,
        )
        for unsupported in (None, "", "TUP", "UNKNOWN", "ve"):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises(ValueError):
                    reference_family_for_type(unsupported)

    def test_loader_projects_only_lookup_and_reference_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_annotations(directory, [annotation()])
            loaded = load_routing_annotations(path)

        self.assertEqual(
            loaded.columns,
            ["ref_annonce", "numero_annonce", JOIN_KEY, "type_op"],
        )
        for excluded in (
            "date_creation_op",
            "siren_cedante",
            "siren_beneficiaire",
            "date_effet_comptable_op",
            "montant",
        ):
            self.assertNotIn(excluded, loaded.columns)

    def test_sampling_is_deterministic_by_each_final_type_and_supports_all(self):
        rows = []
        for index, operation_type in enumerate(FINAL_TYPES):
            rows.extend(
                [
                    annotation(operation_type, issue=200 + index * 2 + 1),
                    annotation(operation_type, issue=200 + index * 2),
                ]
            )
        frame = pl.DataFrame(rows).select(
            routing_benchmark_module.ANNOTATION_COLUMNS
        )

        first = select_routing_annotations(frame, max_per_type=1)
        second = select_routing_annotations(frame.reverse(), max_per_type=1)
        full = select_routing_annotations(frame, max_per_type=None)

        self.assertEqual(first[JOIN_KEY].to_list(), second[JOIN_KEY].to_list())
        self.assertEqual(first["type_op"].to_list(), list(FINAL_TYPES))
        self.assertEqual(first.height, 8)
        self.assertEqual(full.height, 16)

    def test_unsupported_reference_rows_are_reported_not_routed(self):
        rows = [
            annotation("VE"),
            annotation("", issue=148),
            annotation(None, issue=149),
            annotation("TUP", issue=150),
        ]
        router = RecordingRouter(
            {annotation("VE")[JOIN_KEY]: RoutingFamily.VE}
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, router=router)

        self.assertEqual(result.selected_annotations.height, 1)
        self.assertEqual(len(router.calls), 1)
        unsupported = result.errors.filter(
            pl.col("failure_code") == "unsupported_reference_type"
        )
        self.assertEqual(unsupported.height, 3)
        self.assertEqual(
            result.summary["coverage"]["unsupported_reference_rows"], 3
        )

    def test_invalid_and_duplicate_join_keys_are_observable_and_excluded(self):
        duplicate = annotation("VE")
        rows = [
            duplicate,
            {**duplicate, "id_operation": 999},
            annotation("LG", issue=148, ref_annonce_complet=None),
            annotation("TP", issue=149),
        ]
        tp_id = annotation("TP", issue=149)[JOIN_KEY]
        router = RecordingRouter({tp_id: RoutingFamily.TP})

        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, router=router)

        self.assertEqual(result.selected_annotations.height, 4)
        self.assertEqual(result.benchmark_annotations.height, 1)
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(
            result.summary["coverage"]["invalid_join_key_failures"], 1
        )
        self.assertEqual(
            result.summary["coverage"]["duplicate_join_key_failures"], 2
        )

    def test_lookup_and_fetch_failures_remain_in_metric_denominator(self):
        bad_lookup = annotation("VE", ref_annonce="bad-reference")
        failing_fetch = annotation("LG", issue=148)
        succeeding = annotation("TP", issue=149)

        def fetch(announcement_id):
            if announcement_id == failing_fetch[JOIN_KEY]:
                raise BodaccFetchError("network_exception", "offline")
            return fetched_announcement(announcement_id)

        router = RecordingRouter(
            {succeeding[JOIN_KEY]: RoutingFamily.TP}
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [bad_lookup, failing_fetch, succeeding],
                router=router,
                fetch_announcement=fetch,
            )

        metrics = result.summary["metrics"]
        self.assertEqual(metrics["eligible_reference_rows"], 3)
        self.assertEqual(metrics["successful_valid_router_outputs"], 1)
        self.assertEqual(metrics["technical_failures"], 2)
        self.assertEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(
            metrics["confusion_matrix"]["VE"]["__ERROR__"], 1
        )
        self.assertEqual(
            metrics["confusion_matrix"]["LG"]["__ERROR__"], 1
        )

    def test_semantic_unknown_and_invalid_llm_output_stay_distinct(self):
        rows = [annotation("VE"), annotation("LG", issue=148)]
        fake_ask = RecordingAsk(
            [json_response("UNKNOWN"), "not-json"]
        )
        router = FamilyRouter(fake_ask)

        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, router=router)

        self.assertEqual(result.predictions.height, 1)
        self.assertEqual(
            result.predictions["predicted_family"].to_list(), ["UNKNOWN"]
        )
        metrics = result.summary["metrics"]
        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["technical_failures"], 1)
        self.assertEqual(metrics["technical_router_failures"], 1)
        self.assertEqual(
            metrics["confusion_matrix"]["VE"]["UNKNOWN"], 1
        )
        self.assertEqual(
            metrics["confusion_matrix"]["LG"]["__ERROR__"], 1
        )
        self.assertEqual(
            set(result.errors["failure_code"].to_list()),
            {"semantic_unknown", "invalid_json"},
        )
        self.assertEqual(fake_ask.calls[0][1], {"temperature": 0})
        self.assertEqual(fake_ask.calls[1][1], {"temperature": 0})

    def test_llm_execution_and_output_validation_have_separate_stages(self):
        ve = annotation("VE")
        lg = annotation("LG", issue=148)
        router = RecordingRouter(
            failures={
                ve[JOIN_KEY]: RoutingLLMError(
                    "ConnectionError", "LLM routing call failed"
                ),
                lg[JOIN_KEY]: RoutingOutputError(
                    "invalid_family", "Unsupported family"
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, [ve, lg], router=router)

        stages = set(result.errors["failure_stage"].to_list())
        self.assertEqual(stages, {"llm_execution", "llm_output_validation"})
        self.assertEqual(
            result.summary["failures"]["technical_by_stage"],
            {"llm_execution": 1, "llm_output_validation": 1},
        )

    def test_metrics_include_balance_abstention_selective_and_confusion(self):
        rows = [
            annotation("VE", issue=201),
            annotation("LG", issue=202),
            annotation("TP", issue=203),
            annotation("FU", issue=204),
            annotation("AB", issue=205),
            annotation("SP", issue=206),
            annotation("ST", issue=207),
            annotation("AP", issue=208),
        ]
        reference = pl.DataFrame(rows).select(
            routing_benchmark_module.ANNOTATION_COLUMNS
        )
        outcomes = {
            "VE": "VE",
            "LG": "UNKNOWN",
            "FU": "FUSION_FAMILY",
            "AB": "VE",
            "SP": "FUSION_FAMILY",
            "ST": "FUSION_FAMILY",
            "AP": "UNKNOWN",
        }
        prediction_rows = []
        for row in rows:
            predicted = outcomes.get(row["type_op"])
            if predicted is None:
                continue
            reference_family = reference_family_for_type(row["type_op"]).value
            prediction_rows.append(
                {
                    JOIN_KEY: row[JOIN_KEY],
                    "reference_type": row["type_op"],
                    "reference_family": reference_family,
                    "predicted_family": predicted,
                    "correct": predicted == reference_family,
                    "reason": "synthetic",
                    "evidence": ["synthetic"],
                }
            )
        predictions = pl.DataFrame(
            prediction_rows, schema=PREDICTION_SCHEMA, strict=False
        )

        metrics = summarize_routing_metrics(
            reference, predictions, technical_router_failures=1
        )

        self.assertEqual(metrics["eligible_reference_rows"], 8)
        self.assertEqual(metrics["successful_valid_router_outputs"], 7)
        self.assertEqual(metrics["technical_failures"], 1)
        self.assertEqual(metrics["technical_router_failures"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["unknown_rate"], 0.25)
        self.assertEqual(metrics["coverage"], 0.625)
        self.assertEqual(metrics["selective_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["macro_recall"], 0.4)
        self.assertAlmostEqual(metrics["macro_f1"], 17 / 48)
        self.assertEqual(
            metrics["per_reference_family"]["FUSION_FAMILY"]["recall"],
            0.6,
        )
        self.assertEqual(
            metrics["per_reference_family"]["LG"][
                "predicted_unknown_count"
            ],
            1,
        )
        self.assertEqual(
            metrics["confusion_matrix"]["TP"]["__ERROR__"], 1
        )
        self.assertEqual(
            metrics["confusion_matrix"]["FUSION_FAMILY"]["VE"], 1
        )
        self.assertEqual(
            metrics["fusion_family_by_final_type"]["AP"][
                "predicted_unknown_count"
            ],
            1,
        )

    def test_runner_calls_only_router_with_fetched_payload_and_writes_artifacts(self):
        rows = [
            annotation(operation_type, issue=300 + index)
            for index, operation_type in enumerate(FINAL_TYPES)
        ]
        families = {
            row[JOIN_KEY]: reference_family_for_type(row["type_op"])
            for row in rows
        }
        router = RecordingRouter(families)

        with tempfile.TemporaryDirectory() as directory:
            result, output, _ = self._run(
                directory, rows, router=router, max_per_type=1
            )
            persisted_summary = json.loads(
                (output / "routing_summary.json").read_text()
            )
            persisted_predictions = pl.read_parquet(
                output / "routing_predictions.parquet"
            )
            persisted_errors = pl.read_parquet(
                output / "routing_errors.parquet"
            )

        self.assertEqual(len(router.calls), 8)
        self.assertEqual(result.predictions.height, 8)
        self.assertEqual(result.errors.height, 0)
        self.assertEqual(persisted_predictions.columns, list(PREDICTION_SCHEMA))
        self.assertEqual(persisted_errors.columns, list(routing_benchmark_module.ERROR_SCHEMA))
        for call in router.calls:
            self.assertEqual(
                set(call),
                {
                    "id",
                    "registre",
                    "acte",
                    "type_op",
                    "reference_family",
                    "unrelated",
                },
            )
            self.assertNotIn("date_creation_op", call)
            self.assertNotIn("siren_cedante", call)
        metadata = persisted_summary["metadata"]
        self.assertEqual(metadata["llm_model_name"], "fake-offline-model")
        self.assertEqual(metadata["router_prompt_version"], "family-router-v1")
        self.assertEqual(
            metadata["routing_taxonomy_version"], "family-routing-v1"
        )
        self.assertEqual(metadata["max_per_type"], 1)
        self.assertEqual(
            metadata["selected_counts_by_final_type"],
            {operation_type: 1 for operation_type in FINAL_TYPES},
        )
        self.assertEqual(
            persisted_summary["metrics"]["accuracy"], 1.0
        )

    def test_benchmark_path_cannot_leak_reference_fields_into_llm_messages(self):
        fake_ask = RecordingAsk([json_response("VE")])
        router = FamilyRouter(fake_ask)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, [annotation("VE")], router=router
            )

        self.assertEqual(result.predictions.height, 1)
        prompt = json.dumps(fake_ask.calls[0][0], ensure_ascii=False)
        for forbidden in (
            "ANNOTATED_CEDANT_SENTINEL",
            "ANNOTATED_BENEFICIARY_SENTINEL",
            "ANNOTATED_CREATION_DATE_SENTINEL",
            "ANNOTATED_EFFECT_DATE_SENTINEL",
            "ANNOTATED_AMOUNT_SENTINEL",
            "FETCHED_REFERENCE_SENTINEL",
            "FETCHED_EXPECTED_SENTINEL",
            "FETCHED_RAW_SENTINEL",
        ):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("Cession synthétique", prompt)

    def test_semantic_errors_retain_expected_predicted_reason_and_evidence(self):
        ve = annotation("VE")
        router = RecordingRouter({ve[JOIN_KEY]: RoutingFamily.LG})
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, [ve], router=router)

        error = result.errors.row(0, named=True)
        self.assertEqual(error["failure_stage"], "routing_evaluation")
        self.assertEqual(error["failure_code"], "misclassification")
        self.assertEqual(error["reference_family"], "VE")
        self.assertEqual(error["predicted_family"], "LG")
        self.assertEqual(error["reason"], f"raison {ve[JOIN_KEY]}")
        self.assertEqual(error["evidence"], [f"indice {ve[JOIN_KEY]}"])

    def test_module_has_no_operation_skill_dependency(self):
        source = inspect.getsource(routing_benchmark_module)
        self.assertNotIn("src.operation", source)
        self.assertNotIn("vente_skill", source)
        self.assertNotIn("location_gerance_skill", source)
        self.assertNotIn("transmission_patrimoine_skill", source)

    def test_cli_has_conservative_default_and_supports_all(self):
        parser = build_argument_parser()
        default = parser.parse_args(
            ["--annotations", "input.parquet", "--output-dir", "artifacts"]
        )
        full = parser.parse_args(
            [
                "--annotations",
                "input.parquet",
                "--output-dir",
                "artifacts",
                "--max-per-type",
                "all",
            ]
        )
        self.assertEqual(default.max_per_type, 10)
        self.assertIsNone(full.max_per_type)


if __name__ == "__main__":
    unittest.main()
