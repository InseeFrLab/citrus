from datetime import UTC, datetime
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl

from src.bodacc.api import BodaccFetchError
from src.modele.benchmark import JOIN_KEY
import src.modele.fusion_subtype_benchmark as fusion_benchmark_module
from src.modele.fusion_subtype_benchmark import (
    ANNOTATION_COLUMNS,
    ERROR_SCHEMA,
    FUSION_TYPES,
    PREDICTION_SCHEMA,
    build_argument_parser,
    load_fusion_annotations,
    reference_subtype_for_type,
    run_fusion_subtype_benchmark,
    select_fusion_annotations,
    summarize_fusion_subtype_metrics,
)
from src.routing.fusion_subtype import (
    BeneficiaryCount,
    BeneficiaryCreation,
    FusionSubtype,
    FusionSubtypeLLMError,
    FusionSubtypeOutputError,
    FusionSubtypeResult,
    FusionSubtypeRouter,
    TransferorFate,
    TransferScope,
)


def annotation(
    operation_type="FU",
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
        "date_realisation_juridique_op": "ANNOTATED_LEGAL_DATE_SENTINEL",
        "montant": "ANNOTATED_AMOUNT_SENTINEL",
    }
    row.update(overrides)
    return row


def fetched_announcement(announcement_id):
    return {
        "id": announcement_id,
        "registre": "RCS-A",
        "acte": {
            "descriptif": f"Fusion synthétique source {announcement_id}"
        },
        "type_op": "FETCHED_REFERENCE_SENTINEL",
        "reference_subtype": "FETCHED_EXPECTED_SENTINEL",
        "unrelated": "FETCHED_RAW_SENTINEL",
    }


_CONSISTENT_AXES = {
    FusionSubtype.FU: (
        TransferScope.TOTAL,
        TransferorFate.DISAPPEARS,
        BeneficiaryCreation.NEW,
        BeneficiaryCount.ONE,
    ),
    FusionSubtype.AB: (
        TransferScope.TOTAL,
        TransferorFate.DISAPPEARS,
        BeneficiaryCreation.EXISTING,
        BeneficiaryCount.ONE,
    ),
    FusionSubtype.SP: (
        TransferScope.PARTIAL,
        TransferorFate.SURVIVES,
        BeneficiaryCreation.NEW,
        BeneficiaryCount.ONE,
    ),
    FusionSubtype.ST: (
        TransferScope.TOTAL,
        TransferorFate.DISAPPEARS,
        BeneficiaryCreation.MIXED_OR_UNKNOWN,
        BeneficiaryCount.MULTIPLE,
    ),
    FusionSubtype.AP: (
        TransferScope.PARTIAL,
        TransferorFate.SURVIVES,
        BeneficiaryCreation.EXISTING,
        BeneficiaryCount.ONE,
    ),
    FusionSubtype.UNKNOWN: (
        TransferScope.UNKNOWN,
        TransferorFate.UNKNOWN,
        BeneficiaryCreation.MIXED_OR_UNKNOWN,
        BeneficiaryCount.UNKNOWN,
    ),
}


def fusion_result(subtype, **overrides):
    if isinstance(subtype, str):
        subtype = FusionSubtype(subtype)
    scope, fate, creation, count = _CONSISTENT_AXES[subtype]
    values = {
        "subtype": subtype,
        "transfer_scope": scope,
        "transferor_fate": fate,
        "beneficiary_creation": creation,
        "beneficiary_count": count,
        "evidence": (f"indice {subtype.value}",),
        "reason": f"raison {subtype.value}",
    }
    values.update(overrides)
    return FusionSubtypeResult(**values)


class RecordingRouter:
    def __init__(self, results=None, failures=None):
        self.results = results or {}
        self.failures = failures or {}
        self.calls = []

    def route(self, announcement):
        self.calls.append(announcement)
        announcement_id = announcement["id"]
        if announcement_id in self.failures:
            raise self.failures[announcement_id]
        return self.results.get(
            announcement_id, fusion_result(FusionSubtype.UNKNOWN)
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


def json_response(subtype, **overrides):
    result = fusion_result(subtype, **overrides)
    return json.dumps(
        {
            "subtype": result.subtype.value,
            "transfer_scope": result.transfer_scope.value,
            "transferor_fate": result.transferor_fate.value,
            "beneficiary_creation": result.beneficiary_creation.value,
            "beneficiary_count": result.beneficiary_count.value,
            "evidence": list(result.evidence),
            "reason": result.reason,
        }
    )


class FusionSubtypeBenchmarkTest(unittest.TestCase):
    def _write_annotations(self, directory, rows, name="annotations.parquet"):
        path = Path(directory) / name
        pl.DataFrame(rows).write_parquet(path)
        return path

    def _run(self, directory, rows, **overrides):
        path = self._write_annotations(directory, rows)
        output = Path(directory) / "artifacts"
        router = overrides.pop("router", RecordingRouter())
        result = run_fusion_subtype_benchmark(
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

    def test_reference_types_are_exactly_the_fusion_subtypes(self):
        self.assertEqual(FUSION_TYPES, ("FU", "AB", "SP", "ST", "AP"))
        self.assertEqual(
            {
                operation_type: reference_subtype_for_type(operation_type)
                for operation_type in FUSION_TYPES
            },
            {
                operation_type: FusionSubtype(operation_type)
                for operation_type in FUSION_TYPES
            },
        )
        for unsupported in (None, "", "UNKNOWN", "TP", "fu"):
            with self.subTest(unsupported=unsupported):
                with self.assertRaises(ValueError):
                    reference_subtype_for_type(unsupported)

    def test_loader_keeps_only_lookup_and_reference_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_annotations(directory, [annotation()])
            loaded = load_fusion_annotations(path)

        self.assertEqual(loaded.columns, list(ANNOTATION_COLUMNS))
        for excluded in (
            "id_operation",
            "date_creation_op",
            "siren_cedante",
            "siren_beneficiaire",
            "date_effet_comptable_op",
            "date_realisation_juridique_op",
            "montant",
        ):
            self.assertNotIn(excluded, loaded.columns)

    def test_sampling_is_deterministic_by_type_and_supports_all(self):
        rows = []
        for index, operation_type in enumerate(FUSION_TYPES):
            rows.extend(
                [
                    annotation(operation_type, issue=200 + index * 2 + 1),
                    annotation(operation_type, issue=200 + index * 2),
                ]
            )
        frame = pl.DataFrame(rows).select(ANNOTATION_COLUMNS)

        first = select_fusion_annotations(frame, max_per_type=1)
        second = select_fusion_annotations(frame.reverse(), max_per_type=1)
        full = select_fusion_annotations(frame, max_per_type=None)

        self.assertEqual(first[JOIN_KEY].to_list(), second[JOIN_KEY].to_list())
        self.assertEqual(first["type_op"].to_list(), list(FUSION_TYPES))
        self.assertEqual(first.height, 5)
        self.assertEqual(full.height, 10)

    def test_out_of_scope_rows_are_reported_and_never_routed(self):
        fu = annotation("FU")
        rows = [
            fu,
            annotation("VE", issue=148),
            annotation("TP", issue=149),
            annotation(None, issue=150),
        ]
        router = RecordingRouter(
            {fu[JOIN_KEY]: fusion_result(FusionSubtype.FU)}
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, router=router)

        self.assertEqual(result.selected_annotations.height, 1)
        self.assertEqual(len(router.calls), 1)
        self.assertEqual(
            result.summary["coverage"]["out_of_scope_reference_rows"], 3
        )
        self.assertEqual(
            result.summary["coverage"]["out_of_scope_counts_by_type"],
            {"TP": 1, "VE": 1, "__NULL__": 1},
        )
        self.assertEqual(result.summary["failures"]["technical_total"], 0)
        self.assertEqual(result.errors.height, 0)

    def test_invalid_and_duplicate_keys_are_explicit_and_excluded(self):
        duplicate = annotation("FU")
        valid = annotation("SP", issue=149)
        router = RecordingRouter(
            {valid[JOIN_KEY]: fusion_result(FusionSubtype.SP)}
        )
        rows = [
            duplicate,
            {**duplicate, "id_operation": 999},
            annotation("AB", issue=148, ref_annonce_complet=None),
            valid,
        ]

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

    def test_lookup_fetch_llm_and_output_failures_are_separate(self):
        bad_lookup = annotation("FU", ref_annonce="bad-reference")
        bad_fetch = annotation("AB", issue=148)
        bad_llm = annotation("SP", issue=149)
        bad_output = annotation("ST", issue=150)

        def fetch(announcement_id):
            if announcement_id == bad_fetch[JOIN_KEY]:
                raise BodaccFetchError("network_exception", "offline")
            return fetched_announcement(announcement_id)

        router = RecordingRouter(
            failures={
                bad_llm[JOIN_KEY]: FusionSubtypeLLMError(
                    "ConnectionError", "LLM call failed"
                ),
                bad_output[JOIN_KEY]: FusionSubtypeOutputError(
                    "invalid_subtype", "Unsupported subtype"
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [bad_lookup, bad_fetch, bad_llm, bad_output],
                fetch_announcement=fetch,
                router=router,
            )

        self.assertEqual(result.predictions.height, 0)
        self.assertEqual(
            result.summary["failures"]["technical_by_stage"],
            {
                "bodacc_fetch": 1,
                "llm_execution": 1,
                "llm_output_validation": 1,
                "lookup_resolution": 1,
            },
        )
        metrics = result.summary["metrics"]
        self.assertEqual(metrics["eligible_reference_rows"], 4)
        self.assertEqual(metrics["technical_failures"], 4)
        self.assertEqual(metrics["technical_router_failures"], 2)
        for operation_type in FUSION_TYPES[:4]:
            self.assertEqual(
                metrics["confusion_matrix"][operation_type]["__ERROR__"], 1
            )

    def test_unknown_and_invalid_llm_output_remain_distinct_offline(self):
        rows = [annotation("FU"), annotation("AB", issue=148)]
        fake_ask = RecordingAsk(
            [json_response(FusionSubtype.UNKNOWN), "not-json"]
        )
        router = FusionSubtypeRouter(fake_ask)

        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, rows, router=router)

        self.assertEqual(result.predictions.height, 1)
        self.assertEqual(
            result.predictions["predicted_type"].to_list(), ["UNKNOWN"]
        )
        metrics = result.summary["metrics"]
        self.assertEqual(metrics["unknown_count"], 1)
        self.assertEqual(metrics["technical_failures"], 1)
        self.assertEqual(
            metrics["confusion_matrix"]["FU"]["UNKNOWN"], 1
        )
        self.assertEqual(
            metrics["confusion_matrix"]["AB"]["__ERROR__"], 1
        )
        self.assertEqual(
            set(result.errors["failure_code"].to_list()),
            {"semantic_unknown", "invalid_json"},
        )
        self.assertEqual(fake_ask.calls[0][1], {"temperature": 0})
        self.assertEqual(fake_ask.calls[1][1], {"temperature": 0})

    def test_metrics_cover_precision_recall_f1_unknown_error_and_consistency(self):
        references = [
            annotation(operation_type, issue=300 + index)
            for index, operation_type in enumerate(FUSION_TYPES)
        ]
        benchmark = pl.DataFrame(references).select(ANNOTATION_COLUMNS)

        def prediction(reference, predicted, *, consistent=True, issues=None):
            result = fusion_result(predicted)
            return {
                JOIN_KEY: reference[JOIN_KEY],
                "reference_type": reference["type_op"],
                "predicted_type": result.subtype.value,
                "correct": result.subtype.value == reference["type_op"],
                "transfer_scope": result.transfer_scope.value,
                "transferor_fate": result.transferor_fate.value,
                "beneficiary_creation": result.beneficiary_creation.value,
                "beneficiary_count": result.beneficiary_count.value,
                "semantically_consistent": consistent,
                "semantic_consistency_issues": issues or [],
                "reason": result.reason,
                "evidence": list(result.evidence),
            }

        predictions = pl.DataFrame(
            [
                prediction(references[0], FusionSubtype.FU),
                prediction(references[1], FusionSubtype.FU),
                prediction(references[2], FusionSubtype.UNKNOWN),
                prediction(
                    references[4],
                    FusionSubtype.AP,
                    consistent=False,
                    issues=[
                        "beneficiary_creation:expected=EXISTING,actual=NEW"
                    ],
                ),
            ],
            schema=PREDICTION_SCHEMA,
            strict=False,
        )

        metrics = summarize_fusion_subtype_metrics(
            benchmark, predictions, technical_router_failures=1
        )

        self.assertEqual(metrics["eligible_reference_rows"], 5)
        self.assertEqual(metrics["successful_valid_router_outputs"], 4)
        self.assertEqual(metrics["technical_failures"], 1)
        self.assertEqual(metrics["technical_router_failures"], 1)
        self.assertEqual(metrics["accuracy"], 0.4)
        self.assertEqual(metrics["unknown_rate"], 0.2)
        self.assertEqual(metrics["non_unknown_coverage"], 0.6)
        self.assertEqual(metrics["selective_accuracy"], 2 / 3)
        self.assertEqual(metrics["macro_recall"], 0.4)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["per_type"]["FU"]["accuracy"], 1.0)
        self.assertEqual(metrics["per_type"]["FU"]["precision"], 0.5)
        self.assertEqual(metrics["per_type"]["FU"]["recall"], 1.0)
        self.assertAlmostEqual(metrics["per_type"]["FU"]["f1"], 2 / 3)
        self.assertEqual(metrics["per_type"]["SP"]["f1"], 0.0)
        self.assertEqual(metrics["confusion_matrix"]["AB"]["FU"], 1)
        self.assertEqual(metrics["confusion_matrix"]["SP"]["UNKNOWN"], 1)
        self.assertEqual(metrics["confusion_matrix"]["ST"]["__ERROR__"], 1)
        consistency = metrics["semantic_consistency"]
        self.assertEqual(consistency["evaluated_outputs"], 4)
        self.assertEqual(consistency["consistent_count"], 3)
        self.assertEqual(consistency["inconsistent_count"], 1)
        self.assertEqual(consistency["consistency_rate"], 0.75)
        self.assertEqual(
            consistency["accuracy_conditional_on_consistent_outputs"], 1 / 3
        )
        self.assertEqual(
            metrics["priority_confusions"]["FU_AB"],
            {"FU_as_AB": 0, "AB_as_FU": 1, "total": 1},
        )
        self.assertEqual(
            consistency["issues_by_axis"], {"beneficiary_creation": 1}
        )

    def test_correct_but_inconsistent_output_is_inspectable(self):
        fu = annotation("FU")
        inconsistent = fusion_result(
            FusionSubtype.FU,
            beneficiary_creation=BeneficiaryCreation.EXISTING,
        )
        router = RecordingRouter({fu[JOIN_KEY]: inconsistent})

        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(directory, [fu], router=router)

        prediction = result.predictions.row(0, named=True)
        self.assertTrue(prediction["correct"])
        self.assertFalse(prediction["semantically_consistent"])
        self.assertIn(
            "beneficiary_creation:expected=NEW,actual=EXISTING",
            prediction["semantic_consistency_issues"],
        )
        self.assertEqual(result.errors.height, 1)
        error = result.errors.row(0, named=True)
        self.assertEqual(error["failure_code"], "semantic_inconsistency")
        self.assertTrue(error["correct"])
        self.assertEqual(error["predicted_type"], "FU")

    def test_runner_writes_only_dedicated_artifacts_with_axes_and_metadata(self):
        rows = [
            annotation(operation_type, issue=400 + index)
            for index, operation_type in enumerate(FUSION_TYPES)
        ]
        router = RecordingRouter(
            {
                row[JOIN_KEY]: fusion_result(row["type_op"])
                for row in rows
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            result, output, _ = self._run(
                directory, rows, router=router, max_per_type=1
            )
            filenames = {path.name for path in output.iterdir()}
            persisted_summary = json.loads(
                (output / "fusion_summary.json").read_text()
            )
            persisted_predictions = pl.read_parquet(
                output / "fusion_predictions.parquet"
            )
            persisted_errors = pl.read_parquet(output / "fusion_errors.parquet")

        self.assertEqual(
            filenames,
            {
                "fusion_predictions.parquet",
                "fusion_errors.parquet",
                "fusion_summary.json",
            },
        )
        self.assertEqual(len(router.calls), 5)
        self.assertEqual(result.predictions.height, 5)
        self.assertEqual(result.errors.height, 0)
        self.assertEqual(persisted_predictions.columns, list(PREDICTION_SCHEMA))
        self.assertEqual(persisted_errors.columns, list(ERROR_SCHEMA))
        metadata = persisted_summary["metadata"]
        self.assertEqual(metadata["llm_model_name"], "fake-offline-model")
        self.assertEqual(metadata["max_per_type"], 1)
        self.assertEqual(
            metadata["selected_counts_by_type"],
            {operation_type: 1 for operation_type in FUSION_TYPES},
        )
        self.assertEqual(
            persisted_summary["artifacts"],
            {
                "predictions": "fusion_predictions.parquet",
                "errors": "fusion_errors.parquet",
                "summary": "fusion_summary.json",
            },
        )
        self.assertEqual(persisted_summary["metrics"]["accuracy"], 1.0)
        self.assertEqual(
            persisted_summary["confusion_matrix"],
            persisted_summary["metrics"]["confusion_matrix"],
        )
        self.assertEqual(
            persisted_summary["semantic_consistency"],
            persisted_summary["metrics"]["semantic_consistency"],
        )

    def test_annotation_and_fetched_reference_fields_cannot_reach_llm(self):
        fake_ask = RecordingAsk([json_response(FusionSubtype.FU)])
        router = FusionSubtypeRouter(fake_ask)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, [annotation("FU")], router=router
            )

        self.assertEqual(result.predictions.height, 1)
        prompt = json.dumps(fake_ask.calls[0][0], ensure_ascii=False)
        for forbidden in (
            "ANNOTATED_CEDANT_SENTINEL",
            "ANNOTATED_BENEFICIARY_SENTINEL",
            "ANNOTATED_CREATION_DATE_SENTINEL",
            "ANNOTATED_EFFECT_DATE_SENTINEL",
            "ANNOTATED_LEGAL_DATE_SENTINEL",
            "ANNOTATED_AMOUNT_SENTINEL",
            "FETCHED_REFERENCE_SENTINEL",
            "FETCHED_EXPECTED_SENTINEL",
            "FETCHED_RAW_SENTINEL",
        ):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("Fusion synthétique source", prompt)

    def test_module_has_no_generic_router_or_operation_skill_dependency(self):
        source = inspect.getsource(fusion_benchmark_module)
        self.assertNotIn("src.operation", source)
        self.assertNotIn("vente_skill", source)
        self.assertNotIn("location_gerance_skill", source)
        self.assertNotIn("transmission_patrimoine_skill", source)
        self.assertNotIn("family_router", source)
        self.assertNotIn("routing_benchmark", source)

    def test_cli_defaults_to_five_and_supports_all(self):
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
        self.assertEqual(default.max_per_type, 5)
        self.assertIsNone(full.max_per_type)


if __name__ == "__main__":
    unittest.main()
