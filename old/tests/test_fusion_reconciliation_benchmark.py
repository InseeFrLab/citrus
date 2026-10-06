from datetime import UTC, datetime
from pathlib import Path
import tempfile
import unittest

import polars as pl

from src.modele.benchmark import JOIN_KEY
from src.modele.fusion_reconciliation_benchmark import (
    ERROR_SCHEMA,
    PROVISIONAL_SCHEMA,
    RECONCILED_SCHEMA,
    SEMANTIC_SCHEMA,
    build_argument_parser,
    load_fusion_reconciliation_reference,
    run_fusion_reconciliation_benchmark,
    select_fusion_seed_rows,
)
from src.routing.fusion_semantics import (
    FusionSemanticOutputError,
    FusionSemanticResult,
    LegalFamily,
    ParticipantRole,
    PartialAssetTransferWording,
    SemanticParticipant,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)


def _annotation(operation_type, issue, number=1, **overrides):
    prefix = f"A2023{issue:04d}"
    row = {
        "ref_annonce": f"RCS-A_TEST_{prefix}",
        "numero_annonce": number,
        JOIN_KEY: f"{prefix}{number}",
        "type_op": operation_type,
        "siren_cedante": "ANNOTATED_CEDANT_SENTINEL",
        "siren_beneficiaire": "ANNOTATED_BENEFICIARY_SENTINEL",
        "montant": "ANNOTATED_AMOUNT_SENTINEL",
        "date_creation_op": "ANNOTATED_DATE_SENTINEL",
    }
    row.update(overrides)
    return row


def _party(siren, name):
    return {
        "numeroImmatriculation": {"numeroIdentification": siren},
        "denomination": name,
    }


def _raw(
    announcement_id,
    *,
    main_siren,
    main_name,
    previous_siren=None,
    previous_name=None,
    description=None,
):
    payload = {
        "id": announcement_id,
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": _party(main_siren, main_name)
        },
        "acte": {
            "descriptif": (
                description
                or f"Description source indépendante {announcement_id}"
            )
        },
        "dateparution": "2023-06-01",
        "type_op": "FETCHED_LABEL_SENTINEL",
        "montant": "FETCHED_AMOUNT_SENTINEL",
    }
    if previous_siren is not None:
        payload["listeprecedentproprietaire"] = {
            "personne": _party(previous_siren, previous_name)
        }
    return payload


def _participant(siren, name, role):
    return SemanticParticipant(
        siren=siren,
        name=name,
        role=ParticipantRole(role),
    )


def _semantic(
    legal_family,
    *,
    scope="UNKNOWN",
    fate="UNKNOWN",
    creation="MIXED_OR_UNKNOWN",
    wording="UNKNOWN",
    participants=(),
):
    return FusionSemanticResult(
        legal_family=LegalFamily(legal_family),
        transfer_scope=TransferScope(scope),
        transferor_fate=TransferorFate(fate),
        beneficiary_creation=BeneficiaryCreation(creation),
        partial_asset_transfer_wording=PartialAssetTransferWording(wording),
        participants=tuple(participants),
        evidence=(f"indice {legal_family}",),
        reason=f"raison {legal_family}",
    )


class _RecordingParser:
    def __init__(self, results, failures=None):
        self.results = dict(results)
        self.failures = dict(failures or {})
        self.calls = []

    def parse(self, announcement):
        announcement_id = announcement["id"]
        self.calls.append(announcement_id)
        if announcement_id in self.failures:
            raise self.failures[announcement_id]
        return self.results[announcement_id]


def _fixture_rows_and_sources():
    rows = [
        _annotation("AB", 1),
        _annotation("AB", 2),
        _annotation("SP", 3),
        _annotation("SP", 4),
        _annotation("FU", 5),
        _annotation("ST", 6),
        _annotation("AP", 7),
    ]
    keys = [row[JOIN_KEY] for row in rows]
    shared_fusion = "Projet lié exact : bénéficiaire 732 829 320"
    sources = {
        keys[0]: _raw(
            keys[0],
            main_siren="732829320",
            main_name="ABSORBANTE",
            previous_siren="732829320",
            previous_name="ABSORBANTE",
            description=shared_fusion,
        ),
        keys[1]: _raw(
            keys[1],
            main_siren="356000000",
            main_name="ANNONCE LIEE",
            description=shared_fusion,
        ),
        keys[2]: _raw(
            keys[2],
            main_siren="542051180",
            main_name="SCINDEE",
            previous_siren="542051180",
            previous_name="SCINDEE",
            description="Scission ancre 542 051 180",
        ),
        keys[3]: _raw(
            keys[3],
            main_siren="775670417",
            main_name="BENEFICIAIRE SCISSION",
            previous_siren="542051180",
            previous_name="SCINDEE",
            description="Scission liée 542 051 180",
        ),
        keys[4]: _raw(
            keys[4],
            main_siren="784671695",
            main_name="FUSION ISOLEE",
            description="Fusion isolée 784 671 695",
        ),
        keys[5]: _raw(
            keys[5],
            main_siren="775670417",
            main_name="SCISSION ISOLEE",
            previous_siren="123456782",
            previous_name="CEDANTE ISOLEE",
            description="Scission isolée 123 456 782",
        ),
        keys[6]: _raw(
            keys[6],
            main_siren="552100554",
            main_name="AP BENEFICIAIRE",
            description="Apport partiel autonome",
        ),
    }
    results = {
        keys[0]: _semantic(
            "FUSION",
            scope="TOTAL",
            fate="DISAPPEARS",
            participants=(
                _participant("732829320", "ABSORBANTE", "BENEFICIARY"),
            ),
        ),
        keys[1]: _semantic(
            "FUSION",
            scope="TOTAL",
            fate="DISAPPEARS",
            participants=(
                _participant("732829320", "ABSORBANTE", "BENEFICIARY"),
            ),
        ),
        keys[2]: _semantic(
            "SCISSION",
            participants=(
                _participant("542051180", "SCINDEE", "TRANSFEROR"),
            ),
        ),
        keys[3]: _semantic(
            "SCISSION",
            participants=(
                _participant("542051180", "SCINDEE", "TRANSFEROR"),
            ),
        ),
        keys[4]: _semantic(
            "FUSION",
            participants=(
                _participant("784671695", "FUSION ISOLEE", "BENEFICIARY"),
            ),
        ),
        keys[5]: _semantic(
            "SCISSION",
            participants=(
                _participant("123456782", "CEDANTE ISOLEE", "TRANSFEROR"),
            ),
        ),
        keys[6]: _semantic(
            "UNKNOWN",
            wording="YES",
            scope="PARTIAL",
            fate="SURVIVES",
            creation="EXISTING",
        ),
    }
    return rows, sources, results


class FusionReconciliationBenchmarkTest(unittest.TestCase):
    def _write_annotations(self, directory, rows, name="annotations.parquet"):
        path = Path(directory) / name
        pl.DataFrame(rows).write_parquet(path)
        return path

    def _run(
        self,
        directory,
        rows,
        sources,
        results,
        *,
        max_seeds=None,
        failures=None,
        output_name="out",
    ):
        annotations = self._write_annotations(
            directory, rows, f"{output_name}.parquet"
        )
        fetch_calls = []

        def fetch(announcement_id):
            fetch_calls.append(announcement_id)
            return sources[announcement_id]

        parser = _RecordingParser(results, failures)
        result = run_fusion_reconciliation_benchmark(
            annotations,
            Path(directory) / output_name,
            max_seeds=max_seeds,
            fetch_announcement=fetch,
            semantic_parser=parser,
            run_timestamp=datetime(2026, 8, 28, tzinfo=UTC),
            git_commit="test-commit",
            model_name="offline-test-model",
        )
        return result, fetch_calls, parser

    def test_loader_isolates_labels_and_global_seed_selection(self):
        rows, _, _ = _fixture_rows_and_sources()
        rows.append(_annotation("VE", 8))
        rows.append(_annotation("FU", 9, ref_annonce_complet=None))
        duplicate = _annotation("FU", 10)
        rows.extend((duplicate, dict(duplicate)))
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_annotations(directory, rows)
            corpus = load_fusion_reconciliation_reference(path)

        self.assertNotIn("type_op", corpus.source_rows.columns)
        self.assertNotIn("reference_type", corpus.source_rows.columns)
        self.assertEqual(corpus.labels.columns, [JOIN_KEY, "reference_type"])
        self.assertEqual(corpus.out_of_scope_counts, {"VE": 1})
        self.assertEqual(len(corpus.reference_issues), 3)
        seeds = select_fusion_seed_rows(corpus.source_rows, max_seeds=2)
        self.assertEqual(
            seeds[JOIN_KEY].to_list(),
            sorted(corpus.source_rows[JOIN_KEY].to_list())[:2],
        )
        with self.assertRaises(ValueError):
            select_fusion_seed_rows(
                corpus.labels.rename({"reference_type": "type_op"}),
                max_seeds=1,
            )

    def test_sample_fetches_universe_then_expands_linked_group(self):
        rows, sources, results = _fixture_rows_and_sources()
        with tempfile.TemporaryDirectory() as directory:
            result, fetch_calls, parser = self._run(
                directory,
                rows,
                sources,
                results,
                max_seeds=1,
                output_name="sample",
            )
            artifact_names = {
                path.name for path in (Path(directory) / "sample").iterdir()
            }

        keys = [row[JOIN_KEY] for row in rows]
        self.assertEqual(set(fetch_calls), set(keys))
        self.assertEqual(result.seed_rows[JOIN_KEY].to_list(), [keys[0]])
        self.assertEqual(
            result.expanded_source_rows[JOIN_KEY].to_list(),
            [keys[0], keys[1]],
        )
        self.assertEqual(parser.calls, [keys[0], keys[1]])
        self.assertEqual(result.semantic_predictions.height, 2)
        self.assertEqual(result.provisional.height, 2)
        self.assertEqual(result.reconciled.height, 2)
        self.assertEqual(
            result.reconciled["final_predicted_type"].to_list(),
            ["AB", "AB"],
        )
        self.assertEqual(
            set(result.semantic_predictions.columns),
            set(SEMANTIC_SCHEMA),
        )
        self.assertEqual(set(result.provisional.columns), set(PROVISIONAL_SCHEMA))
        self.assertNotIn("reference_type", result.semantic_predictions.columns)
        self.assertNotIn("correct", result.semantic_predictions.columns)
        self.assertNotIn("reference_type", result.provisional.columns)
        self.assertNotIn("correct", result.provisional.columns)
        self.assertEqual(
            artifact_names,
            {
                "fusion_semantic_predictions.parquet",
                "fusion_provisional.parquet",
                "fusion_reconciled.parquet",
                "fusion_reconciliation_errors.parquet",
                "fusion_reconciliation_summary.json",
            },
        )
        self.assertEqual(
            result.summary["metadata"]["benchmark_authority"],
            "diagnostic_sample",
        )
        self.assertFalse(result.summary["sampling"]["discovery_uses_labels"])
        self.assertEqual(
            result.summary["metrics"]["transitions"][
                "required_transition_counts"
            ]["FZ->AB"],
            1,
        )

    def test_full_run_is_authoritative_and_exposes_required_transitions(self):
        rows, sources, results = _fixture_rows_and_sources()
        with tempfile.TemporaryDirectory() as directory:
            result, fetch_calls, parser = self._run(
                directory, rows, sources, results, max_seeds=None
            )

        self.assertEqual(len(fetch_calls), len(rows))
        self.assertEqual(len(parser.calls), len(rows))
        self.assertEqual(result.reconciled.height, len(rows))
        self.assertEqual(set(result.reconciled.columns), set(RECONCILED_SCHEMA))
        predicted = dict(
            result.reconciled.select(
                JOIN_KEY, "final_predicted_type"
            ).iter_rows()
        )
        keys = [row[JOIN_KEY] for row in rows]
        self.assertEqual(predicted[keys[0]], "AB")
        self.assertEqual(predicted[keys[1]], "AB")
        self.assertEqual(predicted[keys[2]], "SP")
        self.assertEqual(predicted[keys[3]], "SP")
        self.assertEqual(predicted[keys[4]], "FU")
        self.assertEqual(predicted[keys[5]], "ST")
        self.assertEqual(predicted[keys[6]], "AP")
        transitions = result.summary["metrics"]["transitions"]
        self.assertEqual(
            transitions["required_transition_counts"],
            {
                "FZ->AB": 1,
                "FZ->FU": 1,
                "SZ->SP": 1,
                "SZ->ST": 1,
            },
        )
        self.assertEqual(transitions["local_global_conflict_rows"], 0)
        self.assertEqual(transitions["source_rows_dropped_by_reconciliation"], 0)
        self.assertEqual(
            result.summary["metadata"]["benchmark_authority"],
            "authoritative",
        )
        self.assertEqual(
            result.summary["coverage"]["rows_added_by_group_expansion"], 0
        )
        self.assertEqual(
            result.summary["metrics"]["final_classification"]["accuracy"],
            1.0,
        )
        self.assertEqual(
            result.summary["metrics"]["local_semantics"]["legal_family_accuracy"],
            1.0,
        )
        local = result.summary["metrics"]["local_semantics"]
        self.assertEqual(local["valid_semantic_output_rate"], 1.0)
        self.assertEqual(local["formatting_failure_rate"], 0.0)
        self.assertEqual(local["transfer_scope_unknown_count"], 4)
        self.assertEqual(local["participant_role_occurrences"], {
            "BENEFICIARY": 3,
            "TRANSFEROR": 3,
        })
        self.assertEqual(transitions["comparable_rows"], 7)
        self.assertAlmostEqual(
            transitions["accuracy_before_reconciliation"], 5 / 7
        )
        self.assertEqual(
            transitions["accuracy_after_reconciliation"], 1.0
        )
        self.assertEqual(
            transitions["changed_from_default_resolution_rows"], 2
        )
        grouping = result.summary["metrics"]["grouping_linkage"]
        self.assertEqual(grouping["description_grouping_coverage"], 1.0)
        self.assertEqual(grouping["ab_anchor_rows"], 1)
        self.assertEqual(grouping["sp_anchor_rows"], 1)

    def test_full_counts_local_global_sp_conflict(self):
        rows = [_annotation("SP", 31), _annotation("ST", 32)]
        anchor_key, related_key = [row[JOIN_KEY] for row in rows]
        transferor = "542051180"
        sources = {
            anchor_key: _raw(
                anchor_key,
                main_siren=transferor,
                main_name="SCINDEE",
                previous_siren=transferor,
                previous_name="SCINDEE",
            ),
            related_key: _raw(
                related_key,
                main_siren="775670417",
                main_name="BENEFICIAIRE",
                previous_siren=transferor,
                previous_name="SCINDEE",
            ),
        }
        participant = _participant(transferor, "SCINDEE", "TRANSFEROR")
        results = {
            anchor_key: _semantic(
                "SCISSION",
                participants=(participant,),
            ),
            related_key: _semantic(
                "SCISSION",
                scope="TOTAL",
                fate="DISAPPEARS",
                participants=(participant,),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory, rows, sources, results, max_seeds=None
            )

        transitions = result.summary["metrics"]["transitions"]
        self.assertEqual(transitions["local_global_conflict_rows"], 1)
        self.assertEqual(
            transitions["local_global_conflicts_by_code"],
            {
                "sp_anchor_conflicts_with_transfer_scope_total": 1,
                "sp_anchor_conflicts_with_transferor_fate_disappears": 1,
            },
        )
        related = result.reconciled.filter(
            pl.col(JOIN_KEY) == related_key
        ).row(0, named=True)
        self.assertEqual(related["final_predicted_type"], "ST")
        self.assertEqual(
            related["reconciliation_rule"],
            "sz_local_semantics_override_sp_anchor_to_st",
        )

    def test_full_expands_unannotated_linked_notice_before_reconciliation(self):
        row = _annotation("AB", 30)
        source_key = row[JOIN_KEY]
        linked_key = "A2023003002"
        description = (
            "Fusion par absorption de A 123 456 782 par la société "
            "absorbante B 732 829 320."
        )
        sources = {
            source_key: _raw(
                source_key,
                main_siren="123456782",
                main_name="A",
                previous_siren="732829320",
                previous_name="B",
                description=description,
            )
        }
        linked = _raw(
            linked_key,
            main_siren="732829320",
            main_name="B",
            previous_siren="732829320",
            previous_name="B",
            description=description,
        )
        semantic = _semantic(
            "FUSION",
            participants=(
                _participant("123456782", "A", "TRANSFEROR"),
                _participant("732829320", "B", "BENEFICIARY"),
            ),
        )
        parser = _RecordingParser(
            {source_key: semantic, linked_key: semantic}
        )
        search_calls = []

        def fetch(announcement_id):
            return sources[announcement_id]

        def search(siren, year):
            search_calls.append((siren, year))
            return [linked]

        with tempfile.TemporaryDirectory() as directory:
            annotations = self._write_annotations(directory, [row])
            result = run_fusion_reconciliation_benchmark(
                annotations,
                Path(directory) / "expanded-full",
                max_seeds=None,
                fetch_announcement=fetch,
                search_linked_announcements=search,
                semantic_parser=parser,
                run_timestamp=datetime(2026, 8, 30, tzinfo=UTC),
                git_commit="test-commit",
                model_name="offline-test-model",
            )

        self.assertTrue(search_calls)
        self.assertEqual(
            result.expanded_source_rows[JOIN_KEY].to_list(),
            sorted((source_key, linked_key)),
        )
        self.assertEqual(result.summary["coverage"]["final_denominator_rows"], 1)
        self.assertEqual(
            result.summary["coverage"]["rows_added_by_group_expansion"], 1
        )
        self.assertEqual(
            result.summary["metrics"]["transitions"][
                "required_transition_counts"
            ]["FZ->AB"],
            1,
        )
        by_ref = {
            item[JOIN_KEY]: item
            for item in result.reconciled.iter_rows(named=True)
        }
        self.assertEqual(by_ref[source_key]["final_predicted_type"], "AB")
        self.assertIsNone(by_ref[linked_key]["reference_type"])
        self.assertIsNone(by_ref[linked_key]["correct"])

    def test_label_permutation_cannot_change_pipeline_outputs(self):
        rows, sources, results = _fixture_rows_and_sources()
        permuted_types = ["FU", "SP", "ST", "AP", "AB", "FU", "SP"]
        permuted = [
            {**row, "type_op": operation_type}
            for row, operation_type in zip(rows, permuted_types, strict=True)
        ]
        with tempfile.TemporaryDirectory() as directory:
            first, _, _ = self._run(
                directory,
                rows,
                sources,
                results,
                max_seeds=None,
                output_name="first",
            )
            second, _, _ = self._run(
                directory,
                permuted,
                sources,
                results,
                max_seeds=None,
                output_name="second",
            )

        self.assertTrue(
            first.semantic_predictions.equals(second.semantic_predictions)
        )
        self.assertTrue(first.provisional.equals(second.provisional))
        label_free_reconciled = [
            column
            for column in first.reconciled.columns
            if column not in ("reference_type", "correct")
        ]
        self.assertTrue(
            first.reconciled.select(label_free_reconciled).equals(
                second.reconciled.select(label_free_reconciled)
            )
        )
        self.assertNotEqual(
            first.summary["metrics"]["final_classification"]["accuracy"],
            second.summary["metrics"]["final_classification"]["accuracy"],
        )

    def test_raw_llm_error_is_persisted_and_counted_as_technical(self):
        row = _annotation("FU", 20)
        key = row[JOIN_KEY]
        sources = {
            key: _raw(
                key,
                main_siren="784671695",
                main_name="SOURCE",
            )
        }
        failure = FusionSemanticOutputError(
            "invalid_json",
            "invalid response",
            raw_response="not-json",
        )
        with tempfile.TemporaryDirectory() as directory:
            result, _, _ = self._run(
                directory,
                [row],
                sources,
                {},
                max_seeds=None,
                failures={key: failure},
            )

        self.assertEqual(set(result.errors.columns), set(ERROR_SCHEMA))
        self.assertEqual(result.errors.height, 1)
        error = result.errors.row(0, named=True)
        self.assertEqual(error["raw_llm_response"], "not-json")
        self.assertEqual(error["failure_stage"], "llm_output_validation")
        self.assertTrue(error["in_final_denominator"])
        self.assertEqual(
            result.summary["metrics"]["local_semantics"][
                "technical_failures"
            ],
            1,
        )
        self.assertEqual(
            result.summary["metrics"]["final_classification"][
                "technical_failures"
            ],
            1,
        )
        local = result.summary["metrics"]["local_semantics"]
        self.assertEqual(local["output_validation_failures"], 1)
        self.assertEqual(local["formatting_failures"], 1)
        self.assertEqual(local["schema_validation_failures"], 0)

    def test_cli_has_explicit_sample_and_full_modes(self):
        parser = build_argument_parser()
        sample = parser.parse_args(
            [
                "--annotations",
                "annotations.parquet",
                "--output-dir",
                "out",
                "--max-seeds",
                "7",
            ]
        )
        full = parser.parse_args(
            [
                "--annotations",
                "annotations.parquet",
                "--output-dir",
                "out",
                "--all",
            ]
        )
        self.assertEqual(sample.max_seeds, 7)
        self.assertFalse(sample.all)
        self.assertTrue(full.all)
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "--annotations",
                    "annotations.parquet",
                    "--output-dir",
                    "out",
                    "--all",
                    "--max-seeds",
                    "1",
                ]
            )


if __name__ == "__main__":
    unittest.main()
