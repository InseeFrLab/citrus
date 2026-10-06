import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import polars as pl

from src.modele.benchmark import (
    BenchmarkValidationError,
    compare_predictions,
    load_annotations_csv,
    normalize_annotations,
    summarize_metrics,
)


def annotation(key="A1", **overrides):
    row = {
        "id_operation": 1,
        "ref_annonce": "RCS-A_TEST",
        "numero_annonce": 1,
        "ref_annonce_complet": key,
        "type_op": "VE",
        "siren_cedante": 55801013,
        "siren_beneficiaire": None,
        "date_creation_op": datetime(2026, 8, 23, 15, 30),
        "date_effet_comptable_op": datetime(2026, 1, 2, 12, 30),
        "date_realisation_juridique_op": None,
        "montant": 155.0,
    }
    row.update(overrides)
    return row


def prediction(key="A1", **overrides):
    row = {
        "ref_annonce_complet": key,
        "typeOperation": "VE",
        "sirenCedant": "055801013",
        "sirenBeneficiaire": None,
        "dateEffetComptable": "2026-01-02",
        "dateRealisationJuridique": None,
        "montantNet": 155,
    }
    row.update(overrides)
    return row


class BenchmarkTest(unittest.TestCase):
    def test_normalizes_siren_null_date_and_excludes_creation_date(self):
        normalized = normalize_annotations(pl.DataFrame([annotation()]))
        self.assertEqual(normalized["sirenCedant"][0], "055801013")
        self.assertIsNone(normalized["sirenBeneficiaire"][0])
        self.assertEqual(normalized["dateEffetComptable"][0], date(2026, 1, 2))
        self.assertNotIn("date_creation_op", normalized.columns)

    def test_flags_fields_exact_row_and_numeric_regression(self):
        result = compare_predictions(
            pl.DataFrame([annotation()]),
            pl.DataFrame([prediction(montantNet=154.8)]),
        )
        row = result.rows.row(0, named=True)
        self.assertTrue(row["sirenCedant_correct"])
        self.assertTrue(row["dateEffetComptable_correct"])
        self.assertTrue(row["sirenBeneficiaire_correct"])
        self.assertFalse(row["montantNet_correct"])
        self.assertFalse(row["exact_row_correct"])

    def test_amounts_are_compared_as_keUR(self):
        result = compare_predictions(
            pl.DataFrame([annotation(montant=155.05)]),
            pl.DataFrame([prediction(montantNet=155.0)]),
        )
        self.assertTrue(result.rows["montantNet_correct"][0])

    def test_accounts_for_missing_extra_and_confusion(self):
        references = pl.DataFrame([
            annotation("A1"),
            annotation("A2", type_op="LG", id_operation=2),
        ])
        predictions = pl.DataFrame([
            prediction("A1", typeOperation="LG"),
            prediction("EXTRA"),
        ])
        summary = summarize_metrics(compare_predictions(references, predictions))
        self.assertEqual(summary["coverage"]["matched_rows"], 1)
        self.assertEqual(summary["coverage"]["missing_predictions"], 1)
        self.assertEqual(summary["coverage"]["extra_predictions"], 1)
        self.assertEqual(summary["classification"]["confusion_matrix"]["VE"]["LG"], 1)
        self.assertEqual(summary["classification"]["confusion_matrix"]["LG"]["__MISSING__"], 1)

    def test_unknown_prediction_has_visible_confusion_bucket(self):
        result = compare_predictions(
            pl.DataFrame([annotation()]), pl.DataFrame([prediction(typeOperation="?")])
        )
        summary = summarize_metrics(result)
        self.assertEqual(summary["classification"]["confusion_matrix"]["VE"]["__UNKNOWN__"], 1)

    def test_rejects_duplicate_keys_on_either_side(self):
        with self.assertRaisesRegex(BenchmarkValidationError, "Duplicate"):
            compare_predictions(
                pl.DataFrame([annotation(), annotation()]),
                pl.DataFrame([prediction()]),
            )
        with self.assertRaisesRegex(BenchmarkValidationError, "Duplicate"):
            compare_predictions(
                pl.DataFrame([annotation()]),
                pl.DataFrame([prediction(), prediction()]),
            )

    def test_malformed_siren_is_rejected(self):
        with self.assertRaisesRegex(BenchmarkValidationError, "SIREN"):
            normalize_annotations(pl.DataFrame([annotation(siren_cedante="12A")]))

    def test_loaded_csv_can_be_compared_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.csv"
            csv_row = annotation(
                date_effet_comptable_op="11/07/2023 02:00:00"
            )
            pl.DataFrame([csv_row]).write_csv(path)

            reference = load_annotations_csv(path)
            comparison = compare_predictions(
                reference,
                pl.DataFrame([
                    prediction(dateEffetComptable="2023-07-11")
                ]),
            )

        self.assertEqual(reference["sirenCedant"][0], "055801013")
        self.assertEqual(reference["dateEffetComptable"][0], date(2023, 7, 11))
        self.assertTrue(comparison.rows["dateEffetComptable_correct"][0])
        self.assertTrue(comparison.rows["exact_row_correct"][0])


if __name__ == "__main__":
    unittest.main()
