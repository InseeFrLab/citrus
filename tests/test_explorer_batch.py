import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import polars as pl

from bourrin import APPROACHES, ANALYSIS_MODES, system_prompt
from explorer_batch import (
    FUSION_FAMILY,
    build_argument_parser,
    compute_metrics,
    evaluate,
    filter_operations,
    format_metrics,
    format_operation_list,
    load_batch,
    parse_types,
    reference_rows,
    run_batch,
    select_annonces,
)
from src.bodacc.api import BodaccFetchError


def annotations_frame():
    rows = [
        # (annonce, type, cédant, bénéficiaire, date d'effet, montant en kEUR)
        ("A1", "VE", 448396085, 514120609, datetime(2023, 7, 11), 155.0),
        ("A2", "VE", 12345678, 514120609, None, None),
        ("B1", "TP", 448396085, 514120609, datetime(2023, 1, 1), None),
        ("B2", "AB", 448396085, 514120609, None, 400.0),
        ("B3", "ST", 448396085, 514120609, None, None),
        ("B3", "ST", 448396085, 732829320, None, None),
        ("B4", "LG", 448396085, 514120609, None, None),
    ]
    return pl.DataFrame(
        {
            "id_operation": list(range(1, len(rows) + 1)),
            "ref_annonce_complet": [row[0] for row in rows],
            "type_op": [row[1] for row in rows],
            "siren_cedante": [row[2] for row in rows],
            "siren_beneficiaire": [row[3] for row in rows],
            "date_effet_comptable_op": [row[4] for row in rows],
            "date_realisation_juridique_op": [None] * len(rows),
            "montant": [row[5] for row in rows],
        },
        schema_overrides={
            "date_effet_comptable_op": pl.Datetime,
            "date_realisation_juridique_op": pl.Datetime,
        },
    )


def llm_answer(code, operations, retenu=True):
    return json.dumps(
        {"retenu": retenu, "codeTypeOperation": code, "operations": operations}
    )


def operation(code, cedant, beneficiaire, date=None, montant=None):
    return {
        "codeTypeOperation": code,
        "sirenCedant": cedant,
        "sirenBeneficiaire": beneficiaire,
        "dateEffetComptable": date,
        "dateRealisationJuridique": None,
        "montantNetEuros": montant,
    }


ANSWERS = {
    # tout juste, montant en euros converti en kEUR
    "A1": llm_answer("VE", [operation("VE", "448396085", "514120609", "11/07/2023", 155000)]),
    # SIREN annoté sans zéro de tête : le complément à 9 chiffres doit l'aligner
    "A2": llm_answer("VE", [operation("VE", "012345678", "514120609")]),
    # mauvais type
    "B1": llm_answer("AB", [operation("AB", "448396085", "514120609", "2023-01-01")]),
    # non retenue
    "B2": llm_answer(None, [], retenu=False),
    # une seule des deux opérations annotées
    "B3": llm_answer("ST", [operation("ST", "448396085", "732829320")]),
}


def fake_fetch(annonce_id):
    if annonce_id == "B4":
        raise BodaccFetchError("not_found", "aucune annonce")
    return {"id": annonce_id, "parution": "20230147"}


def fake_ask(messages, **options):
    payload = json.loads(messages[1]["content"].split("\n", 1)[1])
    return ANSWERS[payload["id"]]


class SelectionTest(unittest.TestCase):
    def test_parse_types_expands_fusion_alias_and_deduplicates(self):
        self.assertEqual(parse_types(["VE", "fusion", "AB"]), ("VE", *FUSION_FAMILY))
        self.assertEqual(parse_types(["VE,LG"]), ("VE", "LG"))
        self.assertEqual(len(parse_types(None)), 8)

    def test_parse_types_rejects_unknown_code(self):
        with self.assertRaises(ValueError):
            parse_types(["XX"])

    def test_all_returns_every_annonce_of_selected_types(self):
        annotations = annotations_frame()
        self.assertEqual(select_annonces(annotations, ("VE",), None), ["A1", "A2"])
        self.assertEqual(
            select_annonces(annotations, parse_types(["FUSION"]), None), ["B2", "B3"]
        )

    def test_sample_is_bounded_and_reproducible(self):
        annotations = annotations_frame()
        first = select_annonces(annotations, parse_types(None), 3, seed=7)
        again = select_annonces(annotations, parse_types(None), 3, seed=7)
        self.assertEqual(first, again)
        self.assertLessEqual(len(first), 3)

    def test_per_type_applies_size_to_each_type(self):
        annotations = annotations_frame()
        selected = select_annonces(annotations, ("VE", "TP"), 1, per_type=True)
        self.assertEqual(len(selected), 2)
        self.assertEqual({annonce[0] for annonce in selected}, {"A", "B"})

    def test_reference_rows_are_normalized_for_comparison(self):
        rows = reference_rows(annotations_frame(), "A2")
        self.assertEqual(rows[0]["siren_cedante"], "012345678")
        self.assertIsNone(rows[0]["montant"])
        self.assertEqual(reference_rows(annotations_frame(), "A1")[0]["date_effet_comptable_op"], "2023-07-11")


class BatchTest(unittest.TestCase):
    def setUp(self):
        self.annotations = annotations_frame()
        self.ids = select_annonces(self.annotations, parse_types(None), None)
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "batch"
        self.records = run_batch(
            self.ids,
            self.annotations,
            workers=2,
            output_dir=self.output,
            fetch=fake_fetch,
            ask_fn=fake_ask,
            progress=lambda line: None,
        )
        self.annonces, self.operations = evaluate(self.records)
        self.metrics = compute_metrics(self.annonces, self.operations)

    def tearDown(self):
        self.tmp.cleanup()

    def test_errors_are_recorded_without_stopping_the_batch(self):
        self.assertEqual(len(self.records), 6)
        failed = [record for record in self.records if record["result"] is None]
        self.assertEqual([record["annonce_id"] for record in failed], ["B4"])
        self.assertEqual(failed[0]["statut"], "erreur BODACC")

    def test_annonce_level_metrics(self):
        self.assertEqual(self.metrics["ok"], 5)
        self.assertEqual(self.metrics["retenu"], 4)
        self.assertEqual(self.metrics["type_ok"], 3)  # A1, A2, B3
        self.assertEqual(self.metrics["nombre_operations_ok"], 3)  # A1, A2, B1
        self.assertEqual(self.metrics["confusion"][("TP", "AB")], 1)
        self.assertEqual(self.metrics["confusion"][("AB", "non retenu")], 1)

    def test_operation_level_metrics(self):
        statuts = self.metrics["operation_statuts"]
        self.assertEqual(statuts["appariée"], 4)  # A1, A2, B1, B3 (1 des 2)
        self.assertEqual(statuts["non prédite"], 2)  # B2, seconde ligne de B3
        self.assertEqual(statuts["erreur"], 1)  # B4
        fields = self.metrics["champs"]
        self.assertEqual(fields["sirenCedant"]["exact"], 4)
        self.assertEqual(fields["typeOperation"]["exact"], 3)
        self.assertEqual(fields["montantNet"]["exact_si_renseigne"], 1)
        self.assertEqual(self.metrics["tout_exact"], 3)  # A1, A2, B3

    def test_st_prediction_is_paired_by_siren_couple(self):
        b3 = self.operations.filter(pl.col("annonce_id") == "B3")
        paired = b3.filter(pl.col("statut") == "appariée")
        self.assertEqual(paired["sirenBeneficiaire_annote"].to_list(), ["732829320"])

    def test_saved_batch_reloads_to_same_metrics(self):
        _, records = load_batch(self.output)
        self.assertEqual(len(records), len(self.records))
        self.assertTrue(all("messages" not in (r["result"] or {}) for r in records))
        reloaded = compute_metrics(*evaluate(records))
        self.assertEqual(reloaded, self.metrics)

    def test_ask_options_reach_the_llm_call(self):
        seen = []

        def recording_ask(messages, **options):
            seen.append(options)
            return fake_ask(messages)

        run_batch(
            ["A1"],
            self.annotations,
            fetch=fake_fetch,
            ask_fn=recording_ask,
            progress=lambda line: None,
            reasoning=False,
        )
        self.assertEqual(seen, [{"reasoning": False}])

    def test_approach_selects_its_own_prompt(self):
        seen = []

        def recording_ask(messages, **options):
            seen.append((messages[0]["content"], options))
            return fake_ask(messages)

        records = run_batch(
            ["A1"],
            self.annotations,
            approach="juridique",
            fetch=fake_fetch,
            ask_fn=recording_ask,
            progress=lambda line: None,
        )
        self.assertEqual(seen, [(system_prompt("juridique", "complete"), {})])
        self.assertNotEqual(system_prompt("juridique"), system_prompt("metier"))
        result = records[0]["result"]
        self.assertEqual(result["envelope"]["approche"], "juridique")
        self.assertEqual(result["prompt_version"], "bourrin-juridique-v2")

    def test_analysis_modes_change_the_output_format(self):
        for approach in APPROACHES:
            prompts = {mode: system_prompt(approach, mode) for mode in ANALYSIS_MODES}
            self.assertEqual(len(set(prompts.values())), 3)
            for mode, prompt in prompts.items():
                self.assertNotIn("{{", prompt)
                self.assertEqual('"analyse":' in prompt, mode != "aucune", (approach, mode))
            self.assertIn("2 à 3 phrases", prompts["courte"])
        with self.assertRaises(ValueError):
            system_prompt("metier", "longue")

    def test_analysis_mode_reaches_the_prompt_not_the_llm_options(self):
        seen = []

        def recording_ask(messages, **options):
            seen.append((messages[0]["content"], options))
            return fake_ask(messages)

        records = run_batch(
            ["A1"],
            self.annotations,
            analysis="aucune",
            fetch=fake_fetch,
            ask_fn=recording_ask,
            progress=lambda line: None,
        )
        self.assertEqual(seen, [(system_prompt("metier", "aucune"), {})])
        result = records[0]["result"]
        self.assertEqual(result["analyse_mode"], "aucune")
        self.assertIsNone(result["envelope"]["analyse"])

    def test_default_approach_is_metier(self):
        self.assertTrue(all(r["result"]["approche"] == "metier" for r in self.records if r["result"]))
        self.assertEqual(build_argument_parser().parse_args([]).approche, "metier")
        self.assertEqual(build_argument_parser().parse_args([]).analyse, "complete")
        self.assertEqual(
            build_argument_parser().parse_args(["--analyse", "courte"]).analyse, "courte"
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argument_parser().parse_args(["--approche", "autre"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_argument_parser().parse_args(["--analyse", "longue"])

    def test_filters_and_rendering(self):
        self.assertEqual(filter_operations(self.operations, ["VE"]).height, 2)
        self.assertEqual(filter_operations(self.operations, ["type"]).height, 1)
        self.assertEqual(filter_operations(self.operations, ["T"]).height, 1)
        self.assertEqual(filter_operations(self.operations, ["erreurs"]).height, 4)
        with self.assertRaises(ValueError):
            filter_operations(self.operations, ["nimporte"])
        self.assertIn("B3", format_operation_list(self.operations))
        self.assertIn("Matrice de confusion", format_metrics(self.metrics, {}))
        self.assertIn(
            "lecture juridique", format_metrics(self.metrics, {"approche": "juridique"})
        )


if __name__ == "__main__":
    unittest.main()
