import copy
import json
import unittest
from unittest.mock import patch

import polars as pl

from src.modele.benchmark import compare_predictions
from src.operation import OperationSkill, vente_skill
from src.operation.vente import extract_amount_vente, parse_vente


def announcement(**overrides):
    payload = {
        "listeprecedentproprietaire": json.dumps({"personne": [{
            "numeroImmatriculation": {"numeroIdentification": "123 456 782"},
            "denomination": "ANCIEN PROPRIETAIRE",
        }]}),
        "listepersonnes": json.dumps({"personne": [{
            "numeroImmatriculation": {"numeroIdentification": "732 829 320"},
            "denomination": "ACQUEREUR",
        }]}),
        "listeetablissements": json.dumps({"etablissement": {
            "origineFonds": "Achat au prix stipulé de 155500 EUR",
        }}),
        "acte": json.dumps({"vente": {
            "publiciteLegale": {"date": "2026-01-02"},
            "descriptif": "Acte signé le 02-01-2026",
        }}),
        "dateparution": "2026-02-03",
        "url_complete": "https://www.bodacc.fr/annonce/detail-annonce/A1",
    }
    payload.update(overrides)
    return payload


class VenteSkillTest(unittest.TestCase):
    def test_contract_and_canonical_fields_preserve_legacy_parse(self):
        payload = announcement()
        legacy = parse_vente(copy.deepcopy(payload))
        with patch("src.operation.vente.ask_json", return_value={"montantNet": 155500}):
            result = vente_skill.extract(copy.deepcopy(payload))

        self.assertIsInstance(vente_skill, OperationSkill)
        for field, value in legacy.items():
            self.assertEqual(result[field], value)
        self.assertEqual(result["typeOperation"], "VE")
        self.assertEqual(result["sirenCedant"], "123456782")
        self.assertEqual(result["sirenBeneficiaire"], "732829320")
        self.assertEqual(result["raisonSocialeCedant"], "ANCIEN PROPRIETAIRE")
        self.assertEqual(result["raisonSocialeBeneficiaire"], "ACQUEREUR")
        self.assertEqual(result["anneeCampagne"], 2026)
        self.assertEqual(result["source"], payload["url_complete"])
        self.assertEqual(result["dateEffetComptable"], "2026-01-02")
        self.assertIsNone(result["dateRealisationJuridique"])

    def test_date_commencement_activite_fallback(self):
        acte = {"dateCommencementActivite": "2026-01-03", "vente": {"descriptif": "unused"}}
        payload = announcement(acte=json.dumps(acte))
        with patch("src.operation.vente.ask_json", return_value={"montantNet": 1000}) as mocked:
            result = vente_skill.extract(payload)
        self.assertEqual(result["dateEffetComptable"], "2026-01-03")
        self.assertEqual(mocked.call_count, 1)

    def test_descriptif_date_uses_existing_llm_boundary(self):
        payload = announcement(acte=json.dumps({"vente": {"descriptif": "Acte le 04-01-2026"}}))
        responses = [{"dateEffetComptable": "04-01-2026"}, {"montantNet": 1000}]
        with patch("src.operation.vente.ask_json", side_effect=responses) as mocked:
            result = vente_skill.extract(payload)
        self.assertEqual(result["dateEffetComptable"], "04-01-2026")
        self.assertEqual(mocked.call_count, 2)

    def test_dateparution_is_final_fallback(self):
        payload = announcement(acte=json.dumps({"vente": {"descriptif": "Sans date"}}))
        responses = [{}, {"montantNet": 1000}]
        with patch("src.operation.vente.ask_json", side_effect=responses):
            result = vente_skill.extract(payload)
        self.assertEqual(result["dateEffetComptable"], "2026-02-03")

    def test_amount_legacy_eur_and_skill_half_away_from_zero_keur(self):
        payload = announcement()
        with patch("src.operation.vente.ask_json", return_value={"montantNet": 155500}):
            legacy_payload = copy.deepcopy(payload)
            parse_vente(legacy_payload)
            legacy_amount = extract_amount_vente(legacy_payload)
            result = vente_skill.extract(copy.deepcopy(payload))
        self.assertEqual(legacy_amount, {"montantNet": 155500})
        self.assertEqual(result["montantNet"], 156)

        with patch("src.operation.vente.ask_json", return_value={"montantNet": -1500}):
            self.assertEqual(vente_skill.extract(announcement())["montantNet"], -2)

    def test_missing_amount_key_returns_none(self):
        with patch("src.operation.vente.ask_json", return_value={}):
            result = vente_skill.extract(announcement())
        self.assertIsNone(result["montantNet"])

    def test_null_amount_returns_none(self):
        with patch("src.operation.vente.ask_json", return_value={"montantNet": None}):
            result = vente_skill.extract(announcement())
        self.assertIsNone(result["montantNet"])

    def test_skill_prediction_can_feed_generic_benchmark_with_attached_key(self):
        with patch("src.operation.vente.ask_json", return_value={"montantNet": 155000}):
            result = vente_skill.extract(announcement())
        prediction = {"ref_annonce_complet": "A1", **result}
        reference = {
            "ref_annonce_complet": "A1", "type_op": "VE",
            "siren_cedante": 123456782, "siren_beneficiaire": 732829320,
            "date_effet_comptable_op": "2026-01-02",
            "date_realisation_juridique_op": None, "montant": 155.0,
        }
        comparison = compare_predictions(pl.DataFrame([reference]), pl.DataFrame([prediction]))
        self.assertTrue(comparison.rows["exact_row_correct"][0])


if __name__ == "__main__":
    unittest.main()
