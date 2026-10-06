import copy
import inspect
import json
import unittest
from unittest.mock import patch

import polars as pl

from src.bodacc import normalize_bodacc_announcement
from src.modele.benchmark import compare_predictions
from src.operation import OperationSkill, location_gerance_skill
import src.operation.location_gerance as location_gerance_module


def person(identifier_key, identifier, name):
    result = {"denomination": name}
    if identifier is not None:
        result["numeroImmatriculation"] = {identifier_key: identifier}
    return result


def rcs_a_announcement(**overrides):
    payload = {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": person(
                "numeroIdentification", "732 829 320", "LOCATAIRE RCS-A"
            )
        },
        "listeprecedentproprietaire": {
            "personne": person(
                "numeroIdentification", "123 456 782", "PROPRIETAIRE IGNORE"
            )
        },
        "listeprecedentexploitant": {
            "personne": person(
                "numeroIdentification", "356 000 000", "EXPLOITANT PRECEDENT"
            )
        },
        "listeetablissements": {
            "etablissement": {
                "origineFonds": "Fonds reçu en location-gérance"
            }
        },
        "acte": {
            "immatriculation": {
                "categorieImmatriculation": "Mise en location gérance",
                "dateImmatriculation": "11/01/2018",
                "dateCommencementActivite": "15/01/2018",
            }
        },
        "dateparution": "2018-01-20",
        "url_complete": "https://www.bodacc.fr/annonce/LGA",
    }
    payload.update(overrides)
    return payload


def rcs_b_announcement(**overrides):
    payload = {
        "registre": "RCS-B",
        "listepersonnes": {
            "personne": person(
                "numeroIdentificationRCS", "732.829.320", "LOCATAIRE RCS-B"
            )
        },
        "modificationsGenerales": {
            "descriptif": (
                "Prise en location-gerance à compter du 30/09/2017"
            ),
            "precedentExploitantPM": person(
                "numeroIdentificationRCS", "356.000.000", "EXPLOITANT RCS-B"
            ),
        },
        "dateparution": "2017-10-05",
        "url": "https://www.bodacc.fr/annonce/LGB",
    }
    payload.update(overrides)
    return payload


class LocationGeranceSkillTest(unittest.TestCase):
    def test_contract_and_rcs_a_parties_dates_amount_source_and_campaign(self):
        result = location_gerance_skill.extract(rcs_a_announcement())

        self.assertIsInstance(location_gerance_skill, OperationSkill)
        self.assertEqual(location_gerance_skill.operation_type, "LG")
        self.assertEqual(result["typeOperation"], "LG")
        self.assertEqual(result["sirenBeneficiaire"], "732829320")
        self.assertEqual(result["raisonSocialeBeneficiaire"], "LOCATAIRE RCS-A")
        self.assertEqual(result["sirenCedant"], "356000000")
        self.assertEqual(result["raisonSocialeCedant"], "EXPLOITANT PRECEDENT")
        self.assertNotEqual(result["sirenCedant"], "123456782")
        self.assertEqual(result["dateEffetComptable"], "2018-01-15")
        self.assertEqual(result["dateRealisationJuridique"], "2018-01-11")
        self.assertIsNone(result["montantNet"])
        self.assertEqual(result["anneeCampagne"], 2018)
        self.assertEqual(
            result["source"], "https://www.bodacc.fr/annonce/LGA"
        )

    def test_campaign_year_accepts_a_publication_year(self):
        result = location_gerance_skill.extract(
            rcs_a_announcement(dateparution="2018")
        )

        self.assertEqual(result["anneeCampagne"], 2018)

    def test_rcs_b_parties_description_date_and_null_legal_date(self):
        result = location_gerance_skill.extract(rcs_b_announcement())

        self.assertEqual(result["sirenBeneficiaire"], "732829320")
        self.assertEqual(result["raisonSocialeBeneficiaire"], "LOCATAIRE RCS-B")
        self.assertEqual(result["sirenCedant"], "356000000")
        self.assertEqual(result["raisonSocialeCedant"], "EXPLOITANT RCS-B")
        self.assertEqual(result["dateEffetComptable"], "2017-09-30")
        self.assertIsNone(result["dateRealisationJuridique"])
        self.assertIsNone(result["montantNet"])
        self.assertEqual(result["source"], "https://www.bodacc.fr/annonce/LGB")

    def test_rcs_b_previous_operator_pm_and_pp_sources_are_normalized(self):
        for operator_key, identifier, expected_siren in (
            ("precedentExploitantPM", "356 000 000", "356000000"),
            ("precedentExploitantPP", "123 456 782", "123456782"),
        ):
            with self.subTest(operator_key=operator_key):
                modifications = {
                    "descriptif": "Location gérance à compter du 2017-09-30",
                    operator_key: person(
                        "numeroIdentificationRCS", identifier, "EXPLOITANT"
                    ),
                }
                result = location_gerance_skill.extract(
                    rcs_b_announcement(modificationsGenerales=modifications)
                )
                self.assertEqual(result["sirenCedant"], expected_siren)

    def test_first_usable_previous_operator_is_selected_once(self):
        operators = {
            "personne": [
                person("numeroIdentification", None, "SANS SIREN"),
                person("numeroIdentification", "123 456 782", "PREMIER UTILISABLE"),
                person("numeroIdentification", "356 000 000", "SECOND UTILISABLE"),
            ]
        }
        result = location_gerance_skill.extract(
            rcs_a_announcement(listeprecedentexploitant=operators)
        )

        self.assertEqual(result["sirenCedant"], "123456782")
        self.assertEqual(result["raisonSocialeCedant"], "PREMIER UTILISABLE")

    def test_missing_operator_never_falls_back_to_owner_and_does_not_crash(self):
        result = location_gerance_skill.extract(
            rcs_a_announcement(listeprecedentexploitant=None)
        )

        self.assertIsNone(result["sirenCedant"])
        self.assertIsNone(result["raisonSocialeCedant"])
        self.assertNotEqual(result["sirenCedant"], "123456782")

    def test_accounting_date_priority_and_final_null(self):
        cases = (
            (
                {
                    "modificationsGenerales": {
                        "descriptif": "Location gérance à compter du 03/04/2020",
                        "dateCommencementActivite": "01/04/2020",
                        "dateEffet": "02/04/2020",
                    },
                    "dateparution": "2020-04-04",
                },
                "2020-04-01",
            ),
            (
                {
                    "modificationsGenerales": {
                        "descriptif": "Location gérance à compter du 03/04/2020",
                        "dateEffet": "02-04-2020",
                    },
                    "dateparution": "2020-04-04",
                },
                "2020-04-02",
            ),
            (
                {
                    "modificationsGenerales": {
                        "descriptif": "Location gérance à compter du 03/04/2020"
                    },
                    "dateparution": "2020-04-04",
                },
                "2020-04-03",
            ),
            (
                {
                    "modificationsGenerales": {
                        "descriptif": "Location gérance sans date d'effet"
                    },
                    "dateparution": "04/04/2020",
                },
                "2020-04-04",
            ),
            (
                {
                    "modificationsGenerales": {
                        "descriptif": "Location gérance sans date d'effet"
                    },
                    "dateparution": None,
                },
                None,
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                result = location_gerance_skill.extract(
                    rcs_b_announcement(**overrides)
                )
                self.assertEqual(result["dateEffetComptable"], expected)

    def test_effective_from_date_formats_are_accent_and_case_tolerant(self):
        for wording, expected in (
            ("LOCATION GERANCE A COMPTER DU 30/09/2017", "2017-09-30"),
            ("location-gérance à compter du 30-09-2017", "2017-09-30"),
            ("Location gerance À COMPTER DU 2017-09-30", "2017-09-30"),
        ):
            with self.subTest(wording=wording):
                result = location_gerance_skill.extract(
                    rcs_b_announcement(
                        modificationsGenerales={"descriptif": wording},
                        dateparution=None,
                    )
                )
                self.assertEqual(result["dateEffetComptable"], expected)

    def test_rcs_a_without_immatriculation_date_has_no_legal_fallback(self):
        payload = rcs_a_announcement(
            acte={
                "immatriculation": {
                    "categorieImmatriculation": "Location-gerance",
                    "dateCommencementActivite": "2018-01-15",
                }
            }
        )

        result = location_gerance_skill.extract(payload)

        self.assertIsNone(result["dateRealisationJuridique"])

    def test_sparse_payload_returns_nullable_fields_without_key_error(self):
        result = location_gerance_skill.extract({})

        self.assertEqual(result["typeOperation"], "LG")
        for field in (
            "anneeCampagne",
            "sirenCedant",
            "raisonSocialeCedant",
            "sirenBeneficiaire",
            "raisonSocialeBeneficiaire",
            "dateEffetComptable",
            "dateRealisationJuridique",
            "montantNet",
            "source",
        ):
            self.assertIsNone(result[field])

    def test_extraction_uses_normalizer_and_does_not_mutate_raw_payload(self):
        payload = rcs_a_announcement(
            listepersonnes=json.dumps(rcs_a_announcement()["listepersonnes"]),
            listeprecedentexploitant=json.dumps(
                rcs_a_announcement()["listeprecedentexploitant"]
            ),
            acte=json.dumps(rcs_a_announcement()["acte"]),
        )
        before = copy.deepcopy(payload)

        with patch.object(
            location_gerance_module,
            "normalize_bodacc_announcement",
            wraps=normalize_bodacc_announcement,
        ) as normalize_mock:
            location_gerance_skill.extract(payload)

        normalize_mock.assert_called_once_with(payload)
        self.assertEqual(payload, before)

    def test_lg_module_has_no_llm_boundary(self):
        source = inspect.getsource(location_gerance_module)

        self.assertNotIn("src.llm", source)
        self.assertNotIn("ask_json", source)

    def test_rcs_a_prediction_is_generic_benchmark_compatible(self):
        prediction = {
            "ref_annonce_complet": "A20180001LG1",
            **location_gerance_skill.extract(rcs_a_announcement()),
        }
        reference = {
            "ref_annonce_complet": "A20180001LG1",
            "type_op": "LG",
            "siren_cedante": 356000000,
            "siren_beneficiaire": 732829320,
            "date_effet_comptable_op": "2018-01-15",
            "date_realisation_juridique_op": "2018-01-11",
            "montant": None,
        }

        comparison = compare_predictions(
            pl.DataFrame([reference]), pl.DataFrame([prediction])
        )

        self.assertTrue(comparison.rows["exact_row_correct"][0])

    def test_rcs_b_prediction_is_generic_benchmark_compatible(self):
        prediction = {
            "ref_annonce_complet": "B20170001LG1",
            **location_gerance_skill.extract(rcs_b_announcement()),
        }
        reference = {
            "ref_annonce_complet": "B20170001LG1",
            "type_op": "LG",
            "siren_cedante": 356000000,
            "siren_beneficiaire": 732829320,
            "date_effet_comptable_op": "2017-09-30",
            "date_realisation_juridique_op": None,
            "montant": None,
        }

        comparison = compare_predictions(
            pl.DataFrame([reference]), pl.DataFrame([prediction])
        )

        self.assertTrue(comparison.rows["exact_row_correct"][0])


if __name__ == "__main__":
    unittest.main()
