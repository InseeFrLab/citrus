import copy
from dataclasses import fields
import json
import unittest

from src.bodacc import (
    BodaccDialect,
    BodaccNormalizationError,
    NormalizedParty,
    extract_siren_candidates,
    normalize_bodacc_announcement,
)


def person(identifier_key, identifier, name):
    return {
        "numeroImmatriculation": {identifier_key: identifier},
        "denomination": name,
    }


def parsed_rcs_a_payload():
    return {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": [
                person("numeroIdentification", "732 829 320", "SOCIETE ACTUELLE"),
                person("numeroIdentification", 55801013, "SECONDE SOCIETE"),
            ]
        },
        "listeprecedentproprietaire": {
            "personne": [
                person("numeroIdentification", "123.456.782", "ANCIEN PROPRIETAIRE"),
                {"denomination": "PROPRIETAIRE SANS SIREN"},
            ]
        },
        "listeprecedentexploitant": {
            "personne": person(
                "numeroIdentification", "356 000 000", "ANCIEN EXPLOITANT"
            )
        },
        "listeetablissements": {
            "etablissement": [
                {"origineFonds": "Achat d'un fonds"},
                {"origineFonds": "Création"},
            ]
        },
        "acte": {
            "dateCommencementActivite": "2026-01-02",
            "dateEffet": "2026-01-03",
            "vente": {
                "descriptif": "Description de vente RCS-A",
                "publiciteLegale": {"date": "2026-01-04"},
            },
        },
        "dateparution": "2026-01-05",
        "url_complete": "https://www.bodacc.fr/annonce/test",
    }


def stringified(payload):
    result = copy.deepcopy(payload)
    for key in (
        "listepersonnes",
        "listeprecedentproprietaire",
        "listeprecedentexploitant",
        "listeetablissements",
        "acte",
    ):
        result[key] = json.dumps(result[key])
    return result


REAL_B202302491051_DESCRIPTION = (
    "modification survenue sur Observation: Décision de l'associé unique en "
    "date du 6 novembre 2023 décidant de la dissolution et de la transmission "
    "universelle du patrimoine de la société à l'associé unique La Société "
    "FSH, 824 640 916 RCS LA ROCHE SUR YON, sans qu'il y ait lieu à "
    "liquidation. Les créanciers peuvent faire opposition à la dissolution "
    "dans le délai de trente jours à compter de la publication de celle-ci "
    "dans Informateur Judiciaire du 10 novembre 2023"
)


def real_b202302491051_payload(modifications_generales):
    return {
        "familleavis": "modification",
        "familleavis_lib": "Modifications diverses",
        "typeavis": "annonce",
        "registre": ["810 379 180", "810379180"],
        "listepersonnes": json.dumps(
            {
                "personne": {
                    "typePersonne": "pm",
                    "numeroImmatriculation": {
                        "numeroIdentification": "810 379 180",
                        "codeRCS": "RCS",
                        "nomGreffeImmat": "Saint-Nazaire",
                    },
                    "denomination": "EZ PRINT 3D",
                    "formeJuridique": (
                        "Société à responsabilité limitée (à associé unique)"
                    ),
                    "capital": {"montantCapital": "5000", "devise": "EUR"},
                    "adresseSiegeSocial": {
                        "typeVoie": "parc",
                        "nomVoie": (
                            "d'Activité du Pont Beranger, 14 rue Henri Becquerel"
                        ),
                        "codePostal": "44680",
                        "ville": "Saint-Hilaire-de-Chaléons",
                    },
                }
            }
        ),
        "modificationsgenerales": modifications_generales,
        "dateparution": "2023-12-26",
        "url_complete": (
            "https://www.bodacc.fr/pages/annonces-commerciales-detail/"
            "?q.id=id:B202302491051"
        ),
    }


class BodaccNormalizationTest(unittest.TestCase):
    def test_real_shaped_complex_acts_preserve_generic_descriptions(self):
        cases = (
            (
                "FU",
                "A20230191265",
                "Avis au Bodacc relatif au projet commun de fusion nationale. "
                "Société absorbante et société absorbée n° 1.",
                None,
                False,
            ),
            (
                "AB",
                "A20230150146",
                "Avis relatif aux sociétés absorbante et absorbée : la fusion "
                "prendrait effet à sa réalisation définitive.",
                "Autre achat, apport, attribution",
                False,
            ),
            (
                "SP",
                "A202301901462",
                "Projet commun de scission nationale. Société scindée et "
                "société bénéficiaire de la scission par apport partiel d'actif.",
                None,
                True,
            ),
            (
                "ST",
                "A20230158736",
                "Projet de traité de scission totale au profit de plusieurs "
                "sociétés bénéficiaires.",
                "Projet de traité de scission totale au profit de plusieurs "
                "sociétés bénéficiaires.",
                False,
            ),
            (
                "AP",
                "A202302081505",
                "AVIS DE PROJET D'APPORT PARTIEL D'ACTIF entre la société "
                "apporteuse et la société bénéficiaire.",
                None,
                False,
            ),
        )

        for operation_type, reference, description, sale_text, encoded in cases:
            with self.subTest(operation_type=operation_type, reference=reference):
                vente = {
                    "categorieVente": (
                        "Autre achat, apport, attribution, immatriculation"
                    )
                }
                if sale_text is not None:
                    vente["descriptif"] = sale_text
                acte = {"descriptif": description, "vente": vente}
                payload = {
                    "id": reference,
                    "registre": "RCS-A",
                    "listepersonnes": {
                        "personne": person(
                            "numeroIdentification",
                            "732 829 320",
                            f"SOCIETE {operation_type}",
                        )
                    },
                    "acte": json.dumps(acte) if encoded else acte,
                }
                before = copy.deepcopy(payload)

                normalized = normalize_bodacc_announcement(payload)

                self.assertEqual(normalized.act_description, description)
                self.assertIn(description, normalized.all_descriptions)
                self.assertEqual(normalized.sale_description, sale_text)
                expected = (
                    (description, sale_text)
                    if sale_text is not None and sale_text != description
                    else (description,)
                )
                self.assertEqual(normalized.all_descriptions, expected)
                self.assertEqual(payload, before)
                self.assertEqual(normalized.primary_description, sale_text)

    def test_act_description_dict_and_stringified_acte_match(self):
        acte = {
            "descriptif": "Projet commun de fusion nationale",
            "vente": {"descriptif": "Description de vente distincte"},
        }
        variants = (
            {"registre": "RCS-A", "acte": acte},
            {"registre": "RCS-A", "acte": json.dumps(acte)},
        )

        normalized = [
            normalize_bodacc_announcement(payload)
            for payload in variants
        ]

        self.assertEqual(normalized[0].act_description, acte["descriptif"])
        self.assertEqual(
            normalized[0].all_descriptions,
            (
                acte["descriptif"],
                acte["vente"]["descriptif"],
            ),
        )
        for field in fields(normalized[0]):
            if field.name != "raw_payload":
                self.assertEqual(
                    getattr(normalized[0], field.name),
                    getattr(normalized[1], field.name),
                )

    def test_real_rcs_b_lowercase_modification_dict_and_json_match(self):
        modification = {"descriptif": REAL_B202302491051_DESCRIPTION}

        for value in (modification, json.dumps(modification)):
            with self.subTest(stringified=isinstance(value, str)):
                normalized = normalize_bodacc_announcement(
                    real_b202302491051_payload(value)
                )

                self.assertEqual(normalized.dialect, BodaccDialect.RCS_B)
                self.assertEqual(normalized.main_siren, "810379180")
                self.assertEqual(
                    normalized.modification_description,
                    REAL_B202302491051_DESCRIPTION,
                )
                self.assertEqual(
                    normalized.primary_description,
                    REAL_B202302491051_DESCRIPTION,
                )

    def test_modification_key_aliases_must_not_conflict(self):
        description = {"descriptif": "Même description"}
        normalized = normalize_bodacc_announcement(
            {
                "familleavis": "modification",
                "modificationsGenerales": description,
                "modificationsgenerales": json.dumps(description),
            }
        )
        self.assertEqual(normalized.modification_description, "Même description")

        with self.assertRaisesRegex(
            BodaccNormalizationError, "Conflicting modification containers"
        ):
            normalize_bodacc_announcement(
                {
                    "modificationsGenerales": {"descriptif": "Première"},
                    "modificationsgenerales": {"descriptif": "Seconde"},
                }
            )

    def test_malformed_lowercase_modification_json_raises(self):
        with self.assertRaisesRegex(
            BodaccNormalizationError,
            "Malformed JSON container in modificationsgenerales",
        ):
            normalize_bodacc_announcement(
                {"modificationsgenerales": "{not-json}"}
            )

    def test_modification_metadata_is_conservative_and_contradictory(self):
        for value in ("modification", " MODIFICATION ", "modifications"):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize_bodacc_announcement(
                        {"familleavis": value}
                    ).dialect,
                    BodaccDialect.RCS_B,
                )

        self.assertEqual(
            normalize_bodacc_announcement(
                {"familleavis_lib": "Modifications diverses"}
            ).dialect,
            BodaccDialect.RCS_B,
        )
        self.assertEqual(
            normalize_bodacc_announcement(
                {"familleavis": "radiation sans autre indication"}
            ).dialect,
            BodaccDialect.UNKNOWN,
        )
        self.assertEqual(
            normalize_bodacc_announcement(
                {"registre": "RCS-A", "familleavis": "modification"}
            ).dialect,
            BodaccDialect.UNKNOWN,
        )

    def test_stringified_and_parsed_containers_normalize_identically(self):
        parsed = normalize_bodacc_announcement(parsed_rcs_a_payload())
        encoded = normalize_bodacc_announcement(stringified(parsed_rcs_a_payload()))

        for field in fields(parsed):
            if field.name != "raw_payload":
                self.assertEqual(
                    getattr(parsed, field.name), getattr(encoded, field.name)
                )

    def test_rcs_a_immatriculation_facts_accept_parsed_and_stringified_input(self):
        variants = []
        for stringify_acte, stringify_immatriculation in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            immatriculation = {
                "categorieImmatriculation": "Mise en location-gérance",
                "dateImmatriculation": "2018-01-11",
            }
            acte = {
                "immatriculation": (
                    json.dumps(immatriculation)
                    if stringify_immatriculation
                    else immatriculation
                )
            }
            variants.append(
                (
                    stringify_acte,
                    stringify_immatriculation,
                    {
                        "registre": "RCS-A",
                        "acte": json.dumps(acte) if stringify_acte else acte,
                    },
                )
            )

        for stringify_acte, stringify_immatriculation, payload in variants:
            with self.subTest(
                acte_stringified=stringify_acte,
                immatriculation_stringified=stringify_immatriculation,
            ):
                normalized = normalize_bodacc_announcement(payload)

                self.assertEqual(
                    normalized.immatriculation_category,
                    "Mise en location-gérance",
                )
                self.assertEqual(
                    normalized.immatriculation_date,
                    "2018-01-11",
                )

    def test_normalization_does_not_mutate_or_retain_nested_input_references(self):
        payload = parsed_rcs_a_payload()
        before = copy.deepcopy(payload)
        normalized = normalize_bodacc_announcement(payload)

        self.assertEqual(payload, before)
        payload["listepersonnes"]["personne"][0]["denomination"] = "CHANGED"
        self.assertEqual(
            normalized.raw_payload["listepersonnes"]["personne"][0]["denomination"],
            "SOCIETE ACTUELLE",
        )

    def test_rcs_a_current_parties_main_accessors_and_siren_format(self):
        normalized = normalize_bodacc_announcement(parsed_rcs_a_payload())

        self.assertEqual(normalized.dialect, BodaccDialect.RCS_A)
        self.assertEqual(normalized.main_siren, "732829320")
        self.assertEqual(normalized.main_name, "SOCIETE ACTUELLE")
        self.assertEqual(
            normalized.current_persons,
            (
                NormalizedParty("732829320", "SOCIETE ACTUELLE"),
                NormalizedParty("055801013", "SECONDE SOCIETE"),
            ),
        )

    def test_rcs_b_identifier_and_modification_description(self):
        payload = {
            "listepersonnes": {
                "personne": person(
                    "numeroIdentificationRCS", "732.829.320", "SOCIETE RCS-B"
                )
            },
            "modificationsGenerales": json.dumps(
                {
                    "descriptif": "Description de modification RCS-B",
                    "dateEffet": "2026-02-01",
                }
            ),
        }

        normalized = normalize_bodacc_announcement(payload)

        self.assertEqual(normalized.dialect, BodaccDialect.RCS_B)
        self.assertEqual(normalized.main_siren, "732829320")
        self.assertEqual(
            normalized.modification_description,
            "Description de modification RCS-B",
        )
        self.assertEqual(
            normalized.primary_description,
            "Description de modification RCS-B",
        )
        self.assertEqual(normalized.effect_date, "2026-02-01")

    def test_previous_owners_and_operators_preserve_order_and_missing_values(self):
        normalized = normalize_bodacc_announcement(parsed_rcs_a_payload())

        self.assertEqual(
            normalized.previous_owners,
            (
                NormalizedParty("123456782", "ANCIEN PROPRIETAIRE"),
                NormalizedParty(None, "PROPRIETAIRE SANS SIREN"),
            ),
        )
        self.assertEqual(
            normalized.previous_operators,
            (NormalizedParty("356000000", "ANCIEN EXPLOITANT"),),
        )

    def test_rcs_a_documented_previous_operators_and_dates(self):
        for stringify_structures in (False, True):
            with self.subTest(stringified=stringify_structures):
                payload = {
                    "registre": "RCS-A",
                    "precedentExploitantPM": person(
                        "numeroIdentification",
                        "123 456 782",
                        "EXPLOITANT PERSONNE MORALE",
                    ),
                    "precedentExploitantPP": person(
                        "numeroIdentification",
                        "356 000 000",
                        "EXPLOITANT PERSONNE PHYSIQUE",
                    ),
                    "acte": {
                        "immatriculation": {
                            "dateCommencementActivite": "2026-03-01",
                            "dateEffet": "2026-03-02",
                        }
                    },
                }
                if stringify_structures:
                    payload["precedentExploitantPM"] = json.dumps(
                        payload["precedentExploitantPM"]
                    )
                    payload["precedentExploitantPP"] = json.dumps(
                        payload["precedentExploitantPP"]
                    )
                    payload["acte"] = json.dumps(payload["acte"])

                normalized = normalize_bodacc_announcement(payload)

                self.assertEqual(
                    normalized.previous_operators,
                    (
                        NormalizedParty(
                            "123456782", "EXPLOITANT PERSONNE MORALE"
                        ),
                        NormalizedParty(
                            "356000000", "EXPLOITANT PERSONNE PHYSIQUE"
                        ),
                    ),
                )
                self.assertEqual(normalized.commencement_date, "2026-03-01")
                self.assertEqual(normalized.effect_date, "2026-03-02")

    def test_rcs_b_documented_previous_operators_and_dates(self):
        for stringify_modifications in (False, True):
            with self.subTest(stringified=stringify_modifications):
                modifications = {
                    "precedentExploitantPM": person(
                        "numeroIdentificationRCS",
                        "123.456.782",
                        "EXPLOITANT RCS-B PM",
                    ),
                    "precedentExploitantPP": json.dumps(
                        person(
                            "numeroIdentificationRCS",
                            "356 000 000",
                            "EXPLOITANT RCS-B PP",
                        )
                    ),
                    "dateCommencementActivite": "2026-04-01",
                    "dateEffet": "2026-04-02",
                }
                payload = {
                    "registre": "RCS-B",
                    "modificationsGenerales": (
                        json.dumps(modifications)
                        if stringify_modifications
                        else modifications
                    ),
                }

                normalized = normalize_bodacc_announcement(payload)

                self.assertEqual(
                    normalized.previous_operators,
                    (
                        NormalizedParty("123456782", "EXPLOITANT RCS-B PM"),
                        NormalizedParty("356000000", "EXPLOITANT RCS-B PP"),
                    ),
                )
                self.assertEqual(normalized.commencement_date, "2026-04-01")
                self.assertEqual(normalized.effect_date, "2026-04-02")

    def test_descriptions_remain_separate_and_unknown_has_no_primary_guess(self):
        payload = parsed_rcs_a_payload()
        payload.pop("registre")
        payload["listepersonnes"] = None
        payload["acte"]["descriptif"] = "Description générique d'acte"
        payload["modificationsGenerales"] = {
            "descriptif": "Description de modification RCS-B"
        }

        normalized = normalize_bodacc_announcement(payload)

        self.assertEqual(normalized.dialect, BodaccDialect.UNKNOWN)
        self.assertEqual(
            normalized.act_description, "Description générique d'acte"
        )
        self.assertEqual(normalized.sale_description, "Description de vente RCS-A")
        self.assertEqual(
            normalized.modification_description,
            "Description de modification RCS-B",
        )
        self.assertEqual(
            normalized.all_descriptions,
            (
                "Description générique d'acte",
                "Description de vente RCS-A",
                "Description de modification RCS-B",
            ),
        )
        self.assertIsNone(normalized.primary_description)

    def test_source_dates_url_and_origin_funds_are_independent(self):
        normalized = normalize_bodacc_announcement(parsed_rcs_a_payload())

        self.assertEqual(normalized.publication_date, "2026-01-05")
        self.assertEqual(normalized.commencement_date, "2026-01-02")
        self.assertEqual(normalized.effect_date, "2026-01-03")
        self.assertEqual(normalized.legal_publication_date, "2026-01-04")
        self.assertEqual(
            normalized.source_url, "https://www.bodacc.fr/annonce/test"
        )
        self.assertEqual(
            normalized.origin_funds, ("Achat d'un fonds", "Création")
        )
        self.assertEqual(normalized.first_origin_funds, "Achat d'un fonds")

    def test_sparse_payload_has_explicit_empty_values(self):
        normalized = normalize_bodacc_announcement(
            {
                "listepersonnes": None,
                "acte": "",
                "modificationsGenerales": "null",
            }
        )

        self.assertEqual(normalized.dialect, BodaccDialect.UNKNOWN)
        self.assertEqual(normalized.current_persons, ())
        self.assertEqual(normalized.previous_owners, ())
        self.assertEqual(normalized.previous_operators, ())
        self.assertEqual(normalized.origin_funds, ())
        self.assertEqual(normalized.all_descriptions, ())
        self.assertIsNone(normalized.act_description)
        self.assertIsNone(normalized.main_siren)
        self.assertIsNone(normalized.main_name)
        self.assertIsNone(normalized.publication_date)

    def test_malformed_or_scalar_expected_json_has_clear_error(self):
        with self.assertRaisesRegex(
            BodaccNormalizationError, "Malformed JSON container in acte"
        ):
            normalize_bodacc_announcement({"acte": "{not-json}"})

        with self.assertRaisesRegex(
            BodaccNormalizationError, "Expected a JSON object or array"
        ):
            normalize_bodacc_announcement({"listepersonnes": '"not-a-container"'})

    def test_siren_candidates_accept_supported_forms_and_preserve_order(self):
        text = (
            "Sociétés 123.456.782, 732 829 320 et 356000000; "
            "puis à nouveau 123456782."
        )

        self.assertEqual(
            extract_siren_candidates(text),
            ("123456782", "732829320", "356000000"),
        )

    def test_siren_candidates_exclude_known_and_reject_invalid_luhn(self):
        text = "Principal 732829320, autre 123456789, cible 123 456 782."

        self.assertEqual(
            extract_siren_candidates(text, excluded_sirens={"732 829 320"}),
            ("123456782",),
        )

    def test_siren_candidates_reject_obvious_currency_amounts(self):
        text = (
            "Prix 732829320 EUR, autre prix 123.456.782 €, "
            "mais société 356 000 000."
        )

        self.assertEqual(extract_siren_candidates(text), ("356000000",))

    def test_siren_candidates_reject_decimal_currency_amounts_only(self):
        amounts = (
            "732 829 320,00 EUR",
            "732.829.320,00 €",
            "732829320.00 EUR",
        )
        for amount in amounts:
            with self.subTest(amount=amount):
                self.assertEqual(extract_siren_candidates(amount), ())

        self.assertEqual(
            extract_siren_candidates("SIREN 732829320, société active."),
            ("732829320",),
        )


if __name__ == "__main__":
    unittest.main()
