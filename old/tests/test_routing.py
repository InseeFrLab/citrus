import copy
import inspect
import json
import unittest
from unittest.mock import patch

from src.bodacc import normalize_bodacc_announcement
from src.routing import (
    FamilyRouter,
    ROUTER_PROMPT_VERSION,
    RoutingFamily,
    RoutingLLMError,
    RoutingOutputError,
    RoutingResult,
    build_routing_context,
    validate_routing_output,
)
from src.routing.prompt import build_family_routing_messages
import src.routing.router as router_module


def party(identifier_key, siren, name):
    return {
        "numeroImmatriculation": {identifier_key: siren},
        "denomination": name,
    }


def raw_announcement(**overrides):
    description = "Cession d'un fonds de commerce au prix de 120000 euros."
    payload = {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": party(
                "numeroIdentification", "732 829 320", "ACQUEREUR"
            )
        },
        "listeprecedentproprietaire": {
            "personne": party(
                "numeroIdentification", "123 456 782", "VENDEUR"
            )
        },
        "listeprecedentexploitant": {
            "personne": party(
                "numeroIdentification", "356 000 000", "EXPLOITANT"
            )
        },
        "listeetablissements": {
            "etablissement": [
                {"origineFonds": "Achat au prix stipulé de 120000 EUR"},
                {"origineFonds": "Achat au prix stipulé de 120000 EUR"},
            ]
        },
        "acte": {
            "vente": {"descriptif": description},
            "immatriculation": {
                "categorieImmatriculation": "Achat d'un fonds"
            },
        },
        "modificationsGenerales": {"descriptif": description},
        "dateparution": "2026-08-27",
        "unrelated_raw_metadata": "RAW_PAYLOAD_SENTINEL",
        "type_op": "REFERENCE_LABEL_SENTINEL",
        "reference_family": "EXPECTED_FAMILY_SENTINEL",
        "date_creation_op": "ANNOTATED_DATE_SENTINEL",
    }
    payload.update(overrides)
    return payload


def response(family="VE", evidence=None, reason="La cession est explicite."):
    return json.dumps(
        {
            "family": family,
            "evidence": (
                ["Cession d'un fonds"] if evidence is None else evidence
            ),
            "reason": reason,
        },
        ensure_ascii=False,
    )


class RecordingAsk:
    def __init__(self, raw_response=None, error=None):
        self.raw_response = raw_response or response()
        self.error = error
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.raw_response


class RoutingTest(unittest.TestCase):
    def test_public_api_and_exact_family_values(self):
        self.assertEqual(
            [family.value for family in RoutingFamily],
            ["VE", "LG", "TP", "FUSION_FAMILY", "UNKNOWN"],
        )
        self.assertEqual(ROUTER_PROMPT_VERSION, "family-router-v1")
        self.assertEqual(
            list(inspect.signature(FamilyRouter.route).parameters),
            ["self", "announcement"],
        )

    def test_context_is_deterministic_compact_and_deduplicated(self):
        normalized = normalize_bodacc_announcement(raw_announcement())

        first = build_routing_context(normalized)
        second = build_routing_context(normalized)

        self.assertEqual(first, second)
        self.assertEqual(
            first,
            {
                "dialect": "RCS-A",
                "main_party": {"name": "ACQUEREUR", "siren": "732829320"},
                "sale_description": (
                    "Cession d'un fonds de commerce au prix de 120000 euros."
                ),
                "origin_funds": [
                    "Achat au prix stipulé de 120000 EUR"
                ],
                "immatriculation_category": "Achat d'un fonds",
                "previous_owners": [
                    {"name": "VENDEUR", "siren": "123456782"}
                ],
                "previous_operators": [
                    {"name": "EXPLOITANT", "siren": "356000000"}
                ],
            },
        )
        self.assertNotIn("modification_description", first)

    def test_act_description_precedes_and_deduplicates_sale_description(self):
        act_text = "Projet commun de fusion nationale"
        modification_text = "Modification distincte"
        payload = raw_announcement(
            acte={
                "descriptif": act_text,
                "vente": {"descriptif": act_text},
            },
            modificationsGenerales={"descriptif": modification_text},
        )

        context = build_routing_context(
            normalize_bodacc_announcement(payload)
        )

        self.assertEqual(context["act_description"], act_text)
        self.assertNotIn("sale_description", context)
        self.assertEqual(
            context["modification_description"], modification_text
        )
        self.assertEqual(
            [
                key
                for key in context
                if key.endswith("_description")
            ],
            ["act_description", "modification_description"],
        )

    def test_complex_act_wording_reaches_llm_without_raw_acte_json(self):
        cases = (
            (
                "A20230191265",
                "Avis au Bodacc relatif au projet commun de fusion nationale. "
                "Société absorbante et société absorbée n° 1.",
                False,
            ),
            (
                "A20230150146",
                "Sociétés absorbante et absorbée : la fusion prendrait effet "
                "à la réalisation définitive de la fusion.",
                True,
            ),
            (
                "A202301901462",
                "Projet commun de scission nationale. Société scindée et "
                "société bénéficiaire de la scission.",
                False,
            ),
            (
                "A202302081505",
                "AVIS DE PROJET D'APPORT PARTIEL D'ACTIF : société apporteuse "
                "et société bénéficiaire.",
                True,
            ),
        )
        fake = RecordingAsk(response("FUSION_FAMILY"))

        for reference, wording, encoded in cases:
            with self.subTest(reference=reference):
                raw_act_sentinel = f"RAW_ACTE_SENTINEL_{reference}"
                acte = {
                    "descriptif": wording,
                    "vente": {
                        "categorieVente": (
                            "Autre achat, apport, attribution, immatriculation"
                        )
                    },
                    "unrelated": raw_act_sentinel,
                }
                payload = {
                    "id": reference,
                    "registre": "RCS-A",
                    "listepersonnes": {
                        "personne": party(
                            "numeroIdentification",
                            "732 829 320",
                            "SOCIETE PRINCIPALE",
                        )
                    },
                    "acte": (
                        json.dumps(acte, ensure_ascii=False)
                        if encoded
                        else acte
                    ),
                    "type_op": "REFERENCE_LABEL_SENTINEL",
                    "reference_family": "EXPECTED_FAMILY_SENTINEL",
                    "date_creation_op": "ANNOTATED_DATE_SENTINEL",
                    "siren_cedante": "ANNOTATED_CEDANT_SENTINEL",
                    "siren_beneficiaire": "ANNOTATED_BENEFICIARY_SENTINEL",
                    "date_effet_comptable_op": "ANNOTATED_EFFECT_SENTINEL",
                    "montant": "ANNOTATED_AMOUNT_SENTINEL",
                }

                FamilyRouter(fake).route(payload)

                serialized = json.dumps(
                    fake.calls[-1][0], ensure_ascii=False
                )
                self.assertIn(wording, serialized)
                self.assertIn("act_description", serialized)
                for forbidden in (
                    raw_act_sentinel,
                    "categorieVente",
                    "REFERENCE_LABEL_SENTINEL",
                    "EXPECTED_FAMILY_SENTINEL",
                    "ANNOTATED_DATE_SENTINEL",
                    "ANNOTATED_CEDANT_SENTINEL",
                    "ANNOTATED_BENEFICIARY_SENTINEL",
                    "ANNOTATED_EFFECT_SENTINEL",
                    "ANNOTATED_AMOUNT_SENTINEL",
                ):
                    self.assertNotIn(forbidden, serialized)

    def test_context_builder_requires_normalized_input(self):
        with self.assertRaises(TypeError):
            build_routing_context(raw_announcement())

    def test_router_normalizes_before_prompting_and_preserves_raw_input(self):
        payload = raw_announcement(
            acte=json.dumps(raw_announcement()["acte"], ensure_ascii=False)
        )
        before = copy.deepcopy(payload)
        fake = RecordingAsk()

        with patch.object(
            router_module,
            "normalize_bodacc_announcement",
            wraps=normalize_bodacc_announcement,
        ) as normalize_mock:
            result = FamilyRouter(fake).route(payload)

        normalize_mock.assert_called_once_with(payload)
        self.assertEqual(result.family, RoutingFamily.VE)
        self.assertEqual(payload, before)
        self.assertEqual(len(fake.calls), 1)

    def test_prompt_uses_only_projected_facts_without_raw_or_reference_leakage(self):
        fake = RecordingAsk()

        FamilyRouter(fake).route(raw_announcement())

        messages = fake.calls[0][0]
        serialized = json.dumps(messages, ensure_ascii=False)
        for forbidden in (
            "RAW_PAYLOAD_SENTINEL",
            "REFERENCE_LABEL_SENTINEL",
            "EXPECTED_FAMILY_SENTINEL",
            "ANNOTATED_DATE_SENTINEL",
            "unrelated_raw_metadata",
            "date_creation_op",
            "reference_family",
            "type_op",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertIn("Cession d'un fonds", serialized)
        self.assertNotIn("dateparution", serialized)

    def test_prompt_contains_all_families_and_explicit_abstention(self):
        messages = build_family_routing_messages({"dialect": "UNKNOWN"})
        prompt = "\n".join(message["content"] for message in messages)

        for family in RoutingFamily:
            self.assertIn(family.value, prompt)
        self.assertIn("ne devine jamais", prompt)
        self.assertIn("UNKNOWN", prompt)
        self.assertIn("exactement les champs", prompt)

    def test_all_five_semantic_outputs_are_valid_including_unknown(self):
        for family in RoutingFamily:
            with self.subTest(family=family.value):
                result = validate_routing_output(response(family.value))
                self.assertEqual(result.family, family)
                self.assertIsInstance(result, RoutingResult)
        unknown = validate_routing_output(
            response("UNKNOWN", evidence=[], reason="Faits insuffisants.")
        )
        self.assertEqual(unknown.family, RoutingFamily.UNKNOWN)
        self.assertEqual(unknown.evidence, ())

    def test_output_normalizes_outer_whitespace_only(self):
        result = validate_routing_output(
            response(
                evidence=["  dissolution sans liquidation  "],
                reason="  Une TUP est explicitement décrite.  ",
            )
        )

        self.assertEqual(result.evidence, ("dissolution sans liquidation",))
        self.assertEqual(result.reason, "Une TUP est explicitement décrite.")

    def test_malformed_or_wrapped_json_is_a_technical_output_failure(self):
        for raw in (
            "not json",
            "```json\n" + response() + "\n```",
            "préface " + response(),
            "",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(RoutingOutputError) as raised:
                    validate_routing_output(raw)
                self.assertEqual(raised.exception.code, "invalid_json")

    def test_invalid_family_is_not_coerced_to_unknown(self):
        for family in ("TUP", "SALE", "FUSION", " VE "):
            with self.subTest(family=family):
                with self.assertRaises(RoutingOutputError) as raised:
                    validate_routing_output(response(family))
                self.assertEqual(raised.exception.code, "invalid_family")

    def test_missing_extra_and_wrong_schema_fields_are_rejected(self):
        payloads = (
            {"family": "VE", "evidence": []},
            {
                "family": "VE",
                "evidence": [],
                "reason": "vente",
                "confidence": 1,
            },
            {"family": 1, "evidence": [], "reason": "vente"},
            {"family": "VE", "evidence": "vente", "reason": "vente"},
            {"family": "VE", "evidence": [], "reason": None},
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RoutingOutputError) as raised:
                    validate_routing_output(json.dumps(payload))
                self.assertEqual(raised.exception.code, "invalid_schema")

    def test_duplicate_fields_and_non_standard_constants_are_rejected(self):
        duplicate = (
            '{"family":"VE","family":"LG","evidence":[],"reason":"x"}'
        )
        with self.assertRaises(RoutingOutputError) as raised:
            validate_routing_output(duplicate)
        self.assertEqual(raised.exception.code, "invalid_schema")

        with self.assertRaises(RoutingOutputError) as raised:
            validate_routing_output(
                '{"family":"VE","evidence":[],"reason":NaN}'
            )
        self.assertEqual(raised.exception.code, "invalid_json")

    def test_evidence_and_reason_limits_are_enforced(self):
        invalid_payloads = (
            {"family": "VE", "evidence": ["a", "b", "c", "d"], "reason": "x"},
            {"family": "VE", "evidence": ["  "], "reason": "x"},
            {"family": "VE", "evidence": [1], "reason": "x"},
            {"family": "VE", "evidence": ["x" * 301], "reason": "x"},
            {"family": "VE", "evidence": [], "reason": "x" * 501},
            {"family": "VE", "evidence": [], "reason": "  "},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(RoutingOutputError):
                    validate_routing_output(json.dumps(payload))

    def test_injected_llm_is_offline_and_receives_temperature_zero(self):
        fake = RecordingAsk(response("LG"))

        result = FamilyRouter(fake).route(raw_announcement())

        self.assertEqual(result.family, RoutingFamily.LG)
        self.assertEqual(fake.calls[0][1], {"temperature": 0})

    def test_llm_exception_remains_distinct_from_invalid_output(self):
        fake = RecordingAsk(error=ConnectionError("offline"))

        with self.assertRaises(RoutingLLMError) as raised:
            FamilyRouter(fake).route(raw_announcement())

        self.assertEqual(raised.exception.code, "ConnectionError")
        self.assertNotIsInstance(raised.exception, RoutingOutputError)

    def test_router_has_no_operation_skill_dependency_or_dispatch(self):
        source = inspect.getsource(router_module)

        self.assertNotIn("src.operation", source)
        self.assertNotIn("vente_skill", source)
        self.assertNotIn("location_gerance_skill", source)
        self.assertNotIn("transmission_patrimoine_skill", source)


if __name__ == "__main__":
    unittest.main()
