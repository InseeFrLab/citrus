import copy
import inspect
import json
import unittest
from unittest.mock import patch

from src.bodacc import normalize_bodacc_announcement
from src.routing import (
    BeneficiaryCount,
    BeneficiaryCreation,
    FUSION_SUBTYPE_PROMPT_VERSION,
    FUSION_SUBTYPE_TAXONOMY_VERSION,
    FusionSubtype,
    FusionSubtypeLLMError,
    FusionSubtypeOutputError,
    FusionSubtypeResult,
    FusionSubtypeRouter,
    TransferScope,
    TransferorFate,
    build_fusion_subtype_context,
    fusion_subtype_router,
    is_semantically_consistent,
    semantic_consistency_issues,
    validate_fusion_subtype_output,
)
from src.routing.fusion_prompt import build_fusion_subtype_messages
import src.routing.fusion_subtype as fusion_subtype_module


def party(identifier_key, siren, name):
    return {
        "numeroImmatriculation": {identifier_key: siren},
        "denomination": name,
    }


def raw_announcement(**overrides):
    act_description = (
        "Avis de projet de fusion : la société absorbée transmettrait "
        "l'intégralité de son patrimoine à la société absorbante."
    )
    payload = {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": party(
                "numeroIdentification", "732 829 320", "BENEFICIAIRE"
            )
        },
        "listeprecedentproprietaire": {
            "personne": party(
                "numeroIdentification", "123 456 782", "APPORTEUSE"
            )
        },
        "listeprecedentexploitant": {
            "personne": party(
                "numeroIdentification", "356 000 000", "EXPLOITANT"
            )
        },
        "listeetablissements": {
            "etablissement": {
                "origineFonds": "Fonds reçu par apport"
            }
        },
        "acte": {
            "descriptif": act_description,
            "vente": {
                "descriptif": act_description,
                "categorieVente": "Donnée brute hors contexte projeté",
            },
            "immatriculation": {
                "categorieImmatriculation": "Création",
                "dateImmatriculation": "2026-06-01",
            },
            "raw_nested": "RAW_ACTE_SENTINEL",
        },
        "modificationsGenerales": {
            "descriptif": "La réalisation définitive emportera dissolution."
        },
        "dateparution": "2026-08-27",
        "dateCommencementActivite": "ANNOTATED_LIKE_SOURCE_DATE",
        "dateEffet": "UNNEEDED_EFFECT_DATE",
        "unrelated_raw_metadata": "RAW_PAYLOAD_SENTINEL",
        "type_op": "REFERENCE_TYPE_SENTINEL",
        "expected_subtype": "EXPECTED_SUBTYPE_SENTINEL",
        "reference_family": "REFERENCE_FAMILY_SENTINEL",
        "siren_cedante": "ANNOTATED_CEDANT_SENTINEL",
        "siren_beneficiaire": "ANNOTATED_BENEFICIARY_SENTINEL",
        "date_creation_op": "ANNOTATED_CREATION_DATE_SENTINEL",
        "date_effet_comptable_op": "ANNOTATED_EFFECT_DATE_SENTINEL",
        "date_realisation_juridique_op": "ANNOTATED_LEGAL_DATE_SENTINEL",
        "montant": "ANNOTATED_AMOUNT_SENTINEL",
    }
    payload.update(overrides)
    return payload


def response(
    subtype="AB",
    *,
    transfer_scope="TOTAL",
    transferor_fate="DISAPPEARS",
    beneficiary_creation="EXISTING",
    beneficiary_count="ONE",
    evidence=None,
    reason="Le bénéficiaire absorbant préexiste et le cédant disparaît.",
):
    return json.dumps(
        {
            "subtype": subtype,
            "transfer_scope": transfer_scope,
            "transferor_fate": transferor_fate,
            "beneficiary_creation": beneficiary_creation,
            "beneficiary_count": beneficiary_count,
            "evidence": (
                ["transmission de l'intégralité du patrimoine"]
                if evidence is None
                else evidence
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


class FusionSubtypeTest(unittest.TestCase):
    def test_public_api_and_exact_enum_values(self):
        self.assertEqual(
            [value.value for value in FusionSubtype],
            ["FU", "AB", "SP", "ST", "AP", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in TransferScope],
            ["TOTAL", "PARTIAL", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in TransferorFate],
            ["DISAPPEARS", "SURVIVES", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in BeneficiaryCreation],
            ["NEW", "EXISTING", "MIXED_OR_UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in BeneficiaryCount],
            ["ONE", "MULTIPLE", "UNKNOWN"],
        )
        self.assertEqual(
            FUSION_SUBTYPE_PROMPT_VERSION, "fusion-subtype-v1"
        )
        self.assertEqual(
            FUSION_SUBTYPE_TAXONOMY_VERSION,
            "fusion-subtype-routing-v1",
        )
        self.assertIsInstance(fusion_subtype_router, FusionSubtypeRouter)
        self.assertEqual(
            list(inspect.signature(FusionSubtypeRouter.route).parameters),
            ["self", "announcement"],
        )

    def test_context_is_deterministic_compact_and_deduplicated(self):
        normalized = normalize_bodacc_announcement(raw_announcement())

        first = build_fusion_subtype_context(normalized)
        second = build_fusion_subtype_context(normalized)

        act_description = raw_announcement()["acte"]["descriptif"]
        modification_description = (
            "La réalisation définitive emportera dissolution."
        )
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            {
                "dialect": "RCS-A",
                "main_party": {
                    "name": "BENEFICIAIRE",
                    "siren": "732829320",
                },
                "act_description": act_description,
                "modification_description": modification_description,
                "all_descriptions": [
                    act_description,
                    modification_description,
                ],
                "previous_owners": [
                    {"name": "APPORTEUSE", "siren": "123456782"}
                ],
                "immatriculation_category": "Création",
                "immatriculation_date": "2026-06-01",
                "publication_date": "2026-08-27",
            },
        )
        for excluded in (
            "sale_description",
            "previous_operators",
            "origin_funds",
            "commencement_date",
            "effect_date",
            "raw_payload",
        ):
            self.assertNotIn(excluded, first)

    def test_router_accepts_raw_mapping_only(self):
        fake = RecordingAsk()
        normalized = normalize_bodacc_announcement(raw_announcement())

        with self.assertRaises(TypeError):
            FusionSubtypeRouter(fake).route(normalized)
        with self.assertRaises(TypeError):
            FusionSubtypeRouter(fake).route([raw_announcement()])
        with self.assertRaises(TypeError):
            build_fusion_subtype_context(raw_announcement())

        self.assertEqual(fake.calls, [])

    def test_normalizes_before_prompting_and_preserves_raw_input(self):
        payload = raw_announcement(
            acte=json.dumps(raw_announcement()["acte"], ensure_ascii=False)
        )
        before = copy.deepcopy(payload)
        fake = RecordingAsk()

        with patch.object(
            fusion_subtype_module,
            "normalize_bodacc_announcement",
            wraps=normalize_bodacc_announcement,
        ) as normalize_mock:
            result = FusionSubtypeRouter(fake).route(payload)

        normalize_mock.assert_called_once_with(payload)
        self.assertEqual(result.subtype, FusionSubtype.AB)
        self.assertEqual(payload, before)
        self.assertEqual(len(fake.calls), 1)

    def test_act_description_reaches_llm_without_raw_or_label_leakage(self):
        legal_wording = (
            "Projet de scission totale : la société scindée sera dissoute "
            "sans liquidation et son patrimoine réparti entre deux sociétés "
            "bénéficiaires."
        )
        acte = {
            "descriptif": legal_wording,
            "vente": {"categorieVente": "RAW_VENTE_SENTINEL"},
            "unrelated": "RAW_ACTE_SENTINEL",
        }
        payload = raw_announcement(
            acte=json.dumps(acte, ensure_ascii=False)
        )
        fake = RecordingAsk(
            response(
                "ST",
                beneficiary_creation="MIXED_OR_UNKNOWN",
                beneficiary_count="MULTIPLE",
            )
        )

        FusionSubtypeRouter(fake).route(payload)

        serialized = json.dumps(fake.calls[0][0], ensure_ascii=False)
        self.assertIn(legal_wording, serialized)
        self.assertIn("act_description", serialized)
        for forbidden in (
            "RAW_VENTE_SENTINEL",
            "RAW_ACTE_SENTINEL",
            "RAW_PAYLOAD_SENTINEL",
            "REFERENCE_TYPE_SENTINEL",
            "EXPECTED_SUBTYPE_SENTINEL",
            "REFERENCE_FAMILY_SENTINEL",
            "ANNOTATED_CEDANT_SENTINEL",
            "ANNOTATED_BENEFICIARY_SENTINEL",
            "ANNOTATED_CREATION_DATE_SENTINEL",
            "ANNOTATED_EFFECT_DATE_SENTINEL",
            "ANNOTATED_LEGAL_DATE_SENTINEL",
            "ANNOTATED_AMOUNT_SENTINEL",
            "ANNOTATED_LIKE_SOURCE_DATE",
            "UNNEEDED_EFFECT_DATE",
            "type_op",
            "date_creation_op",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_prompt_defines_all_subtypes_axes_and_abstention(self):
        messages = build_fusion_subtype_messages({"dialect": "UNKNOWN"})
        prompt = "\n".join(message["content"] for message in messages)

        for subtype in FusionSubtype:
            self.assertIn(subtype.value, prompt)
        for value in (
            "transfer_scope",
            "transferor_fate",
            "beneficiary_creation",
            "beneficiary_count",
            "TOTAL",
            "PARTIAL",
            "DISAPPEARS",
            "SURVIVES",
            "NEW",
            "EXISTING",
            "MIXED_OR_UNKNOWN",
            "ONE",
            "MULTIPLE",
        ):
            self.assertIn(value, prompt)
        self.assertIn("au lieu de deviner", prompt)
        self.assertIn("exactement les champs", prompt)

    def test_prompt_explains_fu_vs_ab_by_beneficiary_creation(self):
        prompt = build_fusion_subtype_messages({})[0]["content"]

        self.assertIn("FU vs AB", prompt)
        self.assertIn("créé pour l'opération (FU)", prompt)
        self.assertIn("préexistant (AB)", prompt)
        self.assertIn("seul mot « fusion » ne suffit pas", prompt)

    def test_prompt_explains_sp_vs_ap_with_surviving_transferor(self):
        prompt = build_fusion_subtype_messages({})[0]["content"]

        self.assertIn("SP vs AP", prompt)
        self.assertIn("survie du cédant", prompt)
        self.assertIn("créé pour l'opération (SP)", prompt)
        self.assertIn("préexistant (AP)", prompt)

    def test_prompt_explains_total_scission_disappearance_and_split(self):
        prompt = build_fusion_subtype_messages({})[0]["content"]

        self.assertIn("ST :", prompt)
        self.assertIn("cédant disparaît", prompt)
        self.assertIn("plusieurs bénéficiaires", prompt)
        self.assertIn("répartit son patrimoine", prompt)

    def test_valid_result_parsing_for_every_subtype(self):
        axes = {
            "FU": ("TOTAL", "DISAPPEARS", "NEW", "ONE"),
            "AB": ("TOTAL", "DISAPPEARS", "EXISTING", "ONE"),
            "SP": ("PARTIAL", "SURVIVES", "NEW", "MULTIPLE"),
            "ST": (
                "TOTAL",
                "DISAPPEARS",
                "MIXED_OR_UNKNOWN",
                "MULTIPLE",
            ),
            "AP": ("PARTIAL", "SURVIVES", "EXISTING", "ONE"),
        }
        for subtype, values in axes.items():
            with self.subTest(subtype=subtype):
                result = validate_fusion_subtype_output(
                    response(
                        subtype,
                        transfer_scope=values[0],
                        transferor_fate=values[1],
                        beneficiary_creation=values[2],
                        beneficiary_count=values[3],
                    )
                )
                self.assertIsInstance(result, FusionSubtypeResult)
                self.assertEqual(result.subtype, FusionSubtype(subtype))
                self.assertTrue(is_semantically_consistent(result))

    def test_semantic_unknown_is_valid_and_distinct_from_failure(self):
        result = validate_fusion_subtype_output(
            response(
                "UNKNOWN",
                transfer_scope="UNKNOWN",
                transferor_fate="UNKNOWN",
                beneficiary_creation="MIXED_OR_UNKNOWN",
                beneficiary_count="UNKNOWN",
                evidence=[],
                reason="Les faits ne permettent pas de distinguer les types.",
            )
        )

        self.assertEqual(result.subtype, FusionSubtype.UNKNOWN)
        self.assertEqual(result.evidence, ())
        self.assertTrue(is_semantically_consistent(result))

    def test_invalid_subtype_is_a_technical_output_failure(self):
        for subtype in ("FUSION", "ABSORPTION", "SCISSION", " AB "):
            with self.subTest(subtype=subtype):
                with self.assertRaises(FusionSubtypeOutputError) as raised:
                    validate_fusion_subtype_output(response(subtype))
                self.assertEqual(raised.exception.code, "invalid_subtype")

    def test_invalid_axis_values_are_technical_output_failures(self):
        invalid_axes = {
            "transfer_scope": "COMPLETE",
            "transferor_fate": "DISSOLVED",
            "beneficiary_creation": "PREEXISTING",
            "beneficiary_count": "TWO",
        }
        for field, value in invalid_axes.items():
            with self.subTest(field=field):
                payload = json.loads(response())
                payload[field] = value
                with self.assertRaises(FusionSubtypeOutputError) as raised:
                    validate_fusion_subtype_output(json.dumps(payload))
                self.assertEqual(
                    raised.exception.code, "invalid_semantic_axis"
                )

    def test_malformed_or_wrapped_json_is_a_technical_failure(self):
        for raw in (
            "not json",
            "```json\n" + response() + "\n```",
            "préface " + response(),
            "",
        ):
            with self.subTest(raw=raw):
                with self.assertRaises(FusionSubtypeOutputError) as raised:
                    validate_fusion_subtype_output(raw)
                self.assertEqual(raised.exception.code, "invalid_json")

    def test_missing_extra_wrong_typed_and_duplicate_fields_are_rejected(self):
        base = json.loads(response())
        missing = dict(base)
        missing.pop("reason")
        extra = {**base, "confidence": 1}
        wrong_types = []
        for field, value in (
            ("subtype", 1),
            ("transfer_scope", None),
            ("transferor_fate", False),
            ("beneficiary_creation", []),
            ("beneficiary_count", 2),
            ("evidence", "indice"),
            ("reason", None),
        ):
            payload = dict(base)
            payload[field] = value
            wrong_types.append(payload)
        for payload in (missing, extra, *wrong_types):
            with self.subTest(payload=payload):
                with self.assertRaises(FusionSubtypeOutputError) as raised:
                    validate_fusion_subtype_output(json.dumps(payload))
                self.assertEqual(raised.exception.code, "invalid_schema")

        duplicate = response().replace(
            '"subtype": "AB",',
            '"subtype": "AB", "subtype": "FU",',
            1,
        )
        with self.assertRaises(FusionSubtypeOutputError) as raised:
            validate_fusion_subtype_output(duplicate)
        self.assertEqual(raised.exception.code, "invalid_schema")

        non_standard = response().replace(
            '"reason": "Le bénéficiaire',
            '"reason": NaN, "discarded": "Le bénéficiaire',
            1,
        )
        with self.assertRaises(FusionSubtypeOutputError) as raised:
            validate_fusion_subtype_output(non_standard)
        self.assertEqual(raised.exception.code, "invalid_json")

    def test_evidence_allows_five_items_but_enforces_brief_limits(self):
        valid = validate_fusion_subtype_output(
            response(
                evidence=[" a ", "b", "c", "d", "e"],
                reason="  Justification concise.  ",
            )
        )
        self.assertEqual(valid.evidence, ("a", "b", "c", "d", "e"))
        self.assertEqual(valid.reason, "Justification concise.")

        invalid_payloads = (
            {**json.loads(response()), "evidence": list("abcdef")},
            {**json.loads(response()), "evidence": ["  "]},
            {**json.loads(response()), "evidence": [1]},
            {**json.loads(response()), "evidence": ["x" * 301]},
            {**json.loads(response()), "reason": "x" * 501},
            {**json.loads(response()), "reason": "  "},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(FusionSubtypeOutputError):
                    validate_fusion_subtype_output(json.dumps(payload))

    def test_consistency_helper_catches_each_obvious_contradiction(self):
        cases = (
            (
                response("FU", beneficiary_creation="EXISTING"),
                "beneficiary_creation:expected=NEW,actual=EXISTING",
            ),
            (
                response("AB", transferor_fate="SURVIVES"),
                "transferor_fate:expected=DISAPPEARS,actual=SURVIVES",
            ),
            (
                response(
                    "SP",
                    transfer_scope="PARTIAL",
                    transferor_fate="SURVIVES",
                    beneficiary_creation="EXISTING",
                ),
                "beneficiary_creation:expected=NEW,actual=EXISTING",
            ),
            (
                response(
                    "AP",
                    transfer_scope="PARTIAL",
                    transferor_fate="DISAPPEARS",
                ),
                "transferor_fate:expected=SURVIVES,actual=DISAPPEARS",
            ),
            (
                response(
                    "ST",
                    beneficiary_creation="MIXED_OR_UNKNOWN",
                    beneficiary_count="ONE",
                ),
                "beneficiary_count:expected=MULTIPLE,actual=ONE",
            ),
        )
        for raw, expected_issue in cases:
            with self.subTest(expected_issue=expected_issue):
                result = validate_fusion_subtype_output(raw)
                chosen_subtype = result.subtype
                issues = semantic_consistency_issues(result)

                self.assertIn(expected_issue, issues)
                self.assertFalse(is_semantically_consistent(result))
                self.assertEqual(result.subtype, chosen_subtype)

    def test_unknown_axis_values_do_not_automatically_conflict(self):
        for subtype in ("FU", "AB", "SP", "ST", "AP"):
            with self.subTest(subtype=subtype):
                result = validate_fusion_subtype_output(
                    response(
                        subtype,
                        transfer_scope="UNKNOWN",
                        transferor_fate="UNKNOWN",
                        beneficiary_creation="MIXED_OR_UNKNOWN",
                        beneficiary_count="UNKNOWN",
                    )
                )
                self.assertEqual(semantic_consistency_issues(result), ())
                self.assertTrue(is_semantically_consistent(result))

    def test_injected_fake_llm_receives_temperature_zero(self):
        fake = RecordingAsk(
            response(
                "AP",
                transfer_scope="PARTIAL",
                transferor_fate="SURVIVES",
            )
        )

        result = FusionSubtypeRouter(fake).route(raw_announcement())

        self.assertEqual(result.subtype, FusionSubtype.AP)
        self.assertEqual(fake.calls[0][1], {"temperature": 0})

    def test_llm_exception_remains_distinct_from_invalid_output(self):
        fake = RecordingAsk(error=ConnectionError("offline"))

        with self.assertRaises(FusionSubtypeLLMError) as raised:
            FusionSubtypeRouter(fake).route(raw_announcement())

        self.assertEqual(raised.exception.code, "ConnectionError")
        self.assertNotIsInstance(raised.exception, FusionSubtypeOutputError)

    def test_router_has_no_family_router_or_operation_skill_dependency(self):
        source = inspect.getsource(fusion_subtype_module)

        self.assertNotIn("FamilyRouter", source)
        self.assertNotIn("family_router", source)
        self.assertNotIn("src.operation", source)
        self.assertNotIn("vente_skill", source)
        self.assertNotIn("location_gerance_skill", source)
        self.assertNotIn("transmission_patrimoine_skill", source)


if __name__ == "__main__":
    unittest.main()
