import copy
import inspect
import json
import unittest

from src.bodacc import normalize_bodacc_announcement
import src.routing.fusion_semantics as semantic_module
from src.routing.fusion_semantics import (
    FUSION_SEMANTICS_PROMPT_VERSION,
    FUSION_SEMANTICS_SCHEMA_VERSION,
    FusionSemanticLLMError,
    FusionSemanticOutputError,
    FusionSemanticParser,
    LegalFamily,
    ParticipantRole,
    PartialAssetTransferWording,
    build_fusion_semantic_context,
    fusion_semantic_parser,
    validate_fusion_semantic_output,
)
from src.routing.fusion_semantics_prompt import (
    build_fusion_semantic_messages,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)


def _party(siren, name):
    return {
        "numeroImmatriculation": {"numeroIdentification": siren},
        "denomination": name,
    }


def _announcement(**overrides):
    description = (
        "Projet de fusion : APPORTEUSE 123 456 782 transmet la totalité "
        "de son patrimoine à BENEFICIAIRE 732 829 320."
    )
    payload = {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": _party("732 829 320", "BENEFICIAIRE")
        },
        "listeprecedentproprietaire": {
            "personne": _party("123 456 782", "APPORTEUSE")
        },
        "acte": {
            "descriptif": description,
            "immatriculation": {
                "categorieImmatriculation": "Création",
                "dateImmatriculation": "2026-01-02",
            },
        },
        "dateparution": "2026-08-27",
        "dateCommencementActivite": "DATE_SENTINEL",
        "montant": "AMOUNT_SENTINEL",
        "type_op": "AB",
        "expected_subtype": "LABEL_SENTINEL",
        "siren_cedante": "ANNOTATION_SENTINEL",
    }
    payload.update(overrides)
    return payload


def _response(
    *,
    legal_family="FUSION",
    transfer_scope="TOTAL",
    transferor_fate="DISAPPEARS",
    beneficiary_creation="MIXED_OR_UNKNOWN",
    partial_asset_transfer_wording="UNKNOWN",
    participants=None,
    evidence=None,
    reason="Les faits explicitement présents décrivent une fusion.",
):
    return json.dumps(
        {
            "legal_family": legal_family,
            "transfer_scope": transfer_scope,
            "transferor_fate": transferor_fate,
            "beneficiary_creation": beneficiary_creation,
            "partial_asset_transfer_wording": partial_asset_transfer_wording,
            "participants": [] if participants is None else participants,
            "evidence": ["Projet de fusion"] if evidence is None else evidence,
            "reason": reason,
        },
        ensure_ascii=False,
    )


class _RecordingAsk:
    def __init__(self, response=None, error=None):
        self.response = response or _response()
        self.error = error
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class FusionSemanticContractTest(unittest.TestCase):
    def test_public_values_and_versions(self):
        self.assertEqual(
            [value.value for value in LegalFamily],
            ["FUSION", "SCISSION", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in PartialAssetTransferWording],
            ["YES", "NO", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in ParticipantRole],
            ["TRANSFEROR", "BENEFICIARY", "BOTH_OR_UNCLEAR"],
        )
        self.assertEqual(FUSION_SEMANTICS_PROMPT_VERSION, "fusion-semantics-v1")
        self.assertEqual(FUSION_SEMANTICS_SCHEMA_VERSION, "fusion-semantics-v1")
        self.assertIsInstance(fusion_semantic_parser, FusionSemanticParser)
        self.assertEqual(
            list(inspect.signature(FusionSemanticParser.parse).parameters),
            ["self", "announcement"],
        )

    def test_prompt_has_semantic_facts_and_no_final_subtype_field(self):
        messages = build_fusion_semantic_messages({"dialect": "RCS-A"})
        combined = "\n".join(message["content"] for message in messages)
        self.assertIn("partial_asset_transfer_wording", combined)
        self.assertIn("legal_family=SCISSION", combined)
        self.assertIn("conserve le reste de son patrimoine", combined)
        self.assertIn("dans ce conflit, retourne UNKNOWN", combined)
        self.assertIn("MIXED_OR_UNKNOWN", combined)
        self.assertIn("BOTH_OR_UNCLEAR", combined)
        self.assertNotIn('"subtype"', combined)
        self.assertNotIn('"type_op"', combined)
        for final_code in ('"FU"', '"AB"', '"SP"', '"ST"', '"AP"'):
            self.assertNotIn(final_code, combined)

    def test_context_is_compact_deterministic_and_excludes_dates_amounts_labels(self):
        normalized = normalize_bodacc_announcement(_announcement())
        first = build_fusion_semantic_context(normalized)
        second = build_fusion_semantic_context(normalized)
        serialized = json.dumps(first, ensure_ascii=False)

        self.assertEqual(first, second)
        self.assertEqual(first["dialect"], "RCS-A")
        self.assertEqual(first["main_party"]["siren"], "732829320")
        self.assertEqual(first["previous_owners"][0]["siren"], "123456782")
        self.assertEqual(first["immatriculation_category"], "Création")
        for forbidden in (
            "publication_date",
            "immatriculation_date",
            "commencement_date",
            "montant",
            "type_op",
            "expected_subtype",
            "LABEL_SENTINEL",
            "DATE_SENTINEL",
            "AMOUNT_SENTINEL",
            "ANNOTATION_SENTINEL",
            "raw_payload",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_parser_normalizes_without_mutating_and_uses_temperature_zero(self):
        participants = [
            {
                "siren": "123456782",
                "name": "APPORTEUSE",
                "role": "TRANSFEROR",
            },
            {
                "siren": "732829320",
                "name": "Bénéficiaire",
                "role": "BENEFICIARY",
            },
        ]
        ask = _RecordingAsk(
            _response(
                participants=participants,
                beneficiary_creation="MIXED_OR_UNKNOWN",
            )
        )
        raw = _announcement()
        before = copy.deepcopy(raw)

        result = FusionSemanticParser(ask).parse(raw)

        self.assertEqual(raw, before)
        self.assertEqual(result.legal_family, LegalFamily.FUSION)
        self.assertEqual(result.transfer_scope, TransferScope.TOTAL)
        self.assertEqual(result.transferor_fate, TransferorFate.DISAPPEARS)
        self.assertEqual(
            result.beneficiary_creation,
            BeneficiaryCreation.MIXED_OR_UNKNOWN,
        )
        self.assertEqual(result.participants[0].siren, "123456782")
        self.assertEqual(result.participants[1].role, ParticipantRole.BENEFICIARY)
        self.assertEqual(len(ask.calls), 1)
        self.assertEqual(ask.calls[0][1], {"temperature": 0})

    def test_unknown_values_are_valid_and_preserved(self):
        result = validate_fusion_semantic_output(
            _response(
                legal_family="UNKNOWN",
                transfer_scope="UNKNOWN",
                transferor_fate="UNKNOWN",
                beneficiary_creation="MIXED_OR_UNKNOWN",
                evidence=[],
                reason="L'annonce ne permet pas de trancher.",
            )
        )
        self.assertEqual(result.legal_family, LegalFamily.UNKNOWN)
        self.assertEqual(result.transfer_scope, TransferScope.UNKNOWN)
        self.assertEqual(result.transferor_fate, TransferorFate.UNKNOWN)
        self.assertEqual(
            result.beneficiary_creation,
            BeneficiaryCreation.MIXED_OR_UNKNOWN,
        )

    def test_partial_asset_transfer_facts_do_not_emit_a_final_code(self):
        result = validate_fusion_semantic_output(
            _response(
                legal_family="UNKNOWN",
                partial_asset_transfer_wording="YES",
                transfer_scope="PARTIAL",
                transferor_fate="SURVIVES",
                beneficiary_creation="EXISTING",
            )
        )
        self.assertEqual(result.legal_family, LegalFamily.UNKNOWN)
        self.assertEqual(
            result.partial_asset_transfer_wording, PartialAssetTransferWording.YES
        )
        self.assertFalse(hasattr(result, "subtype"))
        self.assertFalse(hasattr(result, "type_op"))
    def test_total_scission_can_keep_transferor_fate_unknown(self):
        result = validate_fusion_semantic_output(
            _response(
                legal_family="SCISSION",
                transfer_scope="TOTAL",
                transferor_fate="UNKNOWN",
                beneficiary_creation="MIXED_OR_UNKNOWN",
                reason=(
                    "La scission et plusieurs bénéficiaires sont explicites, "
                    "mais la disparition du cédant ne l'est pas."
                ),
            )
        )
        self.assertEqual(result.legal_family, LegalFamily.SCISSION)
        self.assertEqual(result.transfer_scope, TransferScope.TOTAL)
        self.assertEqual(result.transferor_fate, TransferorFate.UNKNOWN)


class FusionSemanticValidationTest(unittest.TestCase):
    def test_strict_json_and_exact_top_level_schema(self):
        fence = chr(96) * 3
        invalid_responses = [
            "",
            fence + "json\n" + _response() + "\n" + fence,
            "[]",
            _response() + "\ntexte",
            _response().replace(
                '"legal_family": "FUSION"',
                '"legal_family": "FUSION", "legal_family": "SCISSION"',
            ),
        ]
        payload = json.loads(_response())
        payload["subtype"] = "AB"
        invalid_responses.append(json.dumps(payload))
        for raw in invalid_responses:
            with self.subTest(raw=raw[:30]):
                with self.assertRaises(FusionSemanticOutputError) as raised:
                    validate_fusion_semantic_output(raw)
                self.assertEqual(raised.exception.raw_response, raw)

    def test_invalid_enums_remain_technical_errors(self):
        cases = [
            ("legal_family", "ABSORPTION", "invalid_legal_family"),
            ("partial_asset_transfer_wording", "MAYBE", "invalid_semantic_axis"),
            ("transfer_scope", "COMPLETE", "invalid_semantic_axis"),
            ("transferor_fate", "DISSOLVED", "invalid_semantic_axis"),
            ("beneficiary_creation", "OLD", "invalid_semantic_axis"),
        ]
        for field, value, code in cases:
            payload = json.loads(_response())
            payload[field] = value
            raw = json.dumps(payload)
            with self.subTest(field=field):
                with self.assertRaises(FusionSemanticOutputError) as raised:
                    validate_fusion_semantic_output(raw)
                self.assertEqual(raised.exception.code, code)
                self.assertEqual(raised.exception.raw_response, raw)

    def test_participant_siren_and_name_must_be_source_grounded(self):
        source = json.dumps(
            {
                "description": (
                    "APPORTEUSE 123 456 782 apporte à BENEFICIAIRE "
                    "732 829 320"
                )
            }
        )
        valid = validate_fusion_semantic_output(
            _response(
                participants=[
                    {
                        "siren": "123456782",
                        "name": "Apporteuse",
                        "role": "TRANSFEROR",
                    }
                ]
            ),
            source_text=source,
        )
        self.assertEqual(valid.participants[0].siren, "123456782")

        cases = [
            (
                {
                    "siren": "111111111",
                    "name": "APPORTEUSE",
                    "role": "TRANSFEROR",
                },
                "invalid_participant_siren",
            ),
            (
                {
                    "siren": None,
                    "name": "SOCIETE INVENTEE",
                    "role": "BENEFICIARY",
                },
                "invalid_participant_name",
            ),
            (
                {
                    "siren": None,
                    "name": None,
                    "role": "BOTH_OR_UNCLEAR",
                },
                "invalid_schema",
            ),
        ]
        for participant, code in cases:
            with self.subTest(code=code):
                with self.assertRaises(FusionSemanticOutputError) as raised:
                    validate_fusion_semantic_output(
                        _response(participants=[participant]),
                        source_text=source,
                    )
                self.assertEqual(raised.exception.code, code)

    def test_participant_role_and_shape_are_strict(self):
        cases = [
            [{"siren": None, "name": "APPORTEUSE", "role": "UNKNOWN"}],
            [
                {
                    "siren": None,
                    "name": "APPORTEUSE",
                    "role": "TRANSFEROR",
                    "extra": True,
                }
            ],
            ["APPORTEUSE"],
        ]
        for participants in cases:
            with self.subTest(participants=participants):
                with self.assertRaises(FusionSemanticOutputError):
                    validate_fusion_semantic_output(
                        _response(participants=participants),
                        source_text='{"name":"APPORTEUSE"}',
                    )

    def test_bounded_participants_evidence_and_reason(self):
        participant = {
            "siren": None,
            "name": "APPORTEUSE",
            "role": "TRANSFEROR",
        }
        validate_fusion_semantic_output(
            _response(participants=[participant] * 20, evidence=["x"] * 6),
            source_text='{"name":"APPORTEUSE"}',
        )
        for raw in (
            _response(participants=[participant] * 21),
            _response(evidence=["x"] * 7),
            _response(evidence=["x" * 301]),
            _response(reason="x" * 501),
        ):
            with self.subTest(size=len(raw)):
                with self.assertRaises(FusionSemanticOutputError):
                    validate_fusion_semantic_output(
                        raw, source_text='{"name":"APPORTEUSE"}'
                    )

    def test_llm_failure_is_distinct_and_source_has_no_label_dependency(self):
        parser = FusionSemanticParser(
            _RecordingAsk(error=TimeoutError("offline timeout"))
        )
        with self.assertRaises(FusionSemanticLLMError) as raised:
            parser.parse(_announcement())
        self.assertEqual(raised.exception.code, "TimeoutError")
        source = inspect.getsource(semantic_module)
        self.assertNotIn("type_op", source)
        self.assertNotIn("reference_type", source)
        self.assertNotIn("FusionSubtype", source)


if __name__ == "__main__":
    unittest.main()
