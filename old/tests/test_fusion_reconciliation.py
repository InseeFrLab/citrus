from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
import inspect
import unittest

from src.bodacc import normalize_bodacc_announcement
from src.routing.fusion_reconciliation import (
    DESCRIPTION_FINGERPRINT_VERSION,
    FUSION_RECONCILIATION_VERSION,
    HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED,
    HISTORICAL_ISOLATED_AP_FALLBACK_NOTE,
    FinalFusionType,
    FusionProvisionalRecord,
    ProvisionalType,
    build_fusion_provisional,
    canonicalize_description,
    description_fingerprint,
    reconcile_fusion_family,
)
from src.routing.fusion_semantics import (
    FusionSemanticResult,
    ParticipantRole,
    LegalFamily,
    SemanticParticipant,
    PartialAssetTransferWording,
)
from src.routing.fusion_subtype import (
    BeneficiaryCreation,
    TransferScope,
    TransferorFate,
)
import src.routing.fusion_reconciliation as reconciliation_module


MAIN = "732829320"
OTHER = "123456782"
THIRD = "356000000"
FOURTH = "552100554"
FIFTH = "542051180"


def party(siren: str | None, name: str) -> dict[str, object]:
    registration: dict[str, str] = {}
    if siren is not None:
        registration["numeroIdentification"] = siren
    return {
        "numeroImmatriculation": registration,
        "denomination": name,
    }


def normalized_announcement(
    *,
    main_siren: str | None = MAIN,
    main_name: str = "BENEFICIAIRE",
    previous_owners: tuple[tuple[str | None, str], ...] = (
        (OTHER, "CEDANT"),
    ),
    description: str | None = "Projet commun de fusion",
    publication_date: str | None = "2023-06-01",
):
    payload: dict[str, object] = {
        "registre": "RCS-A",
        "listepersonnes": {
            "personne": [party(main_siren, main_name)],
        },
        "listeprecedentproprietaire": {
            "personne": [
                party(owner_siren, owner_name)
                for owner_siren, owner_name in previous_owners
            ]
        },
    }
    if description is not None:
        payload["acte"] = {"descriptif": description}
    if publication_date is not None:
        payload["dateparution"] = publication_date
    return normalize_bodacc_announcement(payload)


def semantic_result(
    legal_family: LegalFamily = LegalFamily.FUSION,
    *,
    transfer_scope: TransferScope = TransferScope.UNKNOWN,
    transferor_fate: TransferorFate = TransferorFate.UNKNOWN,
    beneficiary_creation: BeneficiaryCreation = (
        BeneficiaryCreation.MIXED_OR_UNKNOWN
    ),
    participants: tuple[SemanticParticipant, ...] = (),
    partial_asset_transfer_wording: PartialAssetTransferWording = (
        PartialAssetTransferWording.UNKNOWN
    ),
) -> FusionSemanticResult:
    return FusionSemanticResult(
        legal_family=legal_family,
        transfer_scope=transfer_scope,
        transferor_fate=transferor_fate,
        beneficiary_creation=beneficiary_creation,
        participants=participants,
        partial_asset_transfer_wording=partial_asset_transfer_wording,
        evidence=("indice local",),
        reason="raison locale",
    )


def provisional(
    reference: str,
    *,
    legal_family: LegalFamily = LegalFamily.FUSION,
    main_siren: str | None = MAIN,
    previous_siren: str | None = OTHER,
    publication_date: str | None = "2023-06-01",
    description: str | None = "Projet commun de fusion",
    semantic: FusionSemanticResult | None = None,
) -> FusionProvisionalRecord:
    local_semantic = semantic
    if local_semantic is None:
        participants = (
            (
                SemanticParticipant(
                    siren=main_siren,
                    name="BENEFICIAIRE",
                    role=ParticipantRole.BENEFICIARY,
                ),
            )
            if main_siren is not None
            else ()
        )
        local_semantic = semantic_result(legal_family, participants=participants)
    return build_fusion_provisional(
        reference,
        normalized_announcement(
            main_siren=main_siren,
            previous_owners=((previous_siren, "CEDANT"),),
            publication_date=publication_date,
            description=description,
        ),
        local_semantic,
    )


class DescriptionGroupingTest(unittest.TestCase):
    def test_versions_are_explicit(self):
        self.assertEqual(
            FUSION_RECONCILIATION_VERSION,
            "fusion-reconciliation-v1",
        )
        self.assertEqual(
            DESCRIPTION_FINGERPRINT_VERSION,
            "unicode-nfkc-whitespace-sha256-v1",
        )

    def test_unicode_and_whitespace_are_canonical_but_case_is_exact(self):
        decomposed = "  Projet\u00a0de\nSoci\u0065\u0301te\u0301  "
        composed = "Projet de Société"
        self.assertEqual(canonicalize_description(decomposed), composed)
        self.assertEqual(
            description_fingerprint(decomposed),
            sha256(composed.encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            description_fingerprint(composed),
            description_fingerprint(composed.lower()),
        )

    def test_empty_description_has_no_synthetic_group(self):
        self.assertIsNone(canonicalize_description(None))
        self.assertIsNone(canonicalize_description(" \n\t "))
        self.assertIsNone(description_fingerprint(None))
        self.assertIsNone(description_fingerprint("   "))
        with self.assertRaises(TypeError):
            canonicalize_description(123)  # type: ignore[arg-type]

    def test_record_exposes_exact_description_and_entity_group_keys(self):
        record = provisional("A20230100001")
        fingerprint = description_fingerprint("Projet commun de fusion")
        self.assertEqual(record.canonical_description, "Projet commun de fusion")
        self.assertEqual(record.description_fingerprint, fingerprint)
        self.assertTrue(record.description_present)
        self.assertEqual(
            record.description_group_key,
            f"campaign=2023|description=SHA256:{fingerprint}",
        )
        self.assertEqual(
            record.beneficiary_group_keys,
            ("campaign=2023|beneficiary=SIREN:732829320",),
        )
        self.assertEqual(record.transferor_group_keys, ())
        self.assertEqual(len(record.grouping_keys), 2)


class ProvisionalBuilderTest(unittest.TestCase):
    def test_public_types_separate_internal_and_final_codes(self):
        self.assertEqual(
            [value.value for value in ProvisionalType],
            ["AB", "FZ", "SP", "SZ", "AP", "UNKNOWN"],
        )
        self.assertEqual(
            [value.value for value in FinalFusionType],
            ["FU", "AB", "SP", "ST", "AP", "UNKNOWN"],
        )
        self.assertNotIn("FZ", [value.value for value in FinalFusionType])
        self.assertNotIn("SZ", [value.value for value in FinalFusionType])

    def test_fusion_without_previous_owner_self_relation_is_fz(self):
        record = provisional("A20230100001")
        self.assertEqual(record.legal_family, LegalFamily.FUSION)
        self.assertEqual(record.provisional_type, ProvisionalType.FZ)
        self.assertEqual(record.provisional_rule, "local_fusion_provisional")
        self.assertEqual(record.main_siren, MAIN)
        self.assertEqual(record.previous_owner_sirens, (OTHER,))
        self.assertEqual(record.beneficiary_sirens, (MAIN,))
        self.assertEqual(record.transferor_sirens, ())
        self.assertFalse(record.self_relation)

    def test_main_party_is_not_inferred_as_beneficiary_without_role_evidence(self):
        record = provisional(
            "A20230100028",
            semantic=semantic_result(LegalFamily.FUSION),
        )
        self.assertEqual(record.main_siren, MAIN)
        self.assertEqual(record.beneficiary_sirens, ())
        self.assertEqual(record.beneficiary_link_keys, ())
        self.assertIn("beneficiary_linkage_missing", record.diagnostics)

    def test_fusion_previous_owner_equal_main_is_ab_anchor(self):
        record = provisional(
            "A20230100002",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        self.assertEqual(record.provisional_type, ProvisionalType.AB)
        self.assertEqual(
            record.provisional_rule,
            "local_previous_owner_self_anchor",
        )
        self.assertTrue(record.self_relation)
        self.assertEqual(record.self_relation_sirens, (MAIN,))

    def test_scission_uses_sz_and_same_source_self_relation_sp_anchor(self):
        unresolved = provisional(
            "A20230100003",
            legal_family=LegalFamily.SCISSION,
        )
        anchor = provisional(
            "A20230100004",
            legal_family=LegalFamily.SCISSION,
            main_siren=OTHER,
            previous_siren=OTHER,
        )
        self.assertEqual(unresolved.provisional_type, ProvisionalType.SZ)
        self.assertEqual(anchor.provisional_type, ProvisionalType.SP)
        self.assertTrue(anchor.self_relation)

    def test_ap_requires_every_supported_local_legal_fact(self):
        supported = semantic_result(
            LegalFamily.UNKNOWN,
            partial_asset_transfer_wording=PartialAssetTransferWording.YES,
            transfer_scope=TransferScope.PARTIAL,
            transferor_fate=TransferorFate.SURVIVES,
            beneficiary_creation=BeneficiaryCreation.EXISTING,
        )
        record = provisional("A20230100005", semantic=supported)
        self.assertEqual(record.provisional_type, ProvisionalType.AP)
        self.assertEqual(record.provisional_rule, "local_supported_ap_profile")

        incomplete_results = (
            semantic_result(
                LegalFamily.UNKNOWN,
                partial_asset_transfer_wording=PartialAssetTransferWording.YES,
                transfer_scope=TransferScope.UNKNOWN,
                transferor_fate=TransferorFate.SURVIVES,
                beneficiary_creation=BeneficiaryCreation.EXISTING,
            ),
            semantic_result(
                LegalFamily.UNKNOWN,
                partial_asset_transfer_wording=PartialAssetTransferWording.YES,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.UNKNOWN,
                beneficiary_creation=BeneficiaryCreation.EXISTING,
            ),
            semantic_result(
                LegalFamily.UNKNOWN,
                partial_asset_transfer_wording=PartialAssetTransferWording.YES,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.SURVIVES,
                beneficiary_creation=BeneficiaryCreation.MIXED_OR_UNKNOWN,
            ),
        )
        for index, incomplete in enumerate(incomplete_results):
            with self.subTest(index=index):
                unresolved = provisional(
                    f"A202301001{index:02d}", semantic=incomplete
                )
                self.assertEqual(
                    unresolved.provisional_type,
                    ProvisionalType.UNKNOWN,
                )
                self.assertIn(
                    "partial_asset_transfer_missing_complete_ap_profile",
                    unresolved.diagnostics,
                )

    def test_semantic_unknown_is_preserved(self):
        record = provisional(
            "A20230100006",
            semantic=semantic_result(
                LegalFamily.UNKNOWN,
                transfer_scope=TransferScope.UNKNOWN,
                transferor_fate=TransferorFate.UNKNOWN,
            ),
        )
        self.assertEqual(record.provisional_type, ProvisionalType.UNKNOWN)
        self.assertEqual(record.provisional_rule, "local_semantic_unknown")

    def test_participants_are_revalidated_deduplicated_and_role_scoped(self):
        semantic = semantic_result(
            participants=(
                SemanticParticipant(
                    siren=THIRD,
                    name="  CÉDANT   EXPLICITE ",
                    role=ParticipantRole.TRANSFEROR,
                ),
                SemanticParticipant(
                    siren=FOURTH,
                    name="BÉNÉFICIAIRE EXPLICITE",
                    role=ParticipantRole.BENEFICIARY,
                ),
                SemanticParticipant(
                    siren=FIFTH,
                    name="RÔLE INCERTAIN",
                    role=ParticipantRole.BOTH_OR_UNCLEAR,
                ),
                SemanticParticipant(
                    siren=THIRD,
                    name="CÉDANT EXPLICITE",
                    role=ParticipantRole.TRANSFEROR,
                ),
                SemanticParticipant(
                    siren="123456789",
                    name="SIREN INVALIDE",
                    role=ParticipantRole.BENEFICIARY,
                ),
            )
        )
        record = provisional("A20230100007", semantic=semantic)
        self.assertEqual(len(record.participants), 4)
        self.assertEqual(record.participants[0].name, "CÉDANT EXPLICITE")
        self.assertIsNone(record.participants[-1].siren)
        self.assertEqual(record.transferor_sirens, (THIRD, FIFTH))
        self.assertEqual(record.beneficiary_sirens, (FOURTH,))
        self.assertEqual(record.ambiguous_participant_sirens, (FIFTH,))
        self.assertNotIn(f"SIREN:{FIFTH}", record.beneficiary_link_keys)
        self.assertIn(f"SIREN:{FIFTH}", record.transferor_link_keys)
        self.assertIn("participant_3_duplicate_removed", record.diagnostics)
        self.assertIn("participant_4_invalid_siren_removed", record.diagnostics)

    def test_structured_sirens_are_validated_before_linkage(self):
        record = provisional(
            "A20230100008",
            main_siren="123456789",
            previous_siren=OTHER,
        )
        self.assertIsNone(record.main_siren)
        self.assertEqual(record.beneficiary_link_keys, ())
        self.assertEqual(record.provisional_type, ProvisionalType.FZ)
        self.assertIn("invalid_main_siren_removed", record.diagnostics)
        self.assertIn("beneficiary_linkage_missing", record.diagnostics)

    def test_publication_year_precedes_reference_and_mismatch_is_visible(self):
        record = provisional(
            "A20240100001",
            publication_date="15/06/2023",
        )
        self.assertEqual(record.publication_year, 2023)
        self.assertEqual(record.campaign_year, 2023)
        self.assertIn(
            "publication_reference_year_mismatch",
            record.diagnostics,
        )

        fallback = provisional(
            "A20240100002",
            publication_date=None,
        )
        self.assertIsNone(fallback.publication_year)
        self.assertEqual(fallback.campaign_year, 2024)
        self.assertIn(
            "campaign_year_derived_from_announcement_reference",
            fallback.diagnostics,
        )

        missing = provisional("UNPARSEABLE", publication_date=None)
        self.assertIsNone(missing.campaign_year)
        self.assertEqual(missing.grouping_keys, ())
        self.assertIn("campaign_year_missing", missing.diagnostics)

    def test_builder_contract_rejects_wrong_boundaries(self):
        normalized = normalized_announcement()
        semantic = semantic_result()
        for reference in ("", "   ", None):
            with self.subTest(reference=reference):
                with self.assertRaises(ValueError):
                    build_fusion_provisional(  # type: ignore[arg-type]
                        reference, normalized, semantic
                    )
        with self.assertRaises(TypeError):
            build_fusion_provisional(
                "A20230100009", {}, semantic  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            build_fusion_provisional(
                "A20230100009", normalized, {}  # type: ignore[arg-type]
            )

    def test_provisional_record_is_frozen(self):
        record = provisional("A20230100010")
        with self.assertRaises(FrozenInstanceError):
            record.provisional_type = ProvisionalType.AB  # type: ignore[misc]


class HistoricalRealShapedPatternsTest(unittest.TestCase):
    @staticmethod
    def _fusion_semantic(
        *, beneficiary: str, transferor: str
    ) -> FusionSemanticResult:
        return semantic_result(
            LegalFamily.FUSION,
            participants=(
                SemanticParticipant(
                    siren=transferor,
                    name="SOCIÉTÉ ABSORBÉE",
                    role=ParticipantRole.TRANSFEROR,
                ),
                SemanticParticipant(
                    siren=beneficiary,
                    name="SOCIÉTÉ ABSORBANTE",
                    role=ParticipantRole.BENEFICIARY,
                ),
            ),
        )

    @staticmethod
    def _scission_semantic(
        *,
        transferor: str,
        scope: TransferScope,
        wording: PartialAssetTransferWording = (
            PartialAssetTransferWording.UNKNOWN
        ),
    ) -> FusionSemanticResult:
        return semantic_result(
            LegalFamily.SCISSION,
            transfer_scope=scope,
            partial_asset_transfer_wording=wording,
            participants=(
                SemanticParticipant(
                    siren=transferor,
                    name="SOCIÉTÉ SCINDÉE",
                    role=ParticipantRole.TRANSFEROR,
                ),
            ),
        )

    def test_absorption_two_announcements_shares_detected_beneficiary_key(self):
        description = (
            "Projet de fusion par voie d'absorption. Société absorbante : "
            "BÉNÉFICIAIRE 732 829 320. Société absorbée : A 123 456 782."
        )
        semantic = self._fusion_semantic(
            beneficiary=MAIN,
            transferor=OTHER,
        )
        announcement_a = provisional(
            "A20230100301",
            main_siren=OTHER,
            previous_siren=MAIN,
            description=description,
            semantic=semantic,
        )
        announcement_b = provisional(
            "A20230100302",
            main_siren=MAIN,
            previous_siren=MAIN,
            description=description,
            semantic=semantic,
        )

        self.assertEqual(announcement_a.provisional_type, ProvisionalType.FZ)
        self.assertEqual(announcement_b.provisional_type, ProvisionalType.AB)
        self.assertEqual(
            announcement_a.beneficiary_link_keys,
            ("SIREN:732829320",),
        )
        self.assertEqual(
            announcement_a.beneficiary_link_keys,
            announcement_b.beneficiary_link_keys,
        )
        self.assertEqual(
            announcement_a.description_fingerprint,
            announcement_b.description_fingerprint,
        )

        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family(
                (announcement_a, announcement_b)
            )
        }
        reconciled_a = by_ref[announcement_a.ref_annonce_complet]
        self.assertEqual(reconciled_a.final_type, FinalFusionType.AB)
        self.assertEqual(
            reconciled_a.reconciliation_rule,
            "fz_same_beneficiary_as_ab_anchor",
        )
        self.assertEqual(
            reconciled_a.anchor_refs,
            (announcement_b.ref_annonce_complet,),
        )

    def test_ap_profile_survives_regime_des_scissions_wording(self):
        row = provisional(
            "A20230100308",
            description=(
                "Apport partiel d’actif placé sous le régime des scissions."
            ),
            semantic=semantic_result(
                LegalFamily.SCISSION,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.SURVIVES,
                beneficiary_creation=BeneficiaryCreation.EXISTING,
                partial_asset_transfer_wording=(
                    PartialAssetTransferWording.YES
                ),
            ),
        )
        self.assertEqual(row.provisional_type, ProvisionalType.AP)
        self.assertEqual(row.provisional_rule, "local_supported_ap_profile")
        self.assertEqual(
            reconcile_fusion_family((row,))[0].final_type,
            FinalFusionType.AP,
        )

    def test_partial_surviving_new_beneficiary_supports_sp(self):
        row = provisional(
            "A20230100309",
            main_siren=MAIN,
            previous_siren=OTHER,
            description="Scission partielle au profit d’une société nouvelle.",
            semantic=semantic_result(
                LegalFamily.SCISSION,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.SURVIVES,
                beneficiary_creation=BeneficiaryCreation.NEW,
                partial_asset_transfer_wording=(
                    PartialAssetTransferWording.YES
                ),
            ),
        )
        self.assertEqual(row.provisional_type, ProvisionalType.SP)
        self.assertEqual(row.provisional_rule, "local_supported_sp_profile")
        self.assertEqual(
            reconcile_fusion_family((row,))[0].final_type,
            FinalFusionType.SP,
        )

    def test_true_fusion_without_self_previous_owner_resolves_to_fu(self):
        row = provisional(
            "A20230100303",
            main_siren=MAIN,
            previous_siren=OTHER,
            description=(
                "Fusion entre A 123 456 782 et bénéficiaire 732 829 320."
            ),
            semantic=self._fusion_semantic(
                beneficiary=MAIN,
                transferor=OTHER,
            ),
        )
        self.assertEqual(row.provisional_type, ProvisionalType.FZ)
        self.assertEqual(
            reconcile_fusion_family((row,))[0].final_type,
            FinalFusionType.FU,
        )

    def test_partial_scission_anchor_propagates_by_same_transferor(self):
        description = (
            "Projet de scission partielle de la société A 123 456 782."
        )
        semantic = self._scission_semantic(
            transferor=OTHER,
            scope=TransferScope.PARTIAL,
        )
        announcement_a = provisional(
            "A20230100304",
            main_siren=OTHER,
            previous_siren=OTHER,
            description=description,
            semantic=semantic,
        )
        announcement_b = provisional(
            "A20230100305",
            main_siren=MAIN,
            previous_siren=OTHER,
            description=description,
            semantic=semantic,
        )
        self.assertEqual(announcement_a.provisional_type, ProvisionalType.SP)
        self.assertEqual(announcement_b.provisional_type, ProvisionalType.SZ)
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family(
                (announcement_b, announcement_a)
            )
        }
        self.assertEqual(
            by_ref[announcement_b.ref_annonce_complet].final_type,
            FinalFusionType.SP,
        )

    def test_total_scission_without_anchor_resolves_to_st(self):
        row = provisional(
            "A20230100306",
            legal_family=LegalFamily.SCISSION,
            main_siren=MAIN,
            previous_siren=OTHER,
            description="Scission totale de la société A 123 456 782.",
            semantic=self._scission_semantic(
                transferor=OTHER,
                scope=TransferScope.TOTAL,
            ),
        )
        self.assertEqual(row.provisional_type, ProvisionalType.SZ)
        self.assertEqual(
            reconcile_fusion_family((row,))[0].final_type,
            FinalFusionType.ST,
        )

    def test_sp_with_partial_asset_transfer_wording_is_not_local_ap(self):
        row = provisional(
            "A20230100307",
            main_siren=OTHER,
            previous_siren=OTHER,
            description=(
                "Scission partielle réalisée par apport partiel d'actif de "
                "la société A 123 456 782."
            ),
            semantic=self._scission_semantic(
                transferor=OTHER,
                scope=TransferScope.PARTIAL,
                wording=PartialAssetTransferWording.YES,
            ),
        )
        self.assertEqual(row.legal_family, LegalFamily.SCISSION)
        self.assertEqual(
            row.partial_asset_transfer_wording,
            PartialAssetTransferWording.YES,
        )
        self.assertEqual(row.provisional_type, ProvisionalType.SP)
        self.assertEqual(
            reconcile_fusion_family((row,))[0].final_type,
            FinalFusionType.SP,
        )


class GlobalReconciliationTest(unittest.TestCase):
    def test_ab_anchor_reclassifies_same_campaign_same_beneficiary_fz(self):
        anchor = provisional(
            "A20230100001",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        related = provisional(
            "A20230100002",
            main_siren=MAIN,
            previous_siren=OTHER,
        )
        result = reconcile_fusion_family((related, anchor))
        by_ref = {row.ref_annonce_complet: row for row in result}

        self.assertEqual(by_ref[anchor.ref_annonce_complet].final_type, FinalFusionType.AB)
        reconciled = by_ref[related.ref_annonce_complet]
        self.assertEqual(reconciled.final_type, FinalFusionType.AB)
        self.assertEqual(
            reconciled.reconciliation_rule,
            "fz_same_beneficiary_as_ab_anchor",
        )
        self.assertEqual(reconciled.anchor_refs, (anchor.ref_annonce_complet,))
        self.assertEqual(
            reconciled.reconciliation_group_key,
            "campaign=2023|beneficiary=SIREN:732829320",
        )
        self.assertTrue(reconciled.changed)

    def test_fz_without_anchor_becomes_fu_and_never_ambiguous_ap(self):
        row = provisional("A20230100003")
        reconciled = reconcile_fusion_family((row,))[0]
        self.assertEqual(reconciled.final_type, FinalFusionType.FU)
        self.assertEqual(reconciled.reconciliation_rule, "fz_remaining_to_fu")
        self.assertIn(
            "isolated_ap_fallback_intentionally_not_applied",
            reconciled.diagnostics,
        )
        self.assertFalse(HISTORICAL_ISOLATED_AP_FALLBACK_IMPLEMENTED)
        self.assertIn("not applied", HISTORICAL_ISOLATED_AP_FALLBACK_NOTE)

    def test_ab_anchor_does_not_cross_campaigns(self):
        anchor = provisional(
            "A20230100004",
            main_siren=MAIN,
            previous_siren=MAIN,
            publication_date="2023-12-31",
        )
        next_campaign = provisional(
            "A20240100005",
            main_siren=MAIN,
            previous_siren=OTHER,
            publication_date="2024-01-01",
        )
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((anchor, next_campaign))
        }
        self.assertEqual(
            by_ref[next_campaign.ref_annonce_complet].final_type,
            FinalFusionType.FU,
        )
        self.assertEqual(
            by_ref[next_campaign.ref_annonce_complet].anchor_refs,
            (),
        )

    def test_missing_campaign_cannot_propagate_an_anchor(self):
        anchor = provisional(
            "ANCHOR",
            main_siren=MAIN,
            previous_siren=MAIN,
            publication_date=None,
        )
        related = provisional(
            "RELATED",
            main_siren=MAIN,
            previous_siren=OTHER,
            publication_date=None,
        )
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((anchor, related))
        }
        self.assertEqual(
            by_ref[related.ref_annonce_complet].final_type,
            FinalFusionType.FU,
        )

    def test_same_description_alone_is_inspectable_but_not_an_anchor_rule(self):
        anchor = provisional(
            "A20230100006",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        unrelated = provisional(
            "A20230100007",
            main_siren=THIRD,
            previous_siren=OTHER,
        )
        self.assertEqual(
            anchor.description_group_key,
            unrelated.description_group_key,
        )
        result = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((anchor, unrelated))
        }
        self.assertEqual(
            result[unrelated.ref_annonce_complet].final_type,
            FinalFusionType.FU,
        )

    def test_semantic_beneficiary_siren_can_supply_exact_linkage(self):
        anchor = provisional(
            "A20230100008",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        semantic = semantic_result(
            participants=(
                SemanticParticipant(
                    siren=MAIN,
                    name="BÉNÉFICIAIRE LIÉ",
                    role=ParticipantRole.BENEFICIARY,
                ),
            )
        )
        related = provisional(
            "A20230100009",
            main_siren=THIRD,
            previous_siren=OTHER,
            semantic=semantic,
        )
        result = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((related, anchor))
        }
        self.assertEqual(
            result[related.ref_annonce_complet].final_type,
            FinalFusionType.AB,
        )

    def test_sp_anchor_reclassifies_same_campaign_same_transferor_sz(self):
        anchor = provisional(
            "A20230100010",
            legal_family=LegalFamily.SCISSION,
            main_siren=OTHER,
            previous_siren=OTHER,
        )
        related = provisional(
            "A20230100011",
            main_siren=MAIN,
            previous_siren=OTHER,
            semantic=semantic_result(
                LegalFamily.SCISSION,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.SURVIVES,
                participants=(
                    SemanticParticipant(
                        siren=OTHER,
                        name="SCINDEE",
                        role=ParticipantRole.TRANSFEROR,
                    ),
                ),
            ),
        )
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((related, anchor))
        }
        reconciled = by_ref[related.ref_annonce_complet]
        self.assertEqual(reconciled.final_type, FinalFusionType.SP)
        self.assertEqual(
            reconciled.reconciliation_rule,
            "sz_same_transferor_as_sp_anchor",
        )
        self.assertEqual(reconciled.anchor_refs, (anchor.ref_annonce_complet,))
        self.assertEqual(
            reconciled.reconciliation_group_key,
            "campaign=2023|transferor=SIREN:123456782",
        )
        self.assertNotIn("local_global_sp_conflict", reconciled.diagnostics)

    def test_total_disappearing_sz_is_not_silently_overwritten_by_sp_anchor(self):
        anchor = provisional(
            "A20230100029",
            legal_family=LegalFamily.SCISSION,
            main_siren=OTHER,
            previous_siren=OTHER,
        )
        related = provisional(
            "A20230100030",
            main_siren=MAIN,
            previous_siren=OTHER,
            semantic=semantic_result(
                LegalFamily.SCISSION,
                transfer_scope=TransferScope.TOTAL,
                transferor_fate=TransferorFate.DISAPPEARS,
                participants=(
                    SemanticParticipant(
                        siren=OTHER,
                        name="SCINDEE",
                        role=ParticipantRole.TRANSFEROR,
                    ),
                ),
            ),
        )
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((related, anchor))
        }
        reconciled = by_ref[related.ref_annonce_complet]
        self.assertEqual(reconciled.final_type, FinalFusionType.ST)
        self.assertEqual(
            reconciled.reconciliation_rule,
            "sz_local_semantics_override_sp_anchor_to_st",
        )
        self.assertEqual(reconciled.anchor_refs, (anchor.ref_annonce_complet,))
        self.assertIn("local_global_sp_conflict", reconciled.diagnostics)
        self.assertIn(
            "sp_anchor_conflicts_with_transfer_scope_total",
            reconciled.diagnostics,
        )
        self.assertIn(
            "sp_anchor_conflicts_with_transferor_fate_disappears",
            reconciled.diagnostics,
        )

    def test_sz_without_anchor_becomes_st_not_ap(self):
        row = provisional(
            "A20230100012",
            legal_family=LegalFamily.SCISSION,
            main_siren=MAIN,
            previous_siren=OTHER,
        )
        reconciled = reconcile_fusion_family((row,))[0]
        self.assertEqual(reconciled.final_type, FinalFusionType.ST)
        self.assertEqual(reconciled.reconciliation_rule, "sz_remaining_to_st")
        self.assertIn(
            "isolated_ap_fallback_intentionally_not_applied",
            reconciled.diagnostics,
        )

    def test_local_ap_and_unknown_remain_unchanged(self):
        ap = provisional(
            "A20230100013",
            semantic=semantic_result(
                LegalFamily.UNKNOWN,
                partial_asset_transfer_wording=PartialAssetTransferWording.YES,
                transfer_scope=TransferScope.PARTIAL,
                transferor_fate=TransferorFate.SURVIVES,
                beneficiary_creation=BeneficiaryCreation.EXISTING,
            ),
        )
        unknown = provisional(
            "A20230100014",
            semantic=semantic_result(LegalFamily.UNKNOWN),
        )
        by_ref = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family((unknown, ap))
        }
        self.assertEqual(by_ref[ap.ref_annonce_complet].final_type, FinalFusionType.AP)
        self.assertFalse(by_ref[ap.ref_annonce_complet].changed)
        self.assertEqual(
            by_ref[unknown.ref_annonce_complet].final_type,
            FinalFusionType.UNKNOWN,
        )
        self.assertFalse(by_ref[unknown.ref_annonce_complet].changed)

    def test_self_relation_rows_are_never_removed(self):
        ab = provisional(
            "A20230100015",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        sp = provisional(
            "A20230100016",
            legal_family=LegalFamily.SCISSION,
            main_siren=OTHER,
            previous_siren=OTHER,
        )
        result = reconcile_fusion_family((sp, ab))
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row.self_relation for row in result))
        self.assertEqual(
            {row.ref_annonce_complet for row in result},
            {ab.ref_annonce_complet, sp.ref_annonce_complet},
        )

    def test_multiple_anchor_matches_are_sorted_and_inspectable(self):
        first_anchor = provisional(
            "A20230100020",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        second_anchor = provisional(
            "A20230100018",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        related = provisional(
            "A20230100019",
            main_siren=MAIN,
            previous_siren=OTHER,
        )
        result = {
            row.ref_annonce_complet: row
            for row in reconcile_fusion_family(
                (first_anchor, related, second_anchor)
            )
        }
        self.assertEqual(
            result[related.ref_annonce_complet].anchor_refs,
            (
                second_anchor.ref_annonce_complet,
                first_anchor.ref_annonce_complet,
            ),
        )

    def test_reconciliation_is_order_independent_and_sorted(self):
        rows = (
            provisional(
                "A20230100023",
                legal_family=LegalFamily.SCISSION,
                main_siren=MAIN,
                previous_siren=OTHER,
            ),
            provisional(
                "A20230100021",
                main_siren=MAIN,
                previous_siren=MAIN,
            ),
            provisional(
                "A20230100022",
                main_siren=MAIN,
                previous_siren=THIRD,
            ),
        )
        first = reconcile_fusion_family(rows)
        second = reconcile_fusion_family(tuple(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(
            [row.ref_annonce_complet for row in first],
            sorted(row.ref_annonce_complet for row in rows),
        )
        self.assertEqual(
            [row.ref_annonce_complet for row in rows],
            ["A20230100023", "A20230100021", "A20230100022"],
        )

    def test_empty_input_and_invalid_inputs_are_explicit(self):
        self.assertEqual(reconcile_fusion_family(()), ())
        with self.assertRaises(TypeError):
            reconcile_fusion_family(iter(()))  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reconcile_fusion_family((object(),))  # type: ignore[arg-type]
        duplicate = provisional("A20230100024")
        with self.assertRaisesRegex(ValueError, "duplicate provisional"):
            reconcile_fusion_family((duplicate, duplicate))

    def test_reconciled_record_is_frozen(self):
        record = reconcile_fusion_family(
            (provisional("A20230100025"),)
        )[0]
        with self.assertRaises(FrozenInstanceError):
            record.final_type = FinalFusionType.AP  # type: ignore[misc]


class PurityAndLeakageBoundaryTest(unittest.TestCase):
    def test_public_function_signatures_accept_only_source_semantics(self):
        self.assertEqual(
            list(inspect.signature(build_fusion_provisional).parameters),
            ["ref_annonce_complet", "normalized", "semantic"],
        )
        self.assertEqual(
            list(inspect.signature(reconcile_fusion_family).parameters),
            ["provisional_rows"],
        )

    def test_module_has_no_llm_network_or_annotation_target_dependency(self):
        source = inspect.getsource(reconciliation_module)
        for forbidden in (
            "src.llm",
            "requests",
            "boto3",
            "date_" + "creation_op",
            "siren_" + "cedante",
            "siren_" + "beneficiaire",
            "type_" + "op",
            "reference_" + "type",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_reconciliation_does_not_mutate_input_records(self):
        anchor = provisional(
            "A20230100026",
            main_siren=MAIN,
            previous_siren=MAIN,
        )
        related = provisional(
            "A20230100027",
            main_siren=MAIN,
            previous_siren=OTHER,
        )
        before = (anchor, related)
        reconcile_fusion_family(before)
        self.assertEqual(before, (anchor, related))
        self.assertEqual(related.provisional_type, ProvisionalType.FZ)


if __name__ == "__main__":
    unittest.main()
