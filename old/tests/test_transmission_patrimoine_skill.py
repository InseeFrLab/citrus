import copy
import inspect
import unittest
from unittest.mock import patch

import polars as pl

from src.bodacc import normalize_bodacc_announcement
from src.modele.benchmark import compare_predictions
from src.operation import OperationSkill, transmission_patrimoine_skill
import src.operation.transmission_patrimoine as tp_module
from tests.test_bodacc_normalization import (
    REAL_B202302491051_DESCRIPTION,
    real_b202302491051_payload,
)
from src.operation.transmission_patrimoine import (
    contains_tp_wording,
    extract_description_dates,
)


HISTORICAL_DESCRIPTION = (
    'Décision de l associé unique en date du 10 octobre 2017 décidant de la '
    'dissolution et de la transmission universelle du patrimoine de la société '
    '455 504 597 au profit de RCS Dunkerque 335 275 053, publication du '
    '10 novembre 2017.'
)


def person(siren='455 504 597', name='SOCIETE DISSOUE'):
    return {
        'numeroImmatriculation': {'numeroIdentificationRCS': siren},
        'denomination': name,
    }


def rcs_b_announcement(description=HISTORICAL_DESCRIPTION, **overrides):
    payload = {
        'registre': 'RCS-B',
        'listepersonnes': {'personne': person()},
        'modificationsGenerales': {'descriptif': description},
        'dateparution': '2017-11-15',
        'url_complete': 'https://www.bodacc.fr/annonce/TP1',
    }
    payload.update(overrides)
    return payload


class TransmissionPatrimoineSkillTest(unittest.TestCase):
    def test_real_opendata_lowercase_modification_reaches_public_skill(self):
        result = transmission_patrimoine_skill.extract(
            real_b202302491051_payload(
                {"descriptif": REAL_B202302491051_DESCRIPTION}
            )
        )

        self.assertEqual(result['sirenCedant'], '810379180')
        self.assertEqual(result['sirenBeneficiaire'], '824640916')
        self.assertEqual(result['dateEffetComptable'], '2023-11-06')
        self.assertEqual(result['dateRealisationJuridique'], '2023-11-10')

    def test_public_contract_canonical_roles_dates_and_remaining_fields(self):
        result = transmission_patrimoine_skill.extract(rcs_b_announcement())

        self.assertIsInstance(transmission_patrimoine_skill, OperationSkill)
        self.assertEqual(transmission_patrimoine_skill.operation_type, 'TP')
        self.assertEqual(result['typeOperation'], 'TP')
        self.assertEqual(result['sirenCedant'], '455504597')
        self.assertEqual(result['raisonSocialeCedant'], 'SOCIETE DISSOUE')
        self.assertEqual(result['sirenBeneficiaire'], '335275053')
        self.assertIsNone(result['raisonSocialeBeneficiaire'])
        self.assertEqual(result['dateEffetComptable'], '2017-10-10')
        self.assertEqual(result['dateRealisationJuridique'], '2017-11-10')
        self.assertEqual(result['anneeCampagne'], 2017)
        self.assertIsNone(result['montantNet'])
        self.assertEqual(
            result['source'], 'https://www.bodacc.fr/annonce/TP1'
        )

    def test_second_documented_historical_example(self):
        description = (
            'Décision du 4 septembre 2017 de transmission universelle de '
            'patrimoine au profit de la société 450 150 685. Publication '
            'du 29 septembre 2017.'
        )
        payload = rcs_b_announcement(
            description,
            listepersonnes={
                'personne': person('489 065 573', 'SECOND CEDANT')
            },
        )

        result = transmission_patrimoine_skill.extract(payload)

        self.assertEqual(result['sirenCedant'], '489065573')
        self.assertEqual(result['sirenBeneficiaire'], '450150685')
        self.assertEqual(result['dateEffetComptable'], '2017-09-04')
        self.assertEqual(result['dateRealisationJuridique'], '2017-09-29')

    def test_main_siren_is_excluded_and_first_remaining_candidate_wins(self):
        description = (
            'Transmission universelle de patrimoine 455504597, '
            'bénéficiaire 335.275.053, autre société 450 150 685.'
        )

        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement(description)
        )

        self.assertEqual(result['sirenBeneficiaire'], '335275053')

    def test_supported_candidate_formats_use_existing_ordered_helper(self):
        for candidate in ('335275053', '335 275 053', '335.275.053'):
            with self.subTest(candidate=candidate):
                result = transmission_patrimoine_skill.extract(
                    rcs_b_announcement(
                        f'Transmission universelle du patrimoine RCS {candidate}'
                    )
                )
                self.assertEqual(result['sirenBeneficiaire'], '335275053')

    def test_invalid_luhn_and_monetary_candidates_are_not_beneficiaries(self):
        description = (
            'Transmission universelle de patrimoine, RCS 123 456 789, '
            'valeur 732 829 320 EUR.'
        )

        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement(description)
        )

        self.assertIsNone(result['sirenBeneficiaire'])

    def test_missing_second_siren_remains_nullable(self):
        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement(
                'Transmission universelle du patrimoine de 455 504 597.'
            )
        )

        self.assertIsNone(result['sirenBeneficiaire'])

    def test_tp_wording_variants_are_case_and_accent_tolerant(self):
        variants = (
            'transmission universelle du patrimoine',
            'TRANSMISSION UNIVERSELLE DE PATRIMOINE',
            'transmiss.univers.patrimoine',
            'Transmission-universelle-de-patrimoine',
        )
        for wording in variants:
            with self.subTest(wording=wording):
                self.assertTrue(contains_tp_wording(wording))
        self.assertFalse(contains_tp_wording('cession partielle de patrimoine'))
        self.assertFalse(contains_tp_wording(None))

    def test_french_textual_dates_preserve_left_to_right_order(self):
        description = (
            'Décision du 4 septembre 2017, effet au 10 octobre 2017, '
            'publication du 2 décembre 2017.'
        )

        dates = extract_description_dates(description)
        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement(description)
        )

        self.assertEqual(
            dates, ('2017-09-04', '2017-10-10', '2017-12-02')
        )
        self.assertEqual(result['dateEffetComptable'], '2017-09-04')
        self.assertEqual(result['dateRealisationJuridique'], '2017-12-02')

    def test_accented_and_accentless_french_months_are_supported(self):
        self.assertEqual(
            extract_description_dates(
                '1er fevrier 2020, 2 février 2020, 3 aout 2020, '
                '4 août 2020, 5 decembre 2020, 6 décembre 2020'
            ),
            (
                '2020-02-01',
                '2020-02-02',
                '2020-08-03',
                '2020-08-04',
                '2020-12-05',
                '2020-12-06',
            ),
        )

    def test_numeric_and_iso_dates_keep_text_order_and_reject_malformed(self):
        description = (
            'ISO 2020-12-31, slash 02/01/2020, tiret 03-02-2020, '
            'invalides 31/02/2020 et 2020-13-01.'
        )

        self.assertEqual(
            extract_description_dates(description),
            ('2020-12-31', '2020-01-02', '2020-02-03'),
        )

    def test_no_description_date_uses_publication_only_for_effect(self):
        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement(
                'Transmission universelle du patrimoine sans date.',
                dateparution='29/09/2017',
            )
        )

        self.assertEqual(result['dateEffetComptable'], '2017-09-29')
        self.assertIsNone(result['dateRealisationJuridique'])

    def test_single_description_date_populates_both_date_fields(self):
        result = transmission_patrimoine_skill.extract(
            rcs_b_announcement('Décision de TUP du 4 septembre 2017.')
        )

        self.assertEqual(result['dateEffetComptable'], '2017-09-04')
        self.assertEqual(result['dateRealisationJuridique'], '2017-09-04')

    def test_sparse_and_rcs_a_payloads_do_not_invent_tp_semantics(self):
        payloads = (
            {},
            {
                'registre': 'RCS-A',
                'listepersonnes': {
                    'personne': {
                        'numeroImmatriculation': {
                            'numeroIdentification': '732 829 320'
                        },
                        'denomination': 'ACQUEREUR RCS-A',
                    }
                },
                'acte': {
                    'vente': {
                        'descriptif': '335 275 053 le 10 octobre 2017'
                    }
                },
                'dateparution': '2017-11-15',
            },
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                result = transmission_patrimoine_skill.extract(payload)
                for field in (
                    'sirenCedant',
                    'raisonSocialeCedant',
                    'sirenBeneficiaire',
                    'raisonSocialeBeneficiaire',
                    'dateEffetComptable',
                    'dateRealisationJuridique',
                    'montantNet',
                ):
                    self.assertIsNone(result[field])

    def test_extraction_uses_normalizer_and_does_not_mutate_raw_payload(self):
        payload = rcs_b_announcement()
        before = copy.deepcopy(payload)

        with patch.object(
            tp_module,
            'normalize_bodacc_announcement',
            wraps=normalize_bodacc_announcement,
        ) as normalize_mock:
            transmission_patrimoine_skill.extract(payload)

        normalize_mock.assert_called_once_with(payload)
        self.assertEqual(payload, before)

    def test_tp_module_has_no_llm_network_router_or_raw_traversal_boundary(self):
        source = inspect.getsource(tp_module)

        for forbidden in (
            'src.llm',
            'ask_json',
            'requests',
            'bodacc_api',
            'router',
            'listepersonnes',
            'modificationsGenerales',
        ):
            self.assertNotIn(forbidden, source)

    def test_prediction_is_generic_benchmark_compatible(self):
        prediction = {
            'ref_annonce_complet': 'B20170250TP1',
            **transmission_patrimoine_skill.extract(rcs_b_announcement()),
        }
        reference = {
            'ref_annonce_complet': 'B20170250TP1',
            'type_op': 'TP',
            'siren_cedante': 455504597,
            'siren_beneficiaire': 335275053,
            'date_effet_comptable_op': '2017-10-10',
            'date_realisation_juridique_op': '2017-11-10',
            'montant': None,
        }

        comparison = compare_predictions(
            pl.DataFrame([reference]), pl.DataFrame([prediction])
        )

        self.assertTrue(comparison.rows['exact_row_correct'][0])


if __name__ == '__main__':
    unittest.main()
