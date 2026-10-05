import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from explorer_batch import (
    build_argument_parser as batch_parser,
    compute_metrics,
    evaluate,
    parse_types,
    run_batch,
    select_annonces,
    write_meta,
)
from explorer_grid import (
    DURATIONS_FILE,
    METRICS_TEXT_FILE,
    MODEL_ENV,
    batch_argv,
    build_argument_parser,
    configurations,
    MlflowCredentialsError,
    is_complete,
    main,
    missing_mlflow_credentials,
    mlflow_metrics,
    setup_mlflow,
    run_configuration,
    selection_argv,
)
from tests.test_explorer_batch import annotations_frame, fake_ask, fake_fetch


def fake_batch(output_dir):
    annotations = annotations_frame()
    ids = select_annonces(annotations, parse_types(None), None)
    records = run_batch(
        ids, annotations, workers=2, output_dir=output_dir,
        fetch=fake_fetch, ask_fn=fake_ask, progress=lambda line: None,
    )
    return records


class FakeMlflow:
    def __init__(self):
        self.params, self.metrics, self.tags, self.artifacts = {}, {}, {}, []
        self.status = None

    def start_run(self, **kwargs):
        self.run = kwargs

    def end_run(self, status):
        self.status = status

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics):
        self.metrics.update(metrics)

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_artifacts(self, path):
        self.artifacts.append(path)


class GridTest(unittest.TestCase):
    def test_default_grid_has_every_combination_once(self):
        grid = configurations()
        self.assertEqual(len(grid), 3 * 2 * 3 * 2)
        self.assertEqual(len({config["name"] for config in grid}), len(grid))

    def test_cli_filters_restrict_the_grid(self):
        args = build_argument_parser().parse_args(
            ["--modeles", "gemma4-26b-moe", "--raisonnement", "non"]
        )
        grid = configurations(args.modeles, [False], args.analyses, args.approches)
        self.assertEqual(len(grid), 6)
        self.assertTrue(all(not config["reasoning"] for config in grid))

    def test_batch_argv_is_understood_by_explorer_batch(self):
        args = build_argument_parser().parse_args(["--types", "VE", "LG", "-n", "5", "--seed", "3"])
        with_reasoning, without = (
            config for config in configurations(["qwen3-8-27b"], [True, False], ["courte"], ["juridique"])
        )
        parsed = batch_parser().parse_args(batch_argv(without, selection_argv(args), Path("x")))
        self.assertTrue(parsed.no_reasoning)
        self.assertEqual((parsed.approche, parsed.analyse), ("juridique", "courte"))
        self.assertEqual((parsed.types, parsed.sample_size, parsed.seed), (["VE", "LG"], 5, 3))
        self.assertTrue(parsed.no_browse and parsed.yes)
        self.assertNotIn("--no-reasoning", batch_argv(with_reasoning, [], Path("x")))

    def test_all_and_sample_size_are_exclusive(self):
        for parser in (build_argument_parser(), batch_parser()):
            with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
                parser.parse_args(["--all", "-n", "300"])
            self.assertTrue(parser.parse_args(["--all"]).all)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.records = fake_batch(Path(self.tmp.name) / "batch")
        self.annonces, self.operations = evaluate(self.records)
        self.metrics = compute_metrics(self.annonces, self.operations)

    def tearDown(self):
        self.tmp.cleanup()

    def test_metrics_are_numbers(self):
        values = mlflow_metrics(self.metrics, {"duree_totale_s": 12.0}, self.annonces)
        self.assertTrue(all(isinstance(value, float) for value in values.values()))
        self.assertEqual(values["erreurs"], 1.0)
        self.assertAlmostEqual(values["taux_type_ok"], self.metrics["type_ok"] / self.metrics["ok"])
        self.assertIn("taux_montantNet_exact", values)

    def test_mean_duration_per_annotated_type(self):
        values = mlflow_metrics(self.metrics, {}, self.annonces)
        done = self.annonces.filter(self.annonces["statut"] == "ok")
        for code in set(done["type_annote"].to_list()):
            expected = done.filter(done["type_annote"] == code)["duree_s"].mean()
            self.assertAlmostEqual(values[f"duree_moyenne_s_{code}"], expected)
        # B4 (LG) est en erreur : pas de durée pour ce type
        self.assertNotIn("duree_moyenne_s_LG", values)

    def test_empty_batch_does_not_divide_by_zero(self):
        annonces, operations = evaluate([])
        values = mlflow_metrics(compute_metrics(annonces, operations), {}, annonces)
        self.assertNotIn("taux_type_ok", values)
        self.assertEqual(values["annonces"], 0.0)


class RunConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "config"
        self.config = configurations(["gemma4-26b-moe"], [False], ["aucune"], ["metier"])[0]

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_is_logged_and_model_env_restored(self):
        seen = {}

        def fake_main(argv):
            seen["model"] = os.environ.get(MODEL_ENV)
            seen["argv"] = argv
            fake_batch(self.output)
            write_meta(self.output, {"types": ["VE"], "seed": 0, "duree_totale_s": 1.0})
            return 0

        fake_mlflow = FakeMlflow()
        with mock.patch.dict(os.environ, {MODEL_ENV: "avant"}), \
                mock.patch("explorer_batch.main", fake_main):
            self.assertTrue(run_configuration(fake_mlflow, self.config, [], self.output))
            self.assertEqual(os.environ[MODEL_ENV], "avant")
        self.assertEqual(seen["model"], "gemma4-26b-moe")
        self.assertEqual(fake_mlflow.status, "FINISHED")
        self.assertEqual(fake_mlflow.params["analyse"], "aucune")
        self.assertIn("duree_moyenne_s", fake_mlflow.metrics)
        self.assertTrue((self.output / METRICS_TEXT_FILE).exists())
        self.assertTrue((self.output / DURATIONS_FILE).exists())
        self.assertTrue(is_complete(self.output))

    def test_failing_batch_marks_run_failed(self):
        fake_mlflow = FakeMlflow()
        with mock.patch("explorer_batch.main", side_effect=RuntimeError("boom")):
            self.assertFalse(run_configuration(fake_mlflow, self.config, [], self.output))
        self.assertEqual(fake_mlflow.status, "FAILED")
        self.assertIn("boom", fake_mlflow.tags["erreur"])
        self.assertFalse(is_complete(self.output))


class CredentialsTest(unittest.TestCase):
    def test_missing_credentials_are_listed(self):
        self.assertEqual(missing_mlflow_credentials({}), ["MLFLOW_TRACKING_URI"])
        http = {"MLFLOW_TRACKING_URI": "https://mlflow.example"}
        self.assertEqual(
            missing_mlflow_credentials(http),
            ["MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"],
        )
        self.assertEqual(
            missing_mlflow_credentials({**http, "MLFLOW_TRACKING_USERNAME": "u"}),
            ["MLFLOW_TRACKING_PASSWORD"],
        )
        self.assertEqual(missing_mlflow_credentials({**http, "MLFLOW_TRACKING_TOKEN": "t"}), [])
        self.assertEqual(
            missing_mlflow_credentials(
                {**http, "MLFLOW_TRACKING_USERNAME": "u", "MLFLOW_TRACKING_PASSWORD": "p"}
            ),
            [],
        )

    def test_setup_refuses_missing_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(MlflowCredentialsError):
                setup_mlflow("x")

    def test_main_stops_before_any_llm_call_without_credentials(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("explorer_batch.main") as batch_main, \
                mock.patch("sys.stderr"):
            self.assertEqual(main(["-n", "1", "--yes"]), 2)
        batch_main.assert_not_called()


if __name__ == "__main__":
    unittest.main()
