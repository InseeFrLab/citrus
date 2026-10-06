import contextlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from src.llm.client import ask, trace_attributes


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class AskReasoningTest(unittest.TestCase):
    def test_default_keeps_model_behaviour(self):
        client = FakeClient()
        ask([{"role": "user", "content": "x"}], client=client, temperature=0)
        self.assertNotIn("reasoning_effort", client.calls[0])
        self.assertNotIn("reasoning", client.calls[0])
        self.assertEqual(client.calls[0]["temperature"], 0)

    def test_disabled_sends_reasoning_effort_none(self):
        client = FakeClient()
        ask([{"role": "user", "content": "x"}], client=client, reasoning=False)
        self.assertEqual(client.calls[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning", client.calls[0])


class TraceAttributesTest(unittest.TestCase):
    def test_without_langfuse_keys_is_an_empty_context(self):
        fake_langfuse = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.dict(sys.modules, {"langfuse": fake_langfuse}):
            self.assertIsInstance(trace_attributes(session_id="s"), contextlib.nullcontext)
        fake_langfuse.propagate_attributes.assert_not_called()

    def test_with_langfuse_keys_propagates_non_null_attributes(self):
        fake_langfuse = mock.Mock()
        keys = {"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}
        with mock.patch.dict(os.environ, keys, clear=True), \
                mock.patch.dict(sys.modules, {"langfuse": fake_langfuse}):
            trace_attributes(session_id=None, tags=["metier"], metadata={"annonce_id": "A1"})
        fake_langfuse.propagate_attributes.assert_called_once_with(
            tags=["metier"], metadata={"annonce_id": "A1"}
        )


if __name__ == "__main__":
    unittest.main()
