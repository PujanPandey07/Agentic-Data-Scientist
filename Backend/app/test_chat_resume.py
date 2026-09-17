import unittest
from types import SimpleNamespace

from api.chat import chat
from schema.chat import ChatRequest


class _FakeResult:
    def __init__(self, conversation=None):
        self._conversation = conversation

    def scalar_one_or_none(self):
        return self._conversation


class _FakeSession:
    def __init__(self, conversation):
        self._conversation = conversation
        self.added = []

    async def execute(self, *args, **kwargs):
        return _FakeResult(self._conversation)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        return None


class _FakeTask:
    def __init__(self):
        self.interrupts = [SimpleNamespace(
            value={"summary": "waiting for approval"})]


class _FakeSnapshot:
    def __init__(self, values=None, next_=None, tasks=None):
        self.values = values
        self.next = next_ or []
        self.tasks = tasks or []


class _FakeGraph:
    def __init__(self):
        self.state = _FakeSnapshot(
            values=None,
            next_=["resume"],
            tasks=[_FakeTask()],
        )

    async def aget_state(self, config):
        return self.state

    async def ainvoke(self, payload, config):
        return {
            "__interrupt__": [{"value": {"summary": "waiting for approval"}}],
            "direct_answer": "stale answer from an earlier turn",
        }


class ChatResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_handles_missing_snapshot_values_and_dict_interrupt(self):
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(graph=_FakeGraph()))
        )
        session = _FakeSession(conversation=SimpleNamespace(id=17))

        response = await chat(
            payload=ChatRequest(thread_id="thread-123",
                                decision={"approved": True}),
            request=request,
            user_id=7,
            session=session,
        )

        self.assertTrue(response.interrupted)
        self.assertEqual(response.interrupt["summary"], "waiting for approval")
        self.assertIsNone(response.direct_answer)


if __name__ == "__main__":
    unittest.main()
