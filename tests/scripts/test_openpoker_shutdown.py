import asyncio
import json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.openpoker_bot import graceful_shutdown


class FakeWS:
    def __init__(self):
        self.sent = []
        self.closed = False
    async def send(self, msg):
        self.sent.append(json.loads(msg))
    async def close(self):
        self.closed = True


def test_graceful_shutdown_leaves_table_then_closes():
    ws = FakeWS()
    asyncio.run(graceful_shutdown(ws))
    assert {"type": "leave_table"} in ws.sent
    assert ws.closed
