"""Agent dashboard API: escalation queue + claim/release/resolve workflow.

Claim is atomic (`UPDATE ... WHERE status = 'pending' RETURNING`), so two
agents clicking at the same time produce exactly one winner: the loser gets
409 and refetches the queue. Each mutation broadcasts a `queue_update` event
to every connected `/ws/dashboard` socket.

Transcript view (`GET .../messages`) is open to any authenticated agent so a
pending card can be inspected before claiming; *replying* requires the claim
(checked on every message by the conversation WS).

The routes live in ``queue``, ``payments`` and ``metrics``; importing this
package registers them on the shared ``router``.
"""

from . import metrics as metrics
from . import payments as payments
from . import queue as queue
from .common import _mean, router

__all__ = ["_mean", "router"]
