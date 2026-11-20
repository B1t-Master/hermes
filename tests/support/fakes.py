"""Shared fakes for API-level tests.

``FakeDB`` is an ``AsyncSession`` stand-in that dispatches ``select`` calls by
model class and matches simple ``where`` equality criteria against in-memory
instances. It covers the surface the API routes actually use:
``execute``/``scalar_one_or_none``/``scalars().all``/``add``/``commit``/
``refresh``/``delete``. Register it with FastAPI's dependency overrides:

    app.dependency_overrides[get_db] = lambda: fake_db
"""

import uuid

from sqlalchemy import Delete, Insert, Update
from sqlalchemy.sql.selectable import Select
from sqlalchemy.sql.elements import BooleanClauseList
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.sql.operators import eq as _eq_op


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0] if self._rows else None

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def scalars(self):
        return self

    def unique(self):
        return self


class FakeDB:
    """In-memory AsyncSession stand-in keyed by model class.

    ``store`` maps model classes to lists of instances. ``execute(select(Model))
    ...`` returns the instances of ``Model`` whose attributes match every
    simple ``column == literal`` criterion in the where clause. Criteria that
    cannot be evaluated (e.g. column-to-column comparisons) match nothing,
    except when there is no where clause at all (then everything matches).
    """

    def __init__(self, store=None):
        self.store = store or {}
        self.added = []
        self.commits = 0
        self.deleted = []

    # -- statement helpers -------------------------------------------------
    @staticmethod
    def _entity(stmt):
        try:
            desc = stmt.column_descriptions[0]
        except (AttributeError, IndexError):
            return None
        return desc.get("entity") or desc.get("type")

    @staticmethod
    def _criteria(whereclause):
        """Extract {attr_name: literal_value} from simple == criteria."""
        out = {}

        def add_binary(node):
            if not isinstance(node, BinaryExpression) or node.operator is not _eq_op:
                return False
            name = getattr(node.left, "key", None)
            right = node.right
            if name is None:
                return False
            value = getattr(right, "value", None)
            if getattr(right, "is_literal", False) or value is not None:
                out[name] = value
                return True
            return False

        def walk(node):
            if node is None:
                return
            if isinstance(node, BooleanClauseList):
                for child in list(node):
                    if not add_binary(child):
                        # Un-evaluable criterion: nothing can match.
                        out["__impossible__"] = True
            else:
                if not add_binary(node):
                    out["__impossible__"] = True

        walk(whereclause)
        return out

    # -- session surface ---------------------------------------------------
    async def execute(self, stmt):
        if isinstance(stmt, (Delete, Update, Insert)):
            self.deleted.append(stmt)
            return FakeResult([])
        entity = self._entity(stmt)
        rows = list(self.store.get(entity, [])) + [
            o for o in self.added if isinstance(o, entity)
        ]
        if entity is None:
            rows = []
        criteria = self._criteria(getattr(stmt, "whereclause", None))
        if criteria:
            if criteria.pop("__impossible__", False):
                rows = []
            else:
                rows = [
                    row for row in rows
                    if all(getattr(row, key, None) == val for key, val in criteria.items())
                ]
        return FakeResult(rows)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1
        for obj in self.added:
            self._apply_defaults(obj)

    async def refresh(self, obj):
        self._apply_defaults(obj)

    @staticmethod
    def _apply_defaults(obj):
        if getattr(obj, "id", None) is None and hasattr(obj, "id"):
            obj.id = uuid.uuid4()
        if getattr(obj, "created_at", None) is None and hasattr(obj, "created_at"):
            import datetime

            obj.created_at = datetime.datetime.now(datetime.timezone.utc)

    def register(self, *objects):
        for obj in objects:
            self.store.setdefault(type(obj), []).append(obj)


class FakeHub:
    """Stand-in for ``app.agents.hub.conversation_hub``."""

    def __init__(self):
        self.sent = []

    async def send(self, conversation_id, message):
        self.sent.append((conversation_id, message))

    async def connect(self, *args, **kwargs):
        return None

    async def disconnect(self, *args, **kwargs):
        return None
