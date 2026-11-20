"""Shared fakes for API-level tests.

``FakeDB`` is an ``AsyncSession`` stand-in that dispatches ``select`` calls by
model class and matches simple ``where`` equality criteria against in-memory
instances. It covers the surface the API routes actually use:
``execute``/``scalar_one_or_none``/``scalars().all``/``add``/``commit``/
``refresh``/``delete``. Register it with FastAPI's dependency overrides:

    app.dependency_overrides[get_db] = lambda: fake_db
"""

import uuid

from sqlalchemy import Delete, Update
from sqlalchemy.sql.selectable import Select
from sqlalchemy.sql.elements import BooleanClauseList
from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy.sql.operators import eq as _eq_op, is_not as _is_not_op, ne as _ne_op

# Filled lazily so the fake does not import app.models eagerly.
_model_registry = []

# Sentinel for ``column.is_not(None)`` criteria.
_NOT_NONE = object()

# Sentinel for "no value could be computed" in UPDATE set-clauses.
_SKIP = object()


class _NotEqual:
    """Sentinel wrapper for ``column != literal`` criteria."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar(self):
        return self._rows[0] if self._rows else None

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one(self):
        if not self._rows:
            raise IndexError("no rows")
        return self._rows[0]

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

    Also supports the bits the dashboard routes use: ``select(Model.col)``
    (returns attribute values), ``update(...).values(...).returning(Model.id)``
    (mutates matching rows, returns their ids) and ``select(func.count())``
    (returns the match count).
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
            if not isinstance(node, BinaryExpression):
                return False
            name = getattr(node.left, "key", None)
            right = node.right
            if name is None:
                return False
            if node.operator is _eq_op:
                value = getattr(right, "value", None)
                if getattr(right, "is_literal", False) or value is not None:
                    out[name] = value
                    return True
                return False
            if node.operator in (_ne_op, _is_not_op) and getattr(right, "value", None) is None:
                out[name] = _NOT_NONE
                return True
            if node.operator is _ne_op:
                out[name] = _NotEqual(getattr(right, "value", None))
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

    def _rows_for(self, entity, criteria):
        rows = list(self.store.get(entity, [])) + [
            o for o in self.added if entity is not None and isinstance(o, entity)
        ]
        if entity is None:
            return []
        if criteria:
            if criteria.pop("__impossible__", False):
                return []
            return [
                row for row in rows
                if all(
                    (getattr(row, key, None) is not None)
                    if val is _NOT_NONE
                    else (getattr(row, key, None) != val.value)
                    if isinstance(val, _NotEqual)
                    else (getattr(row, key, None) == val)
                    for key, val in criteria.items()
                )
            ]
        return rows

    @staticmethod
    def _update_target(stmt):
        table = getattr(stmt, "table", None)
        if table is None:
            return None
        return next(
            (
                cls
                for cls in _model_registry
                if getattr(cls, "__table__", None) is not None
                and cls.__table__.name == table.name
            ),
            None,
        )

    # -- session surface ---------------------------------------------------
    async def execute(self, stmt):
        if isinstance(stmt, Delete):
            self.deleted.append(stmt)
            return FakeResult([])
        if isinstance(stmt, Update):
            entity = self._update_target(stmt)
            matches = self._rows_for(entity, self._criteria(stmt.whereclause))
            if matches:
                set_values = getattr(stmt, "_values", None) or {}
                from sqlalchemy.sql.expression import BindParameter
                from sqlalchemy.sql.expression import BinaryExpression as _BinExpr

                for row in matches:
                    for column, value in set_values.items():
                        applied = _SKIP
                        if isinstance(value, BindParameter):
                            applied = value.value
                        elif isinstance(value, _BinExpr):
                            # In-place arithmetic like turn_count + 1.
                            left_key = getattr(value.left, "key", None)
                            step = getattr(value.right, "value", None)
                            if left_key and step is not None and hasattr(row, left_key):
                                applied = (getattr(row, left_key) or 0) + step
                        if applied is not _SKIP and hasattr(row, column.key):
                            setattr(row, column.key, applied)
            return FakeResult([row.id for row in matches])
        descs = list(getattr(stmt, "column_descriptions", None) or [])
        if not descs:
            return FakeResult([])
        entity = self._entity(stmt)
        rows = self._rows_for(entity, self._criteria(stmt.whereclause))

        def value_for(row, desc):
            # select(Model) -> the instance; select(Model.col) -> the value.
            if desc.get("entity") is not None and desc.get("type") is desc["entity"]:
                return row
            name = desc.get("name")
            if name is None:
                return None
            return getattr(row, name, None)

        if len(descs) == 1:
            desc = descs[0]
            name = str(desc.get("name") or "")
            if name.startswith("count"):
                # Aggregates (func.count): the fake reports the match count.
                return FakeResult([len(rows)])
            return FakeResult([value_for(row, desc) for row in rows])
        return FakeResult([tuple(value_for(row, d) for d in descs) for row in rows])

    def add(self, obj):
        self.added.append(obj)

    async def flush(self):
        self._apply_defaults_on_flush = True

    async def rollback(self):
        return None

    async def commit(self):
        self.commits += 1
        for obj in self.added:
            self._apply_defaults(obj)

    async def refresh(self, obj):
        self._apply_defaults(obj)

    @staticmethod
    def _apply_defaults(obj):
        table = getattr(type(obj), "__table__", None)
        if table is not None:
            for column in table.columns:
                if getattr(obj, column.key, None) is not None:
                    continue
                default = getattr(column, "default", None)
                if default is None:
                    continue
                try:
                    if default.is_scalar:
                        setattr(obj, column.key, default.arg)
                    elif default.is_callable:
                        setattr(obj, column.key, default.arg(None))
                except Exception:
                    continue
        if getattr(obj, "id", None) is None and hasattr(obj, "id"):
            obj.id = uuid.uuid4()
        if getattr(obj, "created_at", None) is None and hasattr(obj, "created_at"):
            import datetime

            obj.created_at = datetime.datetime.now(datetime.timezone.utc)

    def register(self, *objects):
        for obj in objects:
            self.store.setdefault(type(obj), []).append(obj)
            if type(obj) not in _model_registry:
                _model_registry.append(type(obj))


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
