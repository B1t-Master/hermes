from app import db as db_module


def test_engine_kwargs_bound_connect_and_command_timeouts():
    """A dead DB must fail fast instead of hanging until the browser gives up."""
    connect_args = db_module._engine_kwargs()["connect_args"]

    assert connect_args["timeout"] == db_module.settings.db_connect_timeout_seconds
    assert connect_args["command_timeout"] == db_module.settings.db_command_timeout_seconds
    assert 0 < connect_args["timeout"] <= 30
    assert 0 < connect_args["command_timeout"] <= 300
