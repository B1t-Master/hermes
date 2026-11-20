"""Auth endpoints, security helpers and deps guards (hermetic, FakeDB)."""

import unittest
import uuid
from unittest.mock import patch

import jwt
from fastapi.testclient import TestClient

from app import security
from app.deps import get_current_agent, get_current_passenger
from app.db import get_db
from app.main import app
from app.models import Agent, Passenger
from tests.support.fakes import FakeDB

PASSENGER = security.PASSENGER
AGENT = security.AGENT


class SecurityTest(unittest.TestCase):
    def test_password_hash_roundtrip(self):
        hashed = security.hash_password("s3cret password")
        self.assertNotEqual(hashed, "s3cret password")
        self.assertTrue(security.verify_password("s3cret password", hashed))
        self.assertFalse(security.verify_password("wrong", hashed))

    def test_verify_password_with_garbage_hash_is_false(self):
        self.assertFalse(security.verify_password("x", "not-a-bcrypt-hash"))

    def test_access_token_roundtrip(self):
        token = security.create_access_token("subject-1", PASSENGER)
        payload = security.decode_access_token(token)
        self.assertEqual(payload["sub"], "subject-1")
        self.assertEqual(payload["role"], PASSENGER)

    def test_expired_token_is_rejected(self):
        token = security.create_access_token("s", PASSENGER, expires_minutes=-1)
        with self.assertRaises(jwt.InvalidTokenError):
            security.decode_access_token(token)


def _agent_auth_headers(agent_id=None):
    token = security.create_access_token(str(agent_id or uuid.uuid4()), AGENT)
    return {"Authorization": f"Bearer {token}"}


class AuthEndpointTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.db = FakeDB()
        app.dependency_overrides[get_db] = lambda: self.db

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_register_creates_a_passenger(self):
        resp = self.client.post("/auth/register", json={
            "first_name": "Ada", "last_name": "Lovelace",
            "email": "ada@example.com", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 201)
        body = resp.json()
        self.assertEqual(body["email"], "ada@example.com")
        passenger = self.db.added[0]
        self.assertNotEqual(passenger.password_hash, "long-password")

    def test_register_rejects_duplicate_email(self):
        self.db.register(Passenger(email="ada@example.com"))
        resp = self.client.post("/auth/register", json={
            "first_name": "Ada", "email": "ada@example.com",
            "password": "long-password",
        })
        self.assertEqual(resp.status_code, 409)

    def test_login_success_and_failure(self):
        passenger = Passenger(email="ada@example.com")
        passenger.password_hash = security.hash_password("long-password")
        self.db.register(passenger)
        resp = self.client.post("/auth/login", json={
            "email": "ada@example.com", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], PASSENGER)

        resp = self.client.post("/auth/login", json={
            "email": "ada@example.com", "password": "wrong-password",
        })
        self.assertEqual(resp.status_code, 401)
        resp = self.client.post("/auth/login", json={
            "email": "nobody@example.com", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 401)

    def test_anonymous_session_creates_a_token(self):
        resp = self.client.post("/auth/anonymous", json={"first_name": "Guest"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], PASSENGER)
        self.assertTrue(self.db.added[0].is_anonymous)

    def test_agent_register_and_login(self):
        resp = self.client.post("/auth/agent/register", json={
            "name": "Ada", "email": "agent@kq.co.ke", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 201)
        self.assertEqual(resp.json()["role"], AGENT)

        resp = self.client.post("/auth/agent/login", json={
            "username": "agent@kq.co.ke", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 200)

        resp = self.client.post("/auth/agent/login", json={
            "username": "agent@kq.co.ke", "password": "wrong",
        })
        self.assertEqual(resp.status_code, 401)

    def test_agent_login_duplicate_register_returns_409(self):
        self.db.register(Agent(email="dup@kq.co.ke", password_hash="x"))
        resp = self.client.post("/auth/agent/register", json={
            "name": "Dup", "email": "dup@kq.co.ke", "password": "long-password",
        })
        self.assertEqual(resp.status_code, 409)

    def test_agent_bootstrap_login_creates_the_env_agent(self):
        from app.config import settings

        resp = self.client.post("/auth/agent/login", json={
            "username": settings.agent_username,
            "password": settings.agent_password,
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], AGENT)
        agent = self.db.added[0]
        self.assertEqual(agent.email, settings.agent_username)

    def test_refresh_issues_a_new_token_for_an_existing_subject(self):
        agent = Agent(id=uuid.uuid4(), email="a@kq.co.ke")
        self.db.register(agent)
        token = security.create_access_token(str(agent.id), AGENT)
        resp = self.client.post(
            "/auth/refresh", json={}, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["role"], AGENT)

    def test_refresh_rejects_bad_tokens_and_unknown_roles(self):
        cases = [
            None,  # no credentials at all
            "Bearer garbage",
            f"Bearer {security.create_access_token(str(uuid.uuid4()), 'unknown-role')}",
        ]
        for credentials in cases:
            headers = {"Authorization": credentials} if credentials else {}
            resp = self.client.post("/auth/refresh", json={}, headers=headers)
            self.assertEqual(resp.status_code, 401, credentials)

    def test_refresh_rejects_deleted_subjects(self):
        token = security.create_access_token(str(uuid.uuid4()), PASSENGER)
        resp = self.client.post(
            "/auth/refresh", json={}, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(resp.status_code, 401)


class _ModelStub:
    id = None


class CurrentUserDepsTest(unittest.TestCase):
    """get_current_passenger / get_current_agent guards."""

    def setUp(self):
        self.client = TestClient(app)
        self.db = FakeDB()
        self.passenger = Passenger(id=uuid.uuid4(), email="p@example.com")
        self.agent = Agent(id=uuid.uuid4(), email="a@kq.co.ke")
        self.db.register(self.passenger, self.agent)
        app.dependency_overrides[get_db] = lambda: self.db
        # Probe endpoints that require each role.
        self.probe_passenger = "/chat/conversations"
        self.probe_agent = "/agents/metrics"

    def tearDown(self):
        app.dependency_overrides.clear()

    def test_passenger_happy_path_reaches_a_protected_route(self):
        token = security.create_access_token(str(self.passenger.id), PASSENGER)
        resp = self.client.get(
            self.probe_passenger, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_agent_happy_path_reaches_a_protected_route(self):
        token = security.create_access_token(str(self.agent.id), AGENT)
        resp = self.client.get(
            self.probe_agent, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(resp.status_code, 200)

    def test_missing_bad_and_wrong_role_tokens_are_401(self):
        bad_tokens = [
            None,
            "Bearer garbage",
            f"Bearer {security.create_access_token(str(self.passenger.id), 'admin')}",
        ]
        for credentials in bad_tokens:
            headers = {"Authorization": credentials} if credentials else {}
            self.assertEqual(
                self.client.get(self.probe_passenger, headers=headers).status_code,
                401, credentials,
            )

    def test_token_for_a_deleted_subject_is_401(self):
        token = security.create_access_token(str(uuid.uuid4()), PASSENGER)
        resp = self.client.get(
            self.probe_passenger, headers={"Authorization": f"Bearer {token}"}
        )
        self.assertEqual(resp.status_code, 401)


if __name__ == "__main__":
    unittest.main()
