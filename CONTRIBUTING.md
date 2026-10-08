# Contributing to hermes

Thanks for helping build the hermes Q&A harness for Kenya Airways support.

## Getting started

1. Fork and clone the repo.
2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   `uv` works too: `uv venv --python 3.11 && uv pip install -r requirements.txt`

3. Copy `.env.example` to `.env` and fill in `DATABASE_URL` (Neon) and
   `GROQ_API_KEY`. Never commit `.env`.
4. Frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

5. Ingest data:

   ```bash
   python -m ingest.run --embed
   ```

## Branch naming

| Prefix   | Use for                          |
| -------- | -------------------------------- |
| `feat/`  | new features                     |
| `fix/`   | bug fixes                        |
| `docs/`  | documentation only               |
| `test/`  | adding or fixing tests           |
| `chore/` | maintenance, tooling, dependencies |
| `refactor/` | code changes with no behaviour change |

Example: `feat/chat-websocket-envelope`

## Commit guide (Conventional Commits)

Every commit message follows this shape:

```
<type>(<scope>): <imperative subject>
```

- **type**: `feat`, `fix`, `test`, `docs`, `refactor`, `chore`, `perf`, `ci`
- **scope** (optional but encouraged): the area touched, e.g. `chat`, `ingest`,
  `agents`, `dashboard`, `auth`, `ws`
- **subject**: imperative mood ("add", not "added" or "adds"), lowercase
  start, no trailing period, max ~72 characters

Examples:

```
feat(chat): add websocket envelope for bot tokens
fix(ingest): skip re-embedding when content hash unchanged
test(agents): cover keyword intent hard-trigger terms
docs: rewrite readme without em dashes
chore(ci): add ruff and pytest workflow
```

Body (optional): wrap at 72 chars, explain *why* not *what*.
Footer (optional): `BREAKING CHANGE:` or `Refs: #12`.

## Pull requests

- Keep PRs small and focused; one concern per PR.
- All checks must be green before review (lint, tests, build).
- Fill in the PR description: what changed, why, how to test.
- At least one review approval before merging to `main`.
- Rebase on `main` if your branch has drifted; avoid merge commits.

## Style notes

- Prose in docs and comments uses normal dashes, semicolons, or colons;
  no em dashes.
- Python: `ruff` for linting; type hints where practical.
- React: TypeScript for all new or touched frontend files (migration in
  progress; see the TS migration tasks).
- Keep secrets out of the repo: `.env` is gitignored, keys live in GitHub
  Secrets for CI.

## Reporting issues

Open an issue with: what you expected, what happened, steps to reproduce,
and your environment (OS, Python version, browser if frontend related).
