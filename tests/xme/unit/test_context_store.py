"""Unit tests for ContextStore UPSERT semantics."""
import pytest
import tempfile
from pathlib import Path
from xme.layers.context import ContextStore
from xme.models import WorkingContext


@pytest.fixture
def store(tmp_path):
    db = str(tmp_path / "test.db")
    s = ContextStore(db)
    s.connect()
    yield s
    s.close()


class TestContextStore:
    def test_get_nonexistent_returns_none(self, store):
        assert store.get("no-project", "no-user") is None

    def test_upsert_and_get(self, store):
        ctx = WorkingContext(
            project_id="proj", user_id="raj",
            current_task="Build XME",
            next_steps="Write tests",
        )
        store.upsert(ctx)
        retrieved = store.get("proj", "raj")
        assert retrieved is not None
        assert retrieved.current_task == "Build XME"
        assert retrieved.next_steps == "Write tests"

    def test_upsert_replaces_existing(self, store):
        ctx = WorkingContext(project_id="proj", user_id="raj", current_task="Task A")
        store.upsert(ctx)

        ctx2 = WorkingContext(project_id="proj", user_id="raj", current_task="Task B")
        store.upsert(ctx2)

        retrieved = store.get("proj", "raj")
        assert retrieved.current_task == "Task B"

    def test_different_users_isolated(self, store):
        store.upsert(WorkingContext(project_id="proj", user_id="raj", current_task="Task A"))
        store.upsert(WorkingContext(project_id="proj", user_id="alice", current_task="Task B"))

        raj = store.get("proj", "raj")
        alice = store.get("proj", "alice")
        assert raj.current_task == "Task A"
        assert alice.current_task == "Task B"

    def test_different_projects_isolated(self, store):
        store.upsert(WorkingContext(project_id="proj-a", user_id="raj", current_task="A"))
        store.upsert(WorkingContext(project_id="proj-b", user_id="raj", current_task="B"))

        a = store.get("proj-a", "raj")
        b = store.get("proj-b", "raj")
        assert a.current_task == "A"
        assert b.current_task == "B"

    def test_list_users(self, store):
        store.upsert(WorkingContext(project_id="proj", user_id="raj"))
        store.upsert(WorkingContext(project_id="proj", user_id="alice"))
        store.upsert(WorkingContext(project_id="other", user_id="bob"))

        users = store.list_users("proj")
        assert set(users) == {"raj", "alice"}

    def test_delete(self, store):
        store.upsert(WorkingContext(project_id="proj", user_id="raj"))
        store.delete("proj", "raj")
        assert store.get("proj", "raj") is None

    def test_update_from_session(self, store):
        ctx = store.update_from_session(
            project_id="proj",
            user_id="raj",
            session_summary="Fixed auth bug",
            recent_decisions=["Use JWT", "Use PostgreSQL"],
            next_steps="Deploy to staging",
            files_touched=["auth.py", "models.py"],
        )
        assert ctx.last_session_summary == "Fixed auth bug"
        assert "Use JWT" in ctx.recent_decisions
        assert ctx.next_steps == "Deploy to staging"
        assert "auth.py" in ctx.files_in_focus

    def test_update_from_session_keeps_max_5_decisions(self, store):
        for i in range(7):
            store.update_from_session(
                "proj", "raj",
                session_summary=f"session {i}",
                recent_decisions=[f"Decision {i}"],
            )
        ctx = store.get("proj", "raj")
        assert len(ctx.recent_decisions) <= 5

    def test_get_prompt_block_no_context(self, store):
        block = store.get_prompt_block("unknown", "user")
        assert "no context yet" in block.lower()

    def test_get_prompt_block_with_context(self, store):
        store.upsert(WorkingContext(
            project_id="proj", user_id="raj",
            current_task="Refactor auth",
            recent_decisions=["Use FastAPI"],
        ))
        block = store.get_prompt_block("proj", "raj")
        assert "Refactor auth" in block
        assert "FastAPI" in block
