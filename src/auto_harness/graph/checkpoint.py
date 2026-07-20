"""SQLite checkpoint manager for LangGraph.

Manages the lifecycle of the SQLite connection and SqliteSaver.
Only available when the langgraph extra is installed.
"""
import sqlite3
from pathlib import Path


class SqliteCheckpointManager:
    """Context manager for SQLite-based LangGraph checkpointing.

    Usage:
        with SqliteCheckpointManager(run_dir) as checkpoint:
            graph = build_graph(deps, checkpoint.saver)
            output = graph.invoke(state, config=checkpoint.config(task_id))
    """

    def __init__(self, run_dir: Path) -> None:
        self.path = Path(run_dir) / "checkpoints" / "langgraph.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.saver = None

    def __enter__(self):
        from langgraph.checkpoint.sqlite import SqliteSaver

        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self.saver = SqliteSaver(self.connection)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.saver = None

    @staticmethod
    def config(task_id):
        """Build the LangGraph config dict for a given task."""
        return {"configurable": {
            "thread_id": task_id,
        }}
