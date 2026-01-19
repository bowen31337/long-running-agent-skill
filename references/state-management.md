# State Management Patterns

## Workflow State Machine

Define explicit states for agent progression. Use an enum for type safety:

```python
from enum import Enum

class WorkflowState(Enum):
    START = "START"
    INITIALIZING = "INITIALIZING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
```

### State Persistence

Store workflow state in JSON with atomic writes:

```python
import json
import os
from pathlib import Path
from datetime import datetime

def save_workflow_state(project_dir: Path, state: WorkflowState, metadata: dict = None):
    """Save workflow state atomically via temp file + rename."""
    state_file = project_dir / "workflow-state.json"
    temp_file = state_file.with_suffix(".tmp")

    data = {
        "current_state": state.value,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }

    with open(temp_file, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, state_file)  # Atomic on POSIX

def load_workflow_state(project_dir: Path) -> dict:
    """Load workflow state, returning defaults if missing."""
    state_file = project_dir / "workflow-state.json"
    if state_file.exists():
        return json.loads(state_file.read_text())
    return {"current_state": "START", "timestamp": None, "metadata": {}}
```

## Task/Feature List Management

Track individual tasks with status flags:

```python
@dataclass
class TaskState:
    id: int
    name: str
    description: str
    is_started: bool = False
    is_complete: bool = False
    is_validated: bool = False
    error_message: str | None = None
    attempts: int = 0

class TaskStateManager:
    """Thread-safe task list management."""

    _instances: dict[str, "TaskStateManager"] = {}
    _lock = threading.Lock()

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.tasks_file = project_dir / "task_list.json"
        self._file_lock = threading.Lock()

    @classmethod
    def get_instance(cls, project_dir: Path) -> "TaskStateManager":
        """Singleton pattern prevents memory leaks in long sessions."""
        key = str(project_dir.resolve())
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(project_dir)
            return cls._instances[key]

    def update_task(self, task_id: int, **updates) -> None:
        """Update task with file locking."""
        with self._file_lock:
            tasks = self._load_tasks()
            for task in tasks:
                if task["id"] == task_id:
                    task.update(updates)
                    break
            self._save_tasks(tasks)

    def get_next_pending_task(self) -> dict | None:
        """Get next task where is_started=false."""
        tasks = self._load_tasks()
        for task in tasks:
            if not task.get("is_started") and not task.get("is_complete"):
                return task
        return None

    def get_tasks_for_validation(self) -> list[dict]:
        """Get tasks where is_complete=true, is_validated=false."""
        tasks = self._load_tasks()
        return [t for t in tasks if t.get("is_complete") and not t.get("is_validated")]
```

## Signal File Coordination

For multi-process agent coordination, use signal files:

```python
from pathlib import Path
import json
from datetime import datetime

SIGNALS_DIR = ".agent-signals"

def write_completion_signal(
    project_dir: Path,
    agent_type: str,
    status: str,
    next_state: str,
    metadata: dict = None
) -> Path:
    """Write agent completion signal for orchestrator polling."""
    signals_dir = project_dir / SIGNALS_DIR
    signals_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    signal_file = signals_dir / f"{agent_type.lower()}-{timestamp}.json"

    signal_data = {
        "agent_type": agent_type,
        "status": status,
        "next_state": next_state,
        "timestamp": datetime.utcnow().isoformat(),
        "metadata": metadata or {}
    }

    signal_file.write_text(json.dumps(signal_data, indent=2))
    return signal_file

def poll_signals(project_dir: Path) -> list[dict]:
    """Poll for unprocessed agent signals."""
    signals_dir = project_dir / SIGNALS_DIR
    if not signals_dir.exists():
        return []

    signals = []
    for signal_file in sorted(signals_dir.glob("*.json")):
        if signal_file.name.startswith("."):
            continue
        data = json.loads(signal_file.read_text())
        data["_file_path"] = signal_file
        signals.append(data)
    return signals

def archive_signal(signal_path: Path) -> None:
    """Move processed signal to archive directory."""
    archive_dir = signal_path.parent / "processed"
    archive_dir.mkdir(exist_ok=True)
    signal_path.rename(archive_dir / signal_path.name)
```

## Bounded Collections for Memory Safety

Prevent memory leaks in long-running sessions by bounding collections:

```python
@dataclass
class SessionState:
    """Track session with bounded lists."""

    MAX_FILES = 100
    MAX_TOOLS = 50
    MAX_TASKS = 50

    files_read: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tasks_worked: list[int] = field(default_factory=list)

    def record_file(self, path: str) -> None:
        self.files_read.append(path)
        self._truncate_lists()

    def _truncate_lists(self) -> None:
        """Truncate lists to prevent unbounded growth."""
        if len(self.files_read) > self.MAX_FILES:
            self.files_read = self.files_read[-self.MAX_FILES:]
        if len(self.tools_used) > self.MAX_TOOLS:
            self.tools_used = self.tools_used[-self.MAX_TOOLS:]
        if len(self.tasks_worked) > self.MAX_TASKS:
            self.tasks_worked = self.tasks_worked[-self.MAX_TASKS:]
```

## State Transition Validation

Prevent invalid state transitions:

```python
VALID_TRANSITIONS = {
    WorkflowState.START: [WorkflowState.INITIALIZING],
    WorkflowState.INITIALIZING: [WorkflowState.PLANNING, WorkflowState.FAILED],
    WorkflowState.PLANNING: [WorkflowState.EXECUTING, WorkflowState.FAILED],
    WorkflowState.EXECUTING: [WorkflowState.VALIDATING, WorkflowState.FAILED],
    WorkflowState.VALIDATING: [WorkflowState.COMPLETE, WorkflowState.EXECUTING, WorkflowState.FAILED],
    WorkflowState.COMPLETE: [],
    WorkflowState.FAILED: [WorkflowState.START],  # Allow retry
}

def transition_state(current: WorkflowState, target: WorkflowState) -> bool:
    """Validate and perform state transition."""
    if target not in VALID_TRANSITIONS.get(current, []):
        raise ValueError(f"Invalid transition: {current.value} -> {target.value}")
    return True
```
