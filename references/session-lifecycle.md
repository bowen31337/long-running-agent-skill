# Session Lifecycle Patterns

## Process Registration and Cleanup

Register spawned processes for cleanup on exit:

```python
import os
import sys
import signal
import atexit
from pathlib import Path

# Global process tracking
_spawned_pids: list[int] = []
_pid_file: Path | None = None
_cleanup_registered = False

def register_cleanup_handler(project_dir: Path) -> None:
    """Register cleanup handlers for graceful shutdown."""
    global _pid_file, _cleanup_registered

    if _cleanup_registered:
        return

    _pid_file = project_dir / ".agent-pids"

    # Register atexit handler
    atexit.register(cleanup_session_processes)

    # Register signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        original_handler = signal.getsignal(sig)

        def handler(signum, frame, original=original_handler):
            cleanup_session_processes()
            if callable(original):
                original(signum, frame)
            sys.exit(128 + signum)

        signal.signal(sig, handler)

    _cleanup_registered = True

def register_spawned_pid(pid: int) -> None:
    """Register a spawned process for cleanup tracking."""
    global _spawned_pids

    if pid not in _spawned_pids:
        _spawned_pids.append(pid)
        _save_pids()

def _save_pids() -> None:
    """Persist PIDs to file for crash recovery."""
    if _pid_file:
        _pid_file.write_text("\n".join(str(p) for p in _spawned_pids))

def _load_pids() -> list[int]:
    """Load PIDs from previous session."""
    if _pid_file and _pid_file.exists():
        try:
            return [int(p) for p in _pid_file.read_text().strip().split("\n") if p]
        except ValueError:
            return []
    return []

def cleanup_session_processes() -> None:
    """Kill all tracked processes on exit."""
    global _spawned_pids

    # Load any orphaned PIDs from previous session
    all_pids = set(_spawned_pids + _load_pids())

    for pid in all_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass  # Already dead
        except PermissionError:
            pass  # Not our process

    # Clean up PID file
    if _pid_file and _pid_file.exists():
        _pid_file.unlink()

    _spawned_pids.clear()

def cleanup_orphaned_processes(
    process_names: list[str] = None,
    dry_run: bool = False
) -> list[int]:
    """Clean orphaned processes from crashed sessions."""
    import subprocess

    if process_names is None:
        # Common development processes
        process_names = [
            "next-server",
            "vite",
            "vitest",
            "jest",
            "esbuild",
            "webpack",
            "uvicorn",
            "gunicorn",
            "node",
        ]

    killed = []

    for name in process_names:
        # Find processes by name
        result = subprocess.run(
            ["pgrep", "-f", name],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = [int(p) for p in result.stdout.strip().split("\n") if p]

            for pid in pids:
                # Skip our own process
                if pid == os.getpid():
                    continue

                if dry_run:
                    killed.append(pid)
                else:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        killed.append(pid)
                    except (ProcessLookupError, PermissionError):
                        pass

    return killed
```

## Context Capacity Management

Track and manage context window usage:

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ContextState:
    """Track context window usage."""

    # Token tracking
    input_tokens: int = 0
    output_tokens: int = 0
    max_context_tokens: int = 180_000  # Buffer from 200K limit

    # Bounded lists (prevent memory leaks)
    files_read: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tasks_worked: list[int] = field(default_factory=list)

    # Limits
    MAX_FILES = 100
    MAX_TOOLS = 50
    MAX_TASKS = 50

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def is_context_full(self, threshold: float = 0.9) -> bool:
        """Check if approaching context capacity."""
        return self.total_tokens >= (self.max_context_tokens * threshold)

    def record_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage from API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def record_file(self, path: str) -> None:
        """Record file read into context."""
        if path not in self.files_read:
            self.files_read.append(path)
            self._truncate_lists()

    def record_tool(self, tool_name: str) -> None:
        """Record tool usage."""
        self.tools_used.append(tool_name)
        self._truncate_lists()

    def record_task(self, task_id: int) -> None:
        """Record task worked on."""
        if task_id not in self.tasks_worked:
            self.tasks_worked.append(task_id)
            self._truncate_lists()

    def _truncate_lists(self) -> None:
        """Truncate bounded lists to prevent memory growth."""
        if len(self.files_read) > self.MAX_FILES:
            self.files_read = self.files_read[-self.MAX_FILES:]
        if len(self.tools_used) > self.MAX_TOOLS:
            self.tools_used = self.tools_used[-self.MAX_TOOLS:]
        if len(self.tasks_worked) > self.MAX_TASKS:
            self.tasks_worked = self.tasks_worked[-self.MAX_TASKS:]

    def get_continuation_context(self) -> dict:
        """Get context summary for session continuation."""
        return {
            "total_tokens_used": self.total_tokens,
            "recent_files": self.files_read[-10:],
            "recent_tools": self.tools_used[-10:],
            "tasks_worked": self.tasks_worked,
        }

class SessionContinuationManager:
    """Manage intelligent session continuation."""

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.state_file = project_dir / ".session-state.json"
        self.context = ContextState()

    def save_state(self) -> None:
        """Save session state for continuation."""
        import json

        data = {
            "input_tokens": self.context.input_tokens,
            "output_tokens": self.context.output_tokens,
            "files_read": self.context.files_read,
            "tools_used": self.context.tools_used,
            "tasks_worked": self.context.tasks_worked,
        }

        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(self.state_file)

    def load_state(self) -> None:
        """Load previous session state."""
        import json

        if not self.state_file.exists():
            return

        data = json.loads(self.state_file.read_text())

        self.context.input_tokens = data.get("input_tokens", 0)
        self.context.output_tokens = data.get("output_tokens", 0)
        self.context.files_read = data.get("files_read", [])
        self.context.tools_used = data.get("tools_used", [])
        self.context.tasks_worked = data.get("tasks_worked", [])

    def should_create_new_session(self) -> bool:
        """Determine if new session needed (context nearly full)."""
        return self.context.is_context_full(threshold=0.85)

    def get_handoff_summary(self) -> str:
        """Generate summary for new session handoff."""
        return f"""Session Continuation Context:
- Tokens used: {self.context.total_tokens:,} / {self.context.max_context_tokens:,}
- Files read: {len(self.context.files_read)} total, recent: {', '.join(self.context.files_read[-5:])}
- Tasks worked: {self.context.tasks_worked}

Resume from where previous session left off."""
```

## Memory Monitoring

Monitor memory usage for leak detection:

```python
import os
import psutil
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MemorySnapshot:
    timestamp: str
    rss_mb: float
    vms_mb: float
    context_label: str

class MemoryMonitor:
    """Monitor memory usage and detect leaks."""

    WARNING_THRESHOLD_MB = 4000
    CRITICAL_THRESHOLD_MB = 6000

    def __init__(self):
        self.snapshots: list[MemorySnapshot] = []
        self.process = psutil.Process(os.getpid())

    def check_and_log(self, context_label: str = "") -> MemorySnapshot:
        """Take memory snapshot and check thresholds."""
        mem_info = self.process.memory_info()

        snapshot = MemorySnapshot(
            timestamp=datetime.utcnow().isoformat(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            context_label=context_label
        )

        self.snapshots.append(snapshot)

        # Bound snapshot list
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-100:]

        return snapshot

    def is_memory_warning(self) -> bool:
        """Check if memory exceeds warning threshold."""
        if not self.snapshots:
            self.check_and_log()
        return self.snapshots[-1].rss_mb >= self.WARNING_THRESHOLD_MB

    def is_memory_critical(self) -> bool:
        """Check if memory exceeds critical threshold."""
        if not self.snapshots:
            self.check_and_log()
        return self.snapshots[-1].rss_mb >= self.CRITICAL_THRESHOLD_MB

    def detect_memory_leak(self, window_size: int = 10) -> bool:
        """Detect potential memory leak (consistent growth)."""
        if len(self.snapshots) < window_size:
            return False

        recent = self.snapshots[-window_size:]

        # Check for consistent growth
        increases = 0
        for i in range(1, len(recent)):
            if recent[i].rss_mb > recent[i-1].rss_mb:
                increases += 1

        # If >80% of measurements show increase, likely leak
        return increases / (window_size - 1) > 0.8

    def get_memory_report(self) -> dict:
        """Generate memory usage report."""
        if not self.snapshots:
            self.check_and_log()

        recent = self.snapshots[-10:] if len(self.snapshots) >= 10 else self.snapshots

        return {
            "current_rss_mb": self.snapshots[-1].rss_mb,
            "current_vms_mb": self.snapshots[-1].vms_mb,
            "peak_rss_mb": max(s.rss_mb for s in self.snapshots),
            "average_rss_mb": sum(s.rss_mb for s in recent) / len(recent),
            "is_warning": self.is_memory_warning(),
            "is_critical": self.is_memory_critical(),
            "potential_leak": self.detect_memory_leak(),
            "snapshot_count": len(self.snapshots),
        }
```

## Token Tracking

Track API token consumption:

```python
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

@dataclass
class APICall:
    timestamp: str
    endpoint: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

class TokenTracker:
    """Track API token consumption."""

    LOG_FILE = "token-usage.log"
    REPORT_FILE = "token-consumption-report.json"
    MAX_CALLS = 100  # Keep last 100 calls in memory

    # Pricing (per 1M tokens, approximate)
    PRICING = {
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
    }

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.log_file = project_dir / "logs" / self.LOG_FILE
        self.report_file = project_dir / self.REPORT_FILE
        self.calls: list[APICall] = []

        # Ensure logs directory exists
        self.log_file.parent.mkdir(exist_ok=True)

    def record_call(
        self,
        endpoint: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0
    ) -> None:
        """Record API call."""
        call = APICall(
            timestamp=datetime.utcnow().isoformat(),
            endpoint=endpoint,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms
        )

        self.calls.append(call)

        # Bound list
        if len(self.calls) > self.MAX_CALLS:
            self.calls = self.calls[-self.MAX_CALLS:]

        # Log to file
        log_entry = f"{call.timestamp} | {endpoint} | {model} | in:{input_tokens} out:{output_tokens} | {latency_ms:.0f}ms\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)

        # Update report
        self._update_report()

    def _update_report(self) -> None:
        """Update consumption report."""
        total_input = sum(c.input_tokens for c in self.calls)
        total_output = sum(c.output_tokens for c in self.calls)

        # Calculate estimated cost
        estimated_cost = 0
        for call in self.calls:
            pricing = self.PRICING.get(call.model, self.PRICING["claude-3-sonnet"])
            estimated_cost += (call.input_tokens / 1_000_000) * pricing["input"]
            estimated_cost += (call.output_tokens / 1_000_000) * pricing["output"]

        report = {
            "total_calls": len(self.calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "estimated_cost_usd": round(estimated_cost, 4),
            "average_input_tokens": total_input // len(self.calls) if self.calls else 0,
            "average_output_tokens": total_output // len(self.calls) if self.calls else 0,
            "last_updated": datetime.utcnow().isoformat(),
        }

        temp_file = self.report_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(report, indent=2))
        temp_file.replace(self.report_file)

    def get_report(self) -> dict:
        """Get current consumption report."""
        if self.report_file.exists():
            return json.loads(self.report_file.read_text())
        return {}
```

## Session Lifecycle Integration

Tie everything together in session lifecycle:

```python
from pathlib import Path
from contextlib import contextmanager

@contextmanager
def agent_session(project_dir: Path):
    """Context manager for agent session lifecycle."""

    # Initialize components
    project_dir = Path(project_dir).resolve()

    # Register cleanup
    register_cleanup_handler(project_dir)

    # Initialize tracking
    memory_monitor = MemoryMonitor()
    continuation_manager = SessionContinuationManager(project_dir)
    token_tracker = TokenTracker(project_dir)

    # Load previous state if continuing
    continuation_manager.load_state()

    session_context = {
        "project_dir": project_dir,
        "memory_monitor": memory_monitor,
        "continuation_manager": continuation_manager,
        "token_tracker": token_tracker,
    }

    try:
        yield session_context

    finally:
        # Save state for continuation
        continuation_manager.save_state()

        # Final memory check
        memory_monitor.check_and_log("session_end")

        # Process cleanup handled by atexit/signals
```
