# Error Handling Patterns

## Error Classification

Classify errors to determine appropriate recovery strategy:

```python
from enum import Enum
from dataclasses import dataclass
import re

class ErrorType(Enum):
    TRANSIENT = "transient"        # Network, rate limit -> retry with backoff
    INFRASTRUCTURE = "infrastructure"  # Missing tools -> self-heal
    TASK_SPECIFIC = "task_specific"    # Code error -> mark task and continue
    FATAL = "fatal"                    # Unrecoverable -> stop agent
    UNKNOWN = "unknown"

@dataclass
class ErrorPattern:
    patterns: list[str]  # Regex patterns
    error_type: ErrorType
    is_recoverable: bool
    suggested_action: str
    max_retries: int = 3

ERROR_PATTERNS = [
    # Transient errors - retry with backoff
    ErrorPattern(
        patterns=[
            r"rate.?limit",
            r"429",
            r"too many requests",
            r"connection.?reset",
            r"connection.?refused",
            r"timeout",
            r"ECONNRESET",
            r"socket hang up",
        ],
        error_type=ErrorType.TRANSIENT,
        is_recoverable=True,
        suggested_action="Wait and retry with exponential backoff",
        max_retries=5
    ),

    # Infrastructure errors - attempt self-heal
    ErrorPattern(
        patterns=[
            r"command not found",
            r"module.?not.?found",
            r"no such file or directory",
            r"permission denied",
            r"ENOENT",
            r"spawn.*failed",
        ],
        error_type=ErrorType.INFRASTRUCTURE,
        is_recoverable=True,
        suggested_action="Attempt infrastructure repair",
        max_retries=2
    ),

    # Task-specific errors - mark and move on
    ErrorPattern(
        patterns=[
            r"syntax.?error",
            r"type.?error",
            r"reference.?error",
            r"compilation.?failed",
            r"test.?failed",
            r"assertion.?error",
        ],
        error_type=ErrorType.TASK_SPECIFIC,
        is_recoverable=True,
        suggested_action="Mark task as failed, continue with other tasks",
        max_retries=2
    ),

    # Fatal errors - stop
    ErrorPattern(
        patterns=[
            r"all.?api.?keys.?exhausted",
            r"authentication.?failed",
            r"invalid.?api.?key",
            r"billing",
            r"quota.?exceeded",
            r"out of memory",
        ],
        error_type=ErrorType.FATAL,
        is_recoverable=False,
        suggested_action="Stop agent, require user intervention",
        max_retries=0
    ),
]

def classify_error(error_text: str, status_code: int = None) -> tuple[ErrorType, ErrorPattern | None]:
    """Classify error and return appropriate pattern."""
    error_lower = error_text.lower()

    for pattern in ERROR_PATTERNS:
        for regex in pattern.patterns:
            if re.search(regex, error_lower, re.IGNORECASE):
                return pattern.error_type, pattern

    # Status code fallback
    if status_code:
        if status_code == 429:
            return ErrorType.TRANSIENT, ERROR_PATTERNS[0]
        if status_code in (401, 403):
            return ErrorType.FATAL, ERROR_PATTERNS[3]
        if status_code >= 500:
            return ErrorType.TRANSIENT, ERROR_PATTERNS[0]

    return ErrorType.UNKNOWN, None
```

## Recovery Strategies

Implement recovery strategies per error type:

```python
from abc import ABC, abstractmethod
import subprocess
import asyncio

class RecoveryStrategy(ABC):
    """Base class for recovery strategies."""

    @abstractmethod
    def can_handle(self, error_type: ErrorType, error_text: str) -> bool:
        """Check if this strategy can handle the error."""
        pass

    @abstractmethod
    async def execute(self, context: dict) -> bool:
        """Execute recovery, return True if successful."""
        pass

class BackoffRetryStrategy(RecoveryStrategy):
    """Retry with exponential backoff for transient errors."""

    def can_handle(self, error_type: ErrorType, error_text: str) -> bool:
        return error_type == ErrorType.TRANSIENT

    async def execute(self, context: dict) -> bool:
        attempt = context.get("attempt", 0)
        max_attempts = context.get("max_attempts", 5)

        if attempt >= max_attempts:
            return False

        # Exponential backoff: 1s, 2s, 4s, 8s, 16s...
        delay = min(2 ** attempt, 60)
        await asyncio.sleep(delay)

        return True  # Signal to retry

class PackageSyncStrategy(RecoveryStrategy):
    """Sync packages for missing module errors."""

    def can_handle(self, error_type: ErrorType, error_text: str) -> bool:
        if error_type != ErrorType.INFRASTRUCTURE:
            return False
        return any(x in error_text.lower() for x in ["module", "import", "package"])

    async def execute(self, context: dict) -> bool:
        project_dir = context.get("project_dir")

        # Try common package managers
        for cmd in [
            ["uv", "sync"],
            ["npm", "install"],
            ["uv", "pip", "install", "-r", "requirements.txt"],
        ]:
            result = subprocess.run(
                cmd,
                cwd=project_dir,
                capture_output=True
            )
            if result.returncode == 0:
                return True

        return False

class LockCleanupStrategy(RecoveryStrategy):
    """Clean stale locks for deadlock situations."""

    def can_handle(self, error_type: ErrorType, error_text: str) -> bool:
        return "lock" in error_text.lower() or "deadlock" in error_text.lower()

    async def execute(self, context: dict) -> bool:
        from pathlib import Path

        project_dir = Path(context.get("project_dir", "."))
        lock_dir = project_dir / ".locks"

        if not lock_dir.exists():
            return False

        cleaned = 0
        for lock_file in lock_dir.glob("*.lock"):
            lock_file.unlink(missing_ok=True)
            cleaned += 1

        return cleaned > 0

class TaskMarkFailedStrategy(RecoveryStrategy):
    """Mark task as failed and continue."""

    def can_handle(self, error_type: ErrorType, error_text: str) -> bool:
        return error_type == ErrorType.TASK_SPECIFIC

    async def execute(self, context: dict) -> bool:
        task_manager = context.get("task_manager")
        task_id = context.get("task_id")
        error_text = context.get("error_text")

        if task_manager and task_id:
            task_manager.update_task(
                task_id,
                is_started=False,
                error_message=error_text,
                attempts=context.get("attempts", 0) + 1
            )
            return True

        return False

class RecoveryEngine:
    """Orchestrate recovery strategy selection and execution."""

    def __init__(self):
        self.strategies = [
            BackoffRetryStrategy(),
            PackageSyncStrategy(),
            LockCleanupStrategy(),
            TaskMarkFailedStrategy(),
        ]

    async def attempt_recovery(
        self,
        error_type: ErrorType,
        error_text: str,
        context: dict
    ) -> bool:
        """Try recovery strategies in order."""

        for strategy in self.strategies:
            if strategy.can_handle(error_type, error_text):
                try:
                    success = await strategy.execute(context)
                    if success:
                        return True
                except Exception:
                    continue

        return False
```

## Error Reporting

Track and report errors for debugging:

```python
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

class ErrorReporter:
    """Track and report errors for debugging."""

    ERROR_LOG_FILE = "agent-errors.json"
    MAX_ERRORS = 100  # Bounded for memory safety

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.error_file = project_dir / self.ERROR_LOG_FILE
        self.errors: list[dict] = []
        self._load_errors()

    def record_error(
        self,
        error_type: ErrorType,
        error_text: str,
        task_id: int = None,
        context: dict = None,
        recovered: bool = False
    ) -> None:
        """Record error occurrence."""

        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": error_type.value,
            "error_text": error_text[:500],  # Truncate long errors
            "task_id": task_id,
            "context": context or {},
            "recovered": recovered
        }

        self.errors.append(entry)

        # Bound list size
        if len(self.errors) > self.MAX_ERRORS:
            self.errors = self.errors[-self.MAX_ERRORS:]

        self._save_errors()

    def get_error_summary(self) -> dict:
        """Generate error summary report."""

        by_type = defaultdict(int)
        by_task = defaultdict(int)
        recovery_rate = {"recovered": 0, "unrecovered": 0}

        for error in self.errors:
            by_type[error["error_type"]] += 1

            if error.get("task_id"):
                by_task[error["task_id"]] += 1

            if error.get("recovered"):
                recovery_rate["recovered"] += 1
            else:
                recovery_rate["unrecovered"] += 1

        total = len(self.errors)
        return {
            "total_errors": total,
            "by_type": dict(by_type),
            "by_task": dict(by_task),
            "recovery_rate": (
                recovery_rate["recovered"] / total * 100
                if total > 0 else 0
            ),
            "most_common_type": max(by_type, key=by_type.get) if by_type else None,
            "problematic_tasks": [
                task for task, count in by_task.items() if count >= 3
            ]
        }

    def _load_errors(self) -> None:
        """Load errors from file."""
        if self.error_file.exists():
            try:
                self.errors = json.loads(self.error_file.read_text())
            except json.JSONDecodeError:
                self.errors = []

    def _save_errors(self) -> None:
        """Save errors atomically."""
        temp_file = self.error_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(self.errors, indent=2))
        temp_file.replace(self.error_file)
```

## Graceful Degradation

Handle partial failures gracefully:

```python
from enum import Enum

class DegradationLevel(Enum):
    FULL = "full"           # All features available
    REDUCED = "reduced"     # Some features disabled
    MINIMAL = "minimal"     # Core features only
    STOPPED = "stopped"     # Agent stopped

class GracefulDegradation:
    """Manage graceful degradation on failures."""

    def __init__(self):
        self.level = DegradationLevel.FULL
        self.disabled_features: set[str] = set()
        self.failure_counts: dict[str, int] = {}

    def record_feature_failure(self, feature: str) -> None:
        """Record failure for a feature."""
        self.failure_counts[feature] = self.failure_counts.get(feature, 0) + 1

        # Disable feature after repeated failures
        if self.failure_counts[feature] >= 3:
            self.disabled_features.add(feature)
            self._update_level()

    def is_feature_available(self, feature: str) -> bool:
        """Check if feature is still available."""
        return feature not in self.disabled_features

    def _update_level(self) -> None:
        """Update degradation level based on disabled features."""
        disabled_count = len(self.disabled_features)

        if disabled_count == 0:
            self.level = DegradationLevel.FULL
        elif disabled_count < 3:
            self.level = DegradationLevel.REDUCED
        elif disabled_count < 6:
            self.level = DegradationLevel.MINIMAL
        else:
            self.level = DegradationLevel.STOPPED

    def should_continue(self) -> bool:
        """Check if agent should continue running."""
        return self.level != DegradationLevel.STOPPED
```
