# Checkpointing Patterns

## Git-Based Checkpoints

Use git tags for recoverable checkpoint system:

```python
import subprocess
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class CheckpointStage(Enum):
    START = "start"
    IMPLEMENTATION = "implementation"
    PRE_VALIDATION = "pre-validation"
    VALIDATED = "validated"

@dataclass
class Checkpoint:
    task_id: int
    stage: CheckpointStage
    commit_hash: str
    description: str
    timestamp: str
    metadata: dict

class CheckpointManager:
    """Git-based checkpoint system for recovery."""

    METADATA_FILE = ".checkpoint-metadata.json"

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.metadata_file = project_dir / self.METADATA_FILE

    def save_checkpoint(
        self,
        task_id: int,
        stage: CheckpointStage,
        description: str,
        metadata: dict = None
    ) -> Checkpoint:
        """Create git checkpoint with tag."""

        # Ensure git repo exists
        self._ensure_git_repo()

        # Auto-commit if there are changes
        if self._has_uncommitted_changes():
            self._auto_commit(f"Checkpoint: {description}")

        # Get current commit hash
        commit_hash = self._get_head_commit()

        # Create tag name
        tag_name = f"checkpoint-t{task_id}-{stage.value}"

        # Delete existing tag if present (allow re-checkpointing)
        self._delete_tag_if_exists(tag_name)

        # Create annotated tag
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", description],
            cwd=self.project_dir,
            capture_output=True,
            check=True
        )

        # Record in metadata
        checkpoint = Checkpoint(
            task_id=task_id,
            stage=stage,
            commit_hash=commit_hash,
            description=description,
            timestamp=datetime.utcnow().isoformat(),
            metadata=metadata or {}
        )

        self._save_checkpoint_metadata(checkpoint)

        return checkpoint

    def rollback_to(
        self,
        task_id: int,
        stage: CheckpointStage,
        hard: bool = False
    ) -> bool:
        """Rollback to checkpoint."""
        tag_name = f"checkpoint-t{task_id}-{stage.value}"

        # Check tag exists
        result = subprocess.run(
            ["git", "tag", "-l", tag_name],
            cwd=self.project_dir,
            capture_output=True,
            text=True
        )

        if not result.stdout.strip():
            return False

        # Perform rollback
        if hard:
            # Hard reset - discards all changes
            subprocess.run(
                ["git", "reset", "--hard", tag_name],
                cwd=self.project_dir,
                check=True
            )
        else:
            # Soft checkout - preserves working directory
            subprocess.run(
                ["git", "checkout", tag_name],
                cwd=self.project_dir,
                check=True
            )

        return True

    def get_recovery_options(self, task_id: int) -> list[Checkpoint]:
        """Get available checkpoints for task."""
        checkpoints = self._load_all_checkpoints()
        return [c for c in checkpoints if c.task_id == task_id]

    def get_latest_checkpoint(self, task_id: int) -> Checkpoint | None:
        """Get most recent checkpoint for task."""
        options = self.get_recovery_options(task_id)
        if not options:
            return None
        return max(options, key=lambda c: c.timestamp)

    def _ensure_git_repo(self) -> None:
        """Initialize git repo if not present."""
        git_dir = self.project_dir / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"],
                cwd=self.project_dir,
                capture_output=True,
                check=True
            )

    def _has_uncommitted_changes(self) -> bool:
        """Check for uncommitted changes."""
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=self.project_dir,
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())

    def _auto_commit(self, message: str) -> None:
        """Auto-commit all changes."""
        subprocess.run(
            ["git", "add", "-A"],
            cwd=self.project_dir,
            capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=self.project_dir,
            capture_output=True
        )

    def _get_head_commit(self) -> str:
        """Get current HEAD commit hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()

    def _delete_tag_if_exists(self, tag_name: str) -> None:
        """Delete tag if it exists."""
        subprocess.run(
            ["git", "tag", "-d", tag_name],
            cwd=self.project_dir,
            capture_output=True
        )

    def _save_checkpoint_metadata(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to metadata file."""
        checkpoints = self._load_all_checkpoints()

        # Remove existing checkpoint for same task/stage
        checkpoints = [
            c for c in checkpoints
            if not (c.task_id == checkpoint.task_id and c.stage == checkpoint.stage)
        ]

        checkpoints.append(checkpoint)

        # Write atomically
        data = [
            {
                "task_id": c.task_id,
                "stage": c.stage.value,
                "commit_hash": c.commit_hash,
                "description": c.description,
                "timestamp": c.timestamp,
                "metadata": c.metadata
            }
            for c in checkpoints
        ]

        temp_file = self.metadata_file.with_suffix(".tmp")
        temp_file.write_text(json.dumps(data, indent=2))
        temp_file.replace(self.metadata_file)

    def _load_all_checkpoints(self) -> list[Checkpoint]:
        """Load all checkpoints from metadata."""
        if not self.metadata_file.exists():
            return []

        data = json.loads(self.metadata_file.read_text())
        return [
            Checkpoint(
                task_id=c["task_id"],
                stage=CheckpointStage(c["stage"]),
                commit_hash=c["commit_hash"],
                description=c["description"],
                timestamp=c["timestamp"],
                metadata=c.get("metadata", {})
            )
            for c in data
        ]
```

## Session Progress Tracking

Track progress notes across sessions:

```python
from pathlib import Path
from datetime import datetime

class ProgressTracker:
    """Track agent progress across sessions."""

    PROGRESS_FILE = "agent-progress.txt"

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.progress_file = project_dir / self.PROGRESS_FILE

    def log_progress(self, message: str, task_id: int = None) -> None:
        """Append timestamped progress note."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        prefix = f"[Task {task_id}] " if task_id else ""
        entry = f"[{timestamp}] {prefix}{message}\n"

        with open(self.progress_file, "a") as f:
            f.write(entry)

    def get_recent_progress(self, lines: int = 20) -> list[str]:
        """Get recent progress entries."""
        if not self.progress_file.exists():
            return []

        all_lines = self.progress_file.read_text().strip().split("\n")
        return all_lines[-lines:]

    def get_task_progress(self, task_id: int) -> list[str]:
        """Get all progress for specific task."""
        if not self.progress_file.exists():
            return []

        all_lines = self.progress_file.read_text().strip().split("\n")
        task_marker = f"[Task {task_id}]"
        return [line for line in all_lines if task_marker in line]
```

## Checkpoint-Based Recovery Flow

Standard recovery workflow:

```python
async def recover_from_failure(
    task_id: int,
    checkpoint_manager: CheckpointManager,
    task_manager: TaskStateManager,
    error: Exception
) -> bool:
    """Attempt checkpoint-based recovery."""

    # Get recovery options
    checkpoints = checkpoint_manager.get_recovery_options(task_id)

    if not checkpoints:
        # No checkpoints available
        return False

    # Determine rollback target based on error type
    if is_implementation_error(error):
        # Rollback to start of implementation
        target_stage = CheckpointStage.START
    elif is_validation_error(error):
        # Rollback to pre-validation
        target_stage = CheckpointStage.IMPLEMENTATION
    else:
        # Rollback to most recent checkpoint
        target_stage = checkpoints[-1].stage

    # Find matching checkpoint
    target = next(
        (c for c in checkpoints if c.stage == target_stage),
        checkpoints[-1]  # Fallback to most recent
    )

    # Perform rollback
    success = checkpoint_manager.rollback_to(
        task_id,
        target.stage,
        hard=True  # Discard corrupted changes
    )

    if success:
        # Reset task state
        task_manager.update_task(
            task_id,
            is_started=False,
            is_complete=False,
            is_validated=False,
            error_message=None
        )

    return success
```

## Checkpoint Cleanup

Clean old checkpoints to prevent repository bloat:

```python
def cleanup_old_checkpoints(
    checkpoint_manager: CheckpointManager,
    max_checkpoints_per_task: int = 5,
    max_age_days: int = 30
) -> int:
    """Remove old checkpoints beyond retention limits."""
    from datetime import timedelta

    all_checkpoints = checkpoint_manager._load_all_checkpoints()
    cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)

    # Group by task
    by_task: dict[int, list[Checkpoint]] = {}
    for c in all_checkpoints:
        by_task.setdefault(c.task_id, []).append(c)

    to_remove = []

    for task_id, checkpoints in by_task.items():
        # Sort by timestamp
        checkpoints.sort(key=lambda c: c.timestamp)

        # Remove old checkpoints
        for c in checkpoints:
            checkpoint_time = datetime.fromisoformat(c.timestamp)
            if checkpoint_time < cutoff_date:
                to_remove.append(c)

        # Keep only max_checkpoints_per_task most recent
        if len(checkpoints) > max_checkpoints_per_task:
            to_remove.extend(checkpoints[:-max_checkpoints_per_task])

    # Remove checkpoint tags
    removed = 0
    for c in to_remove:
        tag_name = f"checkpoint-t{c.task_id}-{c.stage.value}"
        checkpoint_manager._delete_tag_if_exists(tag_name)
        removed += 1

    return removed
```
