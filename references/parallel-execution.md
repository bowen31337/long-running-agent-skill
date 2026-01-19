# Parallel Execution Patterns

## Worker Pool Architecture

Run multiple agent workers concurrently with coordinated task assignment:

```python
import asyncio
from dataclasses import dataclass, field
from enum import Enum

@dataclass
class WorkerHealthState:
    """Track per-worker health for adaptive recovery."""
    worker_id: int
    consecutive_failures: int = 0
    total_recoveries: int = 0
    last_error_type: str | None = None
    recovery_attempts: int = 0

    MAX_CONSECUTIVE_FAILURES = 3
    MAX_RECOVERY_ATTEMPTS = 5

    def record_failure(self, error_type: str) -> None:
        self.consecutive_failures += 1
        self.last_error_type = error_type

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_error_type = None

    def record_recovery(self, success: bool) -> None:
        self.recovery_attempts += 1
        if success:
            self.total_recoveries += 1
            self.consecutive_failures = 0

    def should_attempt_recovery(self) -> bool:
        return self.recovery_attempts < self.MAX_RECOVERY_ATTEMPTS

    def is_healthy(self) -> bool:
        return self.consecutive_failures < self.MAX_CONSECUTIVE_FAILURES

class ParallelSessionManager:
    """Orchestrate multiple concurrent agent sessions."""

    def __init__(self, max_workers: int = 3):
        self.max_workers = max_workers
        self.health_states = [
            WorkerHealthState(worker_id=i) for i in range(max_workers)
        ]

    async def run_parallel_agents(
        self,
        tasks: list[dict],
        task_processor: callable
    ) -> list[dict]:
        """Execute tasks across worker pool with work-stealing."""

        # Create task queue
        task_queue = asyncio.Queue()
        for task in tasks:
            await task_queue.put(task)

        results = []
        results_lock = asyncio.Lock()

        async def worker(worker_id: int):
            health = self.health_states[worker_id]

            while True:
                try:
                    # Non-blocking get with timeout
                    task = await asyncio.wait_for(
                        task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # No more tasks
                    break

                try:
                    result = await task_processor(task, worker_id)
                    health.record_success()

                    async with results_lock:
                        results.append({"task": task, "result": result, "success": True})

                except Exception as e:
                    health.record_failure(type(e).__name__)

                    if health.should_attempt_recovery():
                        # Attempt recovery and requeue task
                        recovered = await self._attempt_recovery(health, e)
                        if recovered:
                            await task_queue.put(task)  # Requeue
                        else:
                            async with results_lock:
                                results.append({
                                    "task": task,
                                    "error": str(e),
                                    "success": False
                                })
                    else:
                        async with results_lock:
                            results.append({
                                "task": task,
                                "error": str(e),
                                "success": False,
                                "exhausted_recovery": True
                            })

                finally:
                    task_queue.task_done()

        # Launch workers
        workers = [
            asyncio.create_task(worker(i))
            for i in range(self.max_workers)
        ]

        await asyncio.gather(*workers)
        return results
```

## File-Based Locking

Prevent multiple workers from accessing the same task:

```python
import fcntl
import os
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime

class TaskLock:
    """File-based locking for parallel task safety."""

    LOCK_DIR = ".locks"
    LOCK_TIMEOUT = 300  # 5 minutes max hold

    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.lock_dir = project_dir / self.LOCK_DIR
        self.lock_dir.mkdir(exist_ok=True)

        # Write .gitignore for lock directory
        gitignore = self.lock_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n!.gitignore\n")

    def acquire(
        self,
        task_id: int,
        holder_id: str = "worker-0",
        timeout: int = 300
    ) -> bool:
        """Acquire exclusive lock with stale detection."""
        lock_file = self.lock_dir / f"task-{task_id}.lock"

        try:
            # Open or create lock file
            fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR)

            # Try non-blocking lock
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # Lock held by another process - check if stale
                os.close(fd)
                if self._is_stale_lock(lock_file):
                    self._clean_stale_lock(lock_file)
                    return self.acquire(task_id, holder_id, timeout)
                return False

            # Write lock metadata
            lock_metadata = {
                "task_id": task_id,
                "holder_id": holder_id,
                "pid": os.getpid(),
                "acquired_at": datetime.utcnow().isoformat(),
                "timeout": timeout
            }

            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, json.dumps(lock_metadata).encode())

            # Store fd for later release
            self._held_locks = getattr(self, '_held_locks', {})
            self._held_locks[task_id] = fd

            return True

        except Exception:
            return False

    def release(self, task_id: int, holder_id: str = None) -> bool:
        """Release lock if held by this holder."""
        held_locks = getattr(self, '_held_locks', {})

        if task_id not in held_locks:
            return False

        fd = held_locks[task_id]
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

            lock_file = self.lock_dir / f"task-{task_id}.lock"
            lock_file.unlink(missing_ok=True)

            del held_locks[task_id]
            return True
        except Exception:
            return False

    def _is_stale_lock(self, lock_file: Path) -> bool:
        """Check if lock is stale (holder process dead or timeout exceeded)."""
        if not lock_file.exists():
            return True

        try:
            metadata = json.loads(lock_file.read_text())

            # Check if process is alive
            pid = metadata.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)  # Signal 0 = check if process exists
                except ProcessLookupError:
                    return True  # Process dead

            # Check timeout
            acquired_at = datetime.fromisoformat(metadata.get("acquired_at", ""))
            timeout = metadata.get("timeout", self.LOCK_TIMEOUT)
            elapsed = (datetime.utcnow() - acquired_at).total_seconds()

            return elapsed > timeout

        except Exception:
            return True  # Corrupted = stale

    def _clean_stale_lock(self, lock_file: Path) -> None:
        """Remove stale lock file."""
        lock_file.unlink(missing_ok=True)

    @contextmanager
    def lock(self, task_id: int, holder_id: str = "worker-0"):
        """Context manager for lock acquisition."""
        acquired = self.acquire(task_id, holder_id=holder_id)
        try:
            yield acquired
        finally:
            if acquired:
                self.release(task_id, holder_id)

    def clean_all_stale_locks(self, max_age_seconds: int = 3600) -> int:
        """Clean all stale locks, return count cleaned."""
        cleaned = 0
        for lock_file in self.lock_dir.glob("task-*.lock"):
            if self._is_stale_lock(lock_file):
                self._clean_stale_lock(lock_file)
                cleaned += 1
        return cleaned
```

## Work-Stealing Pattern

Workers dynamically pick up available tasks:

```python
async def work_stealing_loop(
    worker_id: int,
    task_manager: TaskStateManager,
    task_lock: TaskLock,
    processor: callable
):
    """Worker loop with work-stealing."""

    while True:
        # Get next available task
        task = task_manager.get_next_pending_task()
        if task is None:
            break  # No more tasks

        task_id = task["id"]

        # Try to acquire lock (another worker may have grabbed it)
        with task_lock.lock(task_id, holder_id=f"worker-{worker_id}") as acquired:
            if not acquired:
                # Another worker got it, try next task
                continue

            # Double-check task still available (race condition guard)
            task = task_manager.get_task(task_id)
            if task.get("is_started"):
                continue

            # Mark as started and process
            task_manager.update_task(task_id, is_started=True)

            try:
                await processor(task)
                task_manager.update_task(task_id, is_complete=True)
            except Exception as e:
                task_manager.update_task(
                    task_id,
                    is_started=False,  # Allow retry
                    error_message=str(e),
                    attempts=task.get("attempts", 0) + 1
                )
```

## Adaptive Backoff

Handle rate limits and transient errors:

```python
import random
import asyncio

class AdaptiveBackoff:
    """Exponential backoff with jitter."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.consecutive_failures = 0

    def get_delay(self) -> float:
        """Calculate delay with exponential backoff + jitter."""
        delay = self.base_delay * (2 ** self.consecutive_failures)
        delay = min(delay, self.max_delay)

        # Add jitter
        jitter = delay * self.jitter_factor * random.uniform(-1, 1)
        return max(0, delay + jitter)

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0

    async def wait(self) -> None:
        """Wait for calculated delay."""
        delay = self.get_delay()
        await asyncio.sleep(delay)
```

## API Key Rotation

Rotate API keys on quota/rate limit errors:

```python
import os
from dataclasses import dataclass

@dataclass
class APICredential:
    api_key: str
    base_url: str | None = None
    model: str | None = None
    is_exhausted: bool = False
    last_error: str | None = None

class APIRotationState:
    """Manage API credential rotation."""

    def __init__(self):
        self.credentials = self._load_credentials()
        self.current_index = 0

    def _load_credentials(self) -> list[APICredential]:
        """Load numbered credentials from environment."""
        credentials = []

        # Load numbered credentials (API_KEY_1, API_KEY_2, ...)
        i = 1
        while True:
            api_key = os.environ.get(f"ANTHROPIC_API_KEY_{i}")
            if not api_key:
                break

            credentials.append(APICredential(
                api_key=api_key,
                base_url=os.environ.get(f"ANTHROPIC_BASE_URL_{i}"),
                model=os.environ.get(f"ANTHROPIC_MODEL_{i}")
            ))
            i += 1

        # Fallback to single credential
        if not credentials:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                credentials.append(APICredential(api_key=api_key))

        return credentials

    def get_current(self) -> APICredential | None:
        """Get current active credential."""
        available = [c for c in self.credentials if not c.is_exhausted]
        if not available:
            return None
        return available[self.current_index % len(available)]

    def rotate(self, error_message: str) -> bool:
        """Rotate to next credential, return False if all exhausted."""
        current = self.get_current()
        if current:
            current.last_error = error_message

            # Mark exhausted for certain errors
            if "daily quota" in error_message.lower():
                current.is_exhausted = True

        # Move to next
        self.current_index += 1

        return self.get_current() is not None

    def should_rotate(self, status_code: int, error_text: str) -> bool:
        """Determine if error warrants rotation."""
        if status_code == 429:  # Rate limit
            return True
        if status_code in (401, 403):  # Auth error
            return True
        if "quota" in error_text.lower():
            return True
        return False
```
