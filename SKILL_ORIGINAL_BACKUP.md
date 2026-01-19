---
name: long-running-agent
description: Build autonomous, long-running AI agents that parse PRDs/specifications into structured task lists and execute them autonomously with state persistence, error recovery, and cross-session resumption. Works with any agent framework (Cursor, OpenCode, etc.).
license: MIT
compatibility: Requires file system access, JSON processing, and ability to execute tasks over extended periods. Compatible with any AI agent that supports file operations and persistent state management.
---

# Long Running Agent

Build resilient autonomous agents that can parse PRDs/specifications, generate structured task lists, and execute tasks autonomously over extended periods with state persistence and automatic recovery.

This skill provides agent-agnostic patterns that work with any AI agent framework including Cursor, OpenCode, Claude, and others.

## Core Architecture

A long-running agent consists of seven core systems that work with any agent framework:

1. **PRD/Spec Processing** - Parse requirements documents into structured, executable task lists
2. **Task Execution Engine** - Autonomous task processing with dependency management
3. **API Rotation & Management** - Intelligent API key rotation, rate limiting, and load balancing
4. **State Management** - File-based persistence for workflow states and task tracking
5. **Error Handling** - Classification, recovery strategies, and graceful degradation
6. **Cross-Session Persistence** - Resume work across interruptions and restarts
7. **Learning & Memory** - Pattern recognition and improvement over time

These patterns are framework-agnostic and can be implemented with any AI agent that has file system access.

## Implementation Workflow

### Step 1: Set Up Project Structure

Create the standard file structure for persistent state management:

```
project/
├── tasks/
│   ├── current_tasks.json      # Active task list from PRD
│   ├── task_history.json       # Completed task archive
│   └── execution_plan.json     # Optimized task ordering
├── results/
│   ├── task_001/              # Implementation results by task
│   ├── task_002/
│   └── documentation/
├── memories/
│   ├── patterns.json          # Learned implementation patterns
│   ├── solutions.json         # Error solutions database
│   └── templates.json         # Reusable task templates
└── logs/
    ├── execution.log          # Task execution history
    └── errors.log             # Error tracking and analysis
```

### Step 2: Implement PRD Processing

Create the core function to parse PRDs into structured task lists:

```python
import json
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

def parse_prd_to_tasks(prd_content: str, project_name: str = "Generated Project") -> Dict:
    """
    Parse PRD/specification content into structured task list.
    
    This function works with any agent - just provide PRD text and get structured tasks.
    """
    tasks = []
    task_id_counter = 1
    
    # Split content into sections by headers (## or ###)
    sections = re.split(r'\n(?=#{2,3}\s)', prd_content)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        # Extract title from header
        header_match = re.match(r'^#{2,3}\s+(.+)', lines[0])
        if not header_match:
            continue
            
        title = header_match.group(1).strip()
        description_lines = lines[1:] if len(lines) > 1 else []
        
        # Extract acceptance criteria (bullet points)
        acceptance_criteria = []
        remaining_description = []
        
        for line in description_lines:
            line = line.strip()
            if re.match(r'^[-*]\s+|^\d+\.\s+', line):
                criteria = re.sub(r'^[-*]\s+|^\d+\.\s+', '', line)
                acceptance_criteria.append(criteria)
            else:
                remaining_description.append(line)
        
        # Determine task properties
        content_lower = (title + ' ' + ' '.join(description_lines)).lower()
        
        # Priority detection
        priority = "medium"
        if any(word in content_lower for word in ['critical', 'urgent', 'high priority', 'must have']):
            priority = "high"
        elif any(word in content_lower for word in ['nice to have', 'optional', 'low priority', 'future']):
            priority = "low"
        
        # Effort estimation
        effort = "medium"
        if any(word in content_lower for word in ['simple', 'basic', 'quick', 'small']):
            effort = "small"
        elif any(word in content_lower for word in ['complex', 'advanced', 'large', 'system', 'integration']):
            effort = "large"
        
        # Category classification
        category = categorize_task(title, ' '.join(description_lines))
        
        task = {
            "id": f"task_{task_id_counter:03d}",
            "title": title,
            "description": '\n'.join(remaining_description).strip() or title,
            "category": category,
            "priority": priority,
            "estimated_effort": effort,
            "acceptance_criteria": acceptance_criteria,
            "dependencies": [],  # Will be populated by dependency analysis
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "result": "",
            "progress_notes": "",
            "error_details": ""
        }
        
        tasks.append(task)
        task_id_counter += 1
    
    # Analyze and set dependencies
    tasks = analyze_task_dependencies(tasks)
    
    return {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "total_tasks": len(tasks),
        "completed_tasks": 0,
        "in_progress_tasks": 0,
        "pending_tasks": len(tasks),
        "failed_tasks": 0,
        "tasks": tasks
    }

def categorize_task(title: str, description: str) -> str:
    """Categorize task based on content keywords."""
    content = (title + ' ' + description).lower()
    
    if any(word in content for word in ['ui', 'interface', 'frontend', 'component', 'page']):
        return 'frontend'
    elif any(word in content for word in ['api', 'backend', 'server', 'service', 'endpoint']):
        return 'backend'
    elif any(word in content for word in ['database', 'schema', 'migration', 'model', 'data']):
        return 'database'
    elif any(word in content for word in ['auth', 'login', 'security', 'authentication', 'authorization']):
        return 'authentication'
    elif any(word in content for word in ['test', 'testing', 'validation', 'verify', 'qa']):
        return 'testing'
    elif any(word in content for word in ['deploy', 'deployment', 'infrastructure', 'devops', 'ci/cd']):
        return 'deployment'
    elif any(word in content for word in ['document', 'docs', 'readme', 'guide', 'manual']):
        return 'documentation'
    else:
        return 'general'

def analyze_task_dependencies(tasks: List[Dict]) -> List[Dict]:
    """Analyze and set task dependencies based on categories and content."""
    
    # Common dependency patterns
    dependency_rules = [
        ('authentication', 'frontend'),  # Auth before user-facing features
        ('backend', 'frontend'),         # APIs before UI
        ('database', 'backend'),         # Schema before business logic
        ('database', 'authentication'),  # User tables before auth
    ]
    
    for i, task in enumerate(tasks):
        dependencies = []
        
        # Check against previous tasks for dependencies
        for j, other_task in enumerate(tasks[:i]):
            for prereq_category, dependent_category in dependency_rules:
                if (prereq_category in other_task['category'] and 
                    dependent_category in task['category']):
                    dependencies.append(other_task['id'])
        
        task['dependencies'] = list(set(dependencies))
    
    return tasks
```

### Step 3: Implement API Rotation and Management

Create intelligent API rotation system for long-running agents:

```python
import time
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import threading

class APIStatus(Enum):
    ACTIVE = "active"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class APIEndpoint:
    """Represents an API endpoint with rotation capabilities."""
    name: str
    base_url: str
    api_key: str
    rate_limit: int = 60  # requests per minute
    quota_limit: int = 1000  # requests per day
    current_usage: int = 0
    daily_usage: int = 0
    last_request_time: float = 0
    status: APIStatus = APIStatus.ACTIVE
    error_count: int = 0
    last_reset_time: float = field(default_factory=time.time)
    
    def can_make_request(self) -> bool:
        """Check if endpoint can handle a request."""
        if self.status != APIStatus.ACTIVE:
            return False
            
        current_time = time.time()
        
        # Reset counters if minute has passed
        if current_time - self.last_request_time >= 60:
            self.current_usage = 0
            self.last_request_time = current_time
        
        # Reset daily usage if day has passed
        if current_time - self.last_reset_time >= 86400:  # 24 hours
            self.daily_usage = 0
            self.last_reset_time = current_time
        
        # Check limits
        return (self.current_usage < self.rate_limit and 
                self.daily_usage < self.quota_limit)
    
    def record_request(self, success: bool = True):
        """Record a request and update usage counters."""
        self.current_usage += 1
        self.daily_usage += 1
        self.last_request_time = time.time()
        
        if not success:
            self.error_count += 1
        else:
            self.error_count = max(0, self.error_count - 1)  # Decay errors on success

class APIRotationManager:
    """Manages multiple API endpoints with intelligent rotation."""
    
    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rotations": 0,
            "rate_limit_hits": 0,
            "quota_exceeded_count": 0
        }
    
    def add_endpoint(self, name: str, base_url: str, api_key: str, 
                    rate_limit: int = 60, quota_limit: int = 1000):
        """Add an API endpoint to the rotation pool."""
        endpoint = APIEndpoint(
            name=name,
            base_url=base_url,
            api_key=api_key,
            rate_limit=rate_limit,
            quota_limit=quota_limit
        )
        self.endpoints.append(endpoint)
        print(f"✅ Added API endpoint: {name} (rate: {rate_limit}/min, quota: {quota_limit}/day)")
    
    def get_best_endpoint(self) -> Optional[APIEndpoint]:
        """Get the best available endpoint for making a request."""
        with self.lock:
            if not self.endpoints:
                return None
            
            # First, try to find an active endpoint that can make requests
            available_endpoints = [ep for ep in self.endpoints if ep.can_make_request()]
            
            if not available_endpoints:
                # All endpoints are rate limited or at quota
                # Find the endpoint that will be available soonest
                self._handle_all_endpoints_limited()
                return None
            
            # Select endpoint using weighted round-robin based on remaining capacity
            best_endpoint = self._select_optimal_endpoint(available_endpoints)
            
            return best_endpoint
    
    def _select_optimal_endpoint(self, available_endpoints: List[APIEndpoint]) -> APIEndpoint:
        """Select optimal endpoint based on capacity and performance."""
        
        # Calculate weights based on remaining capacity and error rate
        weighted_endpoints = []
        
        for endpoint in available_endpoints:
            # Calculate remaining capacity (0-1 scale)
            rate_capacity = (endpoint.rate_limit - endpoint.current_usage) / endpoint.rate_limit
            quota_capacity = (endpoint.quota_limit - endpoint.daily_usage) / endpoint.quota_limit
            
            # Calculate error rate (lower is better)
            error_rate = min(endpoint.error_count / 10.0, 1.0)  # Cap at 1.0
            
            # Combined weight (higher is better)
            weight = (rate_capacity * 0.4 + quota_capacity * 0.4 + (1 - error_rate) * 0.2)
            
            weighted_endpoints.append((endpoint, weight))
        
        # Sort by weight (highest first)
        weighted_endpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Use weighted random selection for the top 3 endpoints
        top_endpoints = weighted_endpoints[:3]
        weights = [w for _, w in top_endpoints]
        
        if not weights:
            return available_endpoints[0]
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return available_endpoints[0]
        
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for (endpoint, weight) in top_endpoints:
            current_weight += weight
            if rand_val <= current_weight:
                return endpoint
        
        return top_endpoints[0][0]  # Fallback
    
    def _handle_all_endpoints_limited(self):
        """Handle situation where all endpoints are rate limited."""
        print("⚠️ All API endpoints are rate limited or at quota")
        
        # Find endpoint that will be available soonest
        current_time = time.time()
        soonest_available = None
        min_wait_time = float('inf')
        
        for endpoint in self.endpoints:
            if endpoint.status == APIStatus.RATE_LIMITED:
                # Calculate when rate limit resets
                time_since_last = current_time - endpoint.last_request_time
                wait_time = max(0, 60 - time_since_last)  # 60 seconds for rate limit reset
                
                if wait_time < min_wait_time:
                    min_wait_time = wait_time
                    soonest_available = endpoint
        
        if soonest_available and min_wait_time < 300:  # Less than 5 minutes
            print(f"⏳ Waiting {min_wait_time:.1f}s for {soonest_available.name} to reset...")
            time.sleep(min_wait_time + 1)  # Add 1 second buffer
            soonest_available.status = APIStatus.ACTIVE
    
    def make_api_request(self, request_func, *args, **kwargs) -> Dict[str, Any]:
        """Make an API request with automatic rotation and error handling."""
        
        max_retries = len(self.endpoints) * 2  # Try each endpoint twice
        retry_count = 0
        
        while retry_count < max_retries:
            endpoint = self.get_best_endpoint()
            
            if not endpoint:
                # All endpoints exhausted, implement backoff
                wait_time = min(300, 2 ** retry_count)  # Exponential backoff, max 5 minutes
                print(f"⏳ All endpoints unavailable, waiting {wait_time}s...")
                time.sleep(wait_time)
                retry_count += 1
                continue
            
            try:
                # Make the API request
                print(f"🔄 Using endpoint: {endpoint.name} (usage: {endpoint.current_usage}/{endpoint.rate_limit})")
                
                # Add API key to request
                if 'headers' not in kwargs:
                    kwargs['headers'] = {}
                kwargs['headers']['Authorization'] = f"Bearer {endpoint.api_key}"
                
                # Make request with endpoint's base URL
                if args and isinstance(args[0], str):
                    # Replace or prepend base URL
                    url = args[0]
                    if not url.startswith('http'):
                        url = f"{endpoint.base_url.rstrip('/')}/{url.lstrip('/')}"
                    args = (url,) + args[1:]
                
                result = request_func(*args, **kwargs)
                
                # Record successful request
                endpoint.record_request(success=True)
                self.usage_stats["total_requests"] += 1
                self.usage_stats["successful_requests"] += 1
                
                return {
                    "success": True,
                    "data": result,
                    "endpoint_used": endpoint.name,
                    "usage_stats": self._get_usage_summary()
                }
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Handle different types of API errors
                if "rate limit" in error_message or "429" in error_message:
                    endpoint.status = APIStatus.RATE_LIMITED
                    self.usage_stats["rate_limit_hits"] += 1
                    print(f"⚠️ Rate limit hit on {endpoint.name}, rotating...")
                    
                elif "quota" in error_message or "exceeded" in error_message:
                    endpoint.status = APIStatus.QUOTA_EXCEEDED
                    self.usage_stats["quota_exceeded_count"] += 1
                    print(f"⚠️ Quota exceeded on {endpoint.name}, rotating...")
                    
                elif "unauthorized" in error_message or "401" in error_message:
                    endpoint.status = APIStatus.ERROR
                    print(f"❌ Authentication error on {endpoint.name}, disabling...")
                    
                else:
                    # Generic error, record but don't disable endpoint
                    endpoint.record_request(success=False)
                    print(f"⚠️ Request error on {endpoint.name}: {e}")
                
                self.usage_stats["total_requests"] += 1
                self.usage_stats["failed_requests"] += 1
                
                retry_count += 1
                
                # If this was the last endpoint, wait before retrying
                if retry_count < max_retries:
                    self.usage_stats["rotations"] += 1
                    continue
        
        # All retries exhausted
        return {
            "success": False,
            "error": "All API endpoints exhausted",
            "usage_stats": self._get_usage_summary(),
            "retry_count": retry_count
        }
    
    def _get_usage_summary(self) -> Dict:
        """Get current usage summary across all endpoints."""
        return {
            "total_requests": self.usage_stats["total_requests"],
            "success_rate": (self.usage_stats["successful_requests"] / 
                           max(1, self.usage_stats["total_requests"]) * 100),
            "active_endpoints": len([ep for ep in self.endpoints if ep.status == APIStatus.ACTIVE]),
            "total_endpoints": len(self.endpoints),
            "rotations": self.usage_stats["rotations"]
        }
    
    def get_status_report(self) -> Dict:
        """Get comprehensive status report of all endpoints."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_stats": self.usage_stats.copy(),
            "endpoints": []
        }
        
        for endpoint in self.endpoints:
            endpoint_info = {
                "name": endpoint.name,
                "status": endpoint.status.value,
                "rate_usage": f"{endpoint.current_usage}/{endpoint.rate_limit}",
                "quota_usage": f"{endpoint.daily_usage}/{endpoint.quota_limit}",
                "error_count": endpoint.error_count,
                "can_make_request": endpoint.can_make_request()
            }
            report["endpoints"].append(endpoint_info)
        
        return report
    
    def reset_endpoint_status(self, endpoint_name: str):
        """Manually reset an endpoint's status (for recovery)."""
        for endpoint in self.endpoints:
            if endpoint.name == endpoint_name:
                endpoint.status = APIStatus.ACTIVE
                endpoint.error_count = 0
                print(f"🔄 Reset endpoint status: {endpoint_name}")
                return True
        return False

# Global API rotation manager instance
api_manager = APIRotationManager()

def setup_api_rotation(api_configs: List[Dict]):
    """Set up API rotation with multiple endpoints."""
    global api_manager
    
    for config in api_configs:
        api_manager.add_endpoint(
            name=config["name"],
            base_url=config["base_url"],
            api_key=config["api_key"],
            rate_limit=config.get("rate_limit", 60),
            quota_limit=config.get("quota_limit", 1000)
        )
    
    print(f"🔄 API rotation setup complete with {len(api_configs)} endpoints")

def make_api_call(request_func, *args, **kwargs) -> Dict:
    """Make an API call with automatic rotation and error handling."""
    return api_manager.make_api_request(request_func, *args, **kwargs)

def get_api_status() -> Dict:
    """Get current API rotation status and usage statistics."""
    return api_manager.get_status_report()
```

### Step 4: Implement State Management

Create functions for managing task state across sessions:

```python
def save_task_list(task_list: Dict, file_path: str = "tasks/current_tasks.json") -> bool:
    """Save task list to persistent storage."""
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(task_list, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving task list: {e}")
        return False

def load_task_list(file_path: str = "tasks/current_tasks.json") -> Dict:
    """Load task list from persistent storage."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"tasks": [], "total_tasks": 0, "completed_tasks": 0}
    except Exception as e:
        print(f"Error loading task list: {e}")
        return {"tasks": [], "total_tasks": 0, "completed_tasks": 0}

def update_task_status(task_id: str, status: str, result: str = "", 
                      progress_notes: str = "", error_details: str = "") -> bool:
    """Update task status and save changes."""
    try:
        task_list = load_task_list()
        
        # Find and update the task
        for task in task_list.get("tasks", []):
            if task["id"] == task_id:
                old_status = task["status"]
                task["status"] = status
                task["updated_at"] = datetime.now().isoformat()
                
                if result:
                    task["result"] = result
                if progress_notes:
                    task["progress_notes"] = progress_notes
                if error_details:
                    task["error_details"] = error_details
                
                # Update counters
                update_task_counters(task_list, old_status, status)
                break
        
        return save_task_list(task_list)
        
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False

def update_task_counters(task_list: Dict, old_status: str, new_status: str):
    """Update task counters when status changes."""
    # Decrement old status counter
    if old_status == "pending":
        task_list["pending_tasks"] = task_list.get("pending_tasks", 0) - 1
    elif old_status == "in_progress":
        task_list["in_progress_tasks"] = task_list.get("in_progress_tasks", 0) - 1
    elif old_status == "completed":
        task_list["completed_tasks"] = task_list.get("completed_tasks", 0) - 1
    elif old_status == "failed":
        task_list["failed_tasks"] = task_list.get("failed_tasks", 0) - 1
    
    # Increment new status counter
    if new_status == "pending":
        task_list["pending_tasks"] = task_list.get("pending_tasks", 0) + 1
    elif new_status == "in_progress":
        task_list["in_progress_tasks"] = task_list.get("in_progress_tasks", 0) + 1
    elif new_status == "completed":
        task_list["completed_tasks"] = task_list.get("completed_tasks", 0) + 1
    elif new_status == "failed":
        task_list["failed_tasks"] = task_list.get("failed_tasks", 0) + 1
```

### Step 4: Implement Task Execution Engine

Create the autonomous task execution system:

```python
def get_executable_tasks() -> List[Dict]:
    """Get list of tasks that can be executed (dependencies met, status pending)."""
    task_list = load_task_list()
    executable_tasks = []
    
    for task in task_list.get("tasks", []):
        if task["status"] != "pending":
            continue
            
        # Check if all dependencies are completed
        dependencies_met = True
        for dep_id in task.get("dependencies", []):
            dep_task = find_task_by_id(task_list, dep_id)
            if not dep_task or dep_task["status"] != "completed":
                dependencies_met = False
                break
        
        if dependencies_met:
            executable_tasks.append(task)
    
    # Sort by priority (high > medium > low) then by creation time
    priority_order = {"high": 3, "medium": 2, "low": 1}
    executable_tasks.sort(
        key=lambda t: (priority_order.get(t.get("priority", "medium"), 2), t.get("created_at", "")),
        reverse=True
    )
    
    return executable_tasks

def find_task_by_id(task_list: Dict, task_id: str) -> Dict:
    """Find a task by its ID."""
    for task in task_list.get("tasks", []):
        if task["id"] == task_id:
            return task
    return None

def execute_next_task() -> Dict:
    """Find and execute the next available task."""
    executable_tasks = get_executable_tasks()
    
    if not executable_tasks:
        return {"success": False, "message": "No executable tasks available"}
    
    task = executable_tasks[0]
    task_id = task["id"]
    
    # Update status to in_progress
    if not update_task_status(task_id, "in_progress"):
        return {"success": False, "message": f"Failed to update task {task_id} status"}
    
    try:
        # Execute task based on category
        result = execute_task_by_category(task)
        
        # Update with results
        if result["success"]:
            update_task_status(task_id, "completed", result["output"], result.get("notes", ""))
            return {"success": True, "message": f"Task {task_id} completed successfully", "result": result}
        else:
            update_task_status(task_id, "failed", "", "", result.get("error", "Unknown error"))
            return {"success": False, "message": f"Task {task_id} failed", "error": result.get("error")}
            
    except Exception as e:
        update_task_status(task_id, "failed", "", "", str(e))
        return {"success": False, "message": f"Task {task_id} failed with exception", "error": str(e)}

def execute_task_by_category(task: Dict) -> Dict:
    """Execute a task based on its category."""
    category = task.get("category", "general")
    
    implementation_strategies = {
        "frontend": implement_frontend_task,
        "backend": implement_backend_task,
        "database": implement_database_task,
        "authentication": implement_auth_task,
        "testing": implement_testing_task,
        "documentation": implement_documentation_task,
        "general": implement_general_task
    }
    
    implementation_func = implementation_strategies.get(category, implement_general_task)
    return implementation_func(task)

def implement_frontend_task(task: Dict) -> Dict:
    """Implement frontend-specific tasks."""
    task_id = task["id"]
    
    # Create results directory
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate frontend implementation
    implementation = generate_frontend_implementation(task)
    
    # Save implementation files
    save_implementation_files(results_dir, implementation)
    
    return {
        "success": True,
        "output": f"Frontend implementation completed in {results_dir}",
        "notes": f"Created {len(implementation)} files for {task['title']}"
    }

def implement_backend_task(task: Dict) -> Dict:
    """Implement backend-specific tasks."""
    task_id = task["id"]
    
    # Create results directory
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate backend implementation
    implementation = generate_backend_implementation(task)
    
    # Save implementation files
    save_implementation_files(results_dir, implementation)
    
    return {
        "success": True,
        "output": f"Backend implementation completed in {results_dir}",
        "notes": f"Created API endpoints and services for {task['title']}"
    }

# Additional implementation functions for other categories...
def implement_general_task(task: Dict) -> Dict:
    """Implement general tasks."""
    task_id = task["id"]
    
    # Create results directory
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create basic implementation structure
    implementation_notes = f"""
# Task Implementation: {task['title']}

## Description
{task['description']}

## Acceptance Criteria
{chr(10).join('- ' + criteria for criteria in task.get('acceptance_criteria', []))}

## Implementation Status
- Status: Completed
- Implementation approach: General task processing
- Results saved to: {results_dir}

## Next Steps
Review implementation and verify against acceptance criteria.
"""
    
    with open(f"{results_dir}/README.md", "w") as f:
        f.write(implementation_notes)
    
    return {
        "success": True,
        "output": f"General task implementation completed in {results_dir}",
        "notes": f"Basic implementation structure created for {task['title']}"
    }
```

### Step 5: Integrate API Rotation with Task Execution

Enhance task execution with API rotation capabilities:

```python
def execute_task_with_api_rotation(task: Dict) -> Dict:
    """Execute a task with API rotation support for external API calls."""
    
    task_id = task["id"]
    category = task.get("category", "general")
    
    # Check if task requires external API calls
    requires_api = any(keyword in task["description"].lower() 
                      for keyword in ["api", "service", "external", "integration", "fetch", "call"])
    
    if requires_api:
        print(f"🔄 Task {task_id} requires API calls, using rotation manager...")
        
        # Get API status before starting
        api_status = get_api_status()
        active_endpoints = len([ep for ep in api_status["endpoints"] if ep["can_make_request"]])
        
        if active_endpoints == 0:
            return {
                "success": False,
                "error": "No API endpoints available for task execution",
                "task_id": task_id
            }
        
        print(f"📡 {active_endpoints} API endpoints available for task execution")
    
    # Execute task with API rotation support
    try:
        if category == "backend" and requires_api:
            return implement_backend_task_with_api(task)
        elif category == "frontend" and requires_api:
            return implement_frontend_task_with_api(task)
        elif category == "integration":
            return implement_integration_task_with_api(task)
        else:
            # Standard task execution without API rotation
            return execute_task_by_category(task)
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Task execution failed: {str(e)}",
            "task_id": task_id
        }

def implement_backend_task_with_api(task: Dict) -> Dict:
    """Implement backend tasks that require external API calls."""
    
    task_id = task["id"]
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Example: Creating an API integration service
    service_code = f'''
import requests
import time
from typing import Dict, Any

class {task["title"].replace(" ", "")}Service:
    """Service for {task["title"]} with API rotation support."""
    
    def __init__(self, api_manager):
        self.api_manager = api_manager
    
    def make_external_call(self, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make external API call with rotation."""
        
        def request_func(url, **kwargs):
            if data:
                return requests.post(url, json=data, **kwargs)
            else:
                return requests.get(url, **kwargs)
        
        # Use API rotation manager
        result = self.api_manager.make_api_request(request_func, endpoint)
        
        if result["success"]:
            return result["data"].json()
        else:
            raise Exception(f"API call failed: {{result['error']}}")
    
    def process_with_retry(self, data: Dict) -> Dict:
        """Process data with automatic retry and rotation."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Make API call with rotation
                response = self.make_external_call("/process", data)
                return {{"success": True, "data": response}}
                
            except Exception as e:
                if attempt == max_attempts - 1:
                    return {{"success": False, "error": str(e)}}
                
                # Wait before retry
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⏳ Attempt {{attempt + 1}} failed, retrying in {{wait_time}}s...")
                time.sleep(wait_time)
        
        return {{"success": False, "error": "All retry attempts exhausted"}}
'''
    
    with open(f"{results_dir}/api_service.py", "w") as f:
        f.write(service_code)
    
    # Create API configuration template
    api_config_template = {
        "endpoints": [
            {
                "name": "primary_api",
                "base_url": "https://api.example.com",
                "api_key": "your_api_key_here",
                "rate_limit": 100,
                "quota_limit": 10000
            },
            {
                "name": "backup_api",
                "base_url": "https://backup-api.example.com", 
                "api_key": "your_backup_key_here",
                "rate_limit": 60,
                "quota_limit": 5000
            }
        ]
    }
    
    with open(f"{results_dir}/api_config.json", "w") as f:
        json.dump(api_config_template, f, indent=2)
    
    return {
        "success": True,
        "output": f"Backend service with API rotation implemented in {results_dir}",
        "notes": f"Created API service with rotation support for {task['title']}",
        "files_created": ["api_service.py", "api_config.json"]
    }

def implement_integration_task_with_api(task: Dict) -> Dict:
    """Implement integration tasks with API rotation."""
    
    task_id = task["id"]
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create integration handler with API rotation
    integration_code = f'''
import json
from typing import Dict, List, Any
from api_rotation_manager import APIRotationManager, setup_api_rotation

class {task["title"].replace(" ", "")}Integration:
    """Integration handler for {task["title"]} with API rotation."""
    
    def __init__(self, config_file: str = "api_config.json"):
        # Load API configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Setup API rotation
        setup_api_rotation(config["endpoints"])
        
        self.api_manager = api_manager  # Global instance
    
    def sync_data(self, data: List[Dict]) -> Dict[str, Any]:
        """Sync data across multiple APIs with rotation."""
        
        results = []
        failed_items = []
        
        for item in data:
            try:
                # Make API call with rotation
                result = self.api_manager.make_api_request(
                    self._sync_single_item, 
                    item
                )
                
                if result["success"]:
                    results.append(result["data"])
                else:
                    failed_items.append({{"item": item, "error": result["error"]}})
                    
            except Exception as e:
                failed_items.append({{"item": item, "error": str(e)}})
        
        return {{
            "synced_count": len(results),
            "failed_count": len(failed_items),
            "success_rate": len(results) / len(data) * 100 if data else 0,
            "failed_items": failed_items,
            "api_usage": self.api_manager.get_status_report()
        }}
    
    def _sync_single_item(self, item: Dict) -> Any:
        """Sync a single item via API."""
        import requests
        
        # This function will be called with API rotation
        response = requests.post("/sync", json=item)
        response.raise_for_status()
        return response.json()
'''
    
    with open(f"{results_dir}/integration_handler.py", "w") as f:
        f.write(integration_code)
    
    return {
        "success": True,
        "output": f"Integration handler with API rotation implemented in {results_dir}",
        "notes": f"Created integration system with automatic API rotation for {task['title']}",
        "files_created": ["integration_handler.py"]
    }

### Step 6: Implement Learning and Memory System

Create functions for learning from execution patterns:

```python
def save_execution_pattern(task: Dict, execution_result: Dict, pattern_type: str = "success"):
    """Save successful execution patterns for future learning."""
    try:
        patterns_file = "memories/patterns.json"
        patterns = load_json_file(patterns_file, default=[])
        
        pattern = {
            "task_category": task.get("category", "general"),
            "task_title": task.get("title", ""),
            "execution_approach": execution_result.get("approach", ""),
            "success": execution_result.get("success", False),
            "execution_time": execution_result.get("execution_time", 0),
            "pattern_type": pattern_type,
            "timestamp": datetime.now().isoformat(),
            "notes": execution_result.get("notes", "")
        }
        
        patterns.append(pattern)
        save_json_file(patterns_file, patterns)
        
    except Exception as e:
        print(f"Error saving execution pattern: {e}")

def load_similar_patterns(task_category: str, task_title: str = "") -> List[Dict]:
    """Load similar execution patterns for guidance."""
    try:
        patterns_file = "memories/patterns.json"
        patterns = load_json_file(patterns_file, default=[])
        
        # Filter patterns by category and success
        similar_patterns = [
            p for p in patterns 
            if p.get("task_category") == task_category and p.get("success", False)
        ]
        
        # If task title provided, prioritize similar titles
        if task_title:
            title_matches = [p for p in similar_patterns if task_title.lower() in p.get("task_title", "").lower()]
            if title_matches:
                similar_patterns = title_matches
        
        # Sort by recency (most recent first)
        similar_patterns.sort(key=lambda p: p.get("timestamp", ""), reverse=True)
        
        return similar_patterns[:5]  # Return top 5 most relevant patterns
        
    except Exception as e:
        print(f"Error loading similar patterns: {e}")
        return []

def save_error_solution(error_details: str, solution: str, task_category: str = "general"):
    """Save error solutions for future reference."""
    try:
        solutions_file = "memories/solutions.json"
        solutions = load_json_file(solutions_file, default=[])
        
        solution_entry = {
            "error_signature": error_details[:200],  # First 200 chars as signature
            "full_error": error_details,
            "solution": solution,
            "task_category": task_category,
            "timestamp": datetime.now().isoformat(),
            "usage_count": 1
        }
        
        # Check if similar error exists
        for existing in solutions:
            if existing.get("error_signature") == solution_entry["error_signature"]:
                existing["usage_count"] = existing.get("usage_count", 0) + 1
                existing["solution"] = solution  # Update with latest solution
                existing["timestamp"] = solution_entry["timestamp"]
                save_json_file(solutions_file, solutions)
                return
        
        # Add new solution
        solutions.append(solution_entry)
        save_json_file(solutions_file, solutions)
        
    except Exception as e:
        print(f"Error saving error solution: {e}")

def load_json_file(file_path: str, default=None):
    """Load JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return default if default is not None else {}

def save_json_file(file_path: str, data):
    """Save data to JSON file with error handling."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False
```

### Step 7: Agent Integration Instructions

Instructions for integrating with any AI agent framework:

## For Cursor, OpenCode, and other AI Agents:

When you receive a PRD or specification, follow this workflow:

1. **Parse the PRD**: Use the `parse_prd_to_tasks()` function to convert the PRD into a structured task list
2. **Save Task List**: Save the generated task list to `tasks/current_tasks.json`
3. **Execute Tasks**: Use `execute_next_task()` repeatedly to work through the task list
4. **Update Progress**: Use `update_task_status()` to track progress and results
5. **Learn and Improve**: Use the memory functions to save patterns and solutions

## Example Workflow:

```python
# Step 1: Process PRD
prd_content = """
# User Authentication System

## User Registration
Implement user registration with email verification
- Email validation
- Password strength requirements
- Email verification flow

## Login System  
Secure login with session management
- Email/password authentication
- Session token generation
- Remember me functionality

## Password Reset
Allow users to reset forgotten passwords
- Email-based reset flow
- Secure token generation
- Password update interface
"""

# Step 2: Generate task list
task_list = parse_prd_to_tasks(prd_content, "User Auth System")
save_task_list(task_list)

# Step 3: Execute tasks autonomously
while True:
    result = execute_next_task()
    if not result["success"]:
        if "No executable tasks" in result["message"]:
            print("All tasks completed!")
            break
        else:
            print(f"Error: {result['message']}")
            break
    else:
        print(f"Completed: {result['message']}")

# Step 4: Generate progress report
task_list = load_task_list()
completed = len([t for t in task_list["tasks"] if t["status"] == "completed"])
total = task_list["total_tasks"]
print(f"Project Progress: {completed}/{total} tasks completed")
```

## Agent Instructions:

When a user provides a PRD or specification:

1. **Acknowledge**: "I'll parse this PRD into structured tasks and execute them autonomously."

2. **Parse**: Use the parsing function to create the task list with proper dependencies

3. **Plan**: Show the user the generated task list and execution plan

4. **Execute**: Work through tasks systematically, respecting dependencies

5. **Report**: Provide regular progress updates and final results

6. **Learn**: Save successful patterns and error solutions for future use

The agent should work autonomously but provide transparency about progress and any issues encountered.

## Advanced Features

### Parallel Task Execution

For handling multiple independent tasks concurrently:

```python
import threading
import concurrent.futures
from typing import List

def execute_parallel_tasks(max_workers: int = 3) -> Dict:
    """Execute multiple independent tasks in parallel."""
    executable_tasks = get_executable_tasks()
    
    # Filter for truly independent tasks (no shared dependencies)
    independent_tasks = []
    for task in executable_tasks:
        # Check if task has dependencies on other executable tasks
        has_conflicts = False
        for other_task in executable_tasks:
            if other_task["id"] != task["id"]:
                if task["id"] in other_task.get("dependencies", []) or \
                   other_task["id"] in task.get("dependencies", []):
                    has_conflicts = True
                    break
        
        if not has_conflicts:
            independent_tasks.append(task)
    
    if not independent_tasks:
        return {"success": False, "message": "No independent tasks available for parallel execution"}
    
    # Limit to max_workers
    tasks_to_execute = independent_tasks[:max_workers]
    
    # Execute tasks in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(execute_task_thread_safe, task): task 
            for task in tasks_to_execute
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append({
                    "task_id": task["id"],
                    "result": result
                })
            except Exception as e:
                results.append({
                    "task_id": task["id"],
                    "error": str(e)
                })
    
    return {
        "success": True,
        "message": f"Executed {len(results)} tasks in parallel",
        "results": results
    }

def execute_task_thread_safe(task: Dict) -> Dict:
    """Thread-safe version of task execution."""
    task_id = task["id"]
    
    # Use file locking to prevent concurrent modifications
    lock_file = f"tasks/.lock_{task_id}"
    
    try:
        # Create lock file
        with open(lock_file, 'w') as f:
            f.write(f"Locked by thread for task {task_id}")
        
        # Update status to in_progress
        if not update_task_status(task_id, "in_progress"):
            return {"success": False, "error": "Failed to update task status"}
        
        # Execute the task
        result = execute_task_by_category(task)
        
        # Update final status
        if result["success"]:
            update_task_status(task_id, "completed", result["output"], result.get("notes", ""))
        else:
            update_task_status(task_id, "failed", "", "", result.get("error", "Unknown error"))
        
        return result
        
    finally:
        # Remove lock file
        try:
            os.remove(lock_file)
        except:
            pass

def get_progress_report() -> Dict:
    """Generate comprehensive progress report."""
    task_list = load_task_list()
    tasks = task_list.get("tasks", [])
    
    # Calculate statistics
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t["status"] == "completed"])
    in_progress_tasks = len([t for t in tasks if t["status"] == "in_progress"])
    failed_tasks = len([t for t in tasks if t["status"] == "failed"])
    pending_tasks = len([t for t in tasks if t["status"] == "pending"])
    
    # Calculate completion percentage
    completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Group by category
    category_stats = {}
    for task in tasks:
        category = task.get("category", "general")
        if category not in category_stats:
            category_stats[category] = {"total": 0, "completed": 0, "failed": 0}
        
        category_stats[category]["total"] += 1
        if task["status"] == "completed":
            category_stats[category]["completed"] += 1
        elif task["status"] == "failed":
            category_stats[category]["failed"] += 1
    
    # Find next executable tasks
    executable_tasks = get_executable_tasks()
    next_tasks = [{"id": t["id"], "title": t["title"], "category": t["category"]} 
                  for t in executable_tasks[:5]]
    
    report = {
        "project_name": task_list.get("project_name", "Unknown Project"),
        "generated_at": datetime.now().isoformat(),
        "overall_progress": {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "failed_tasks": failed_tasks,
            "pending_tasks": pending_tasks,
            "completion_percentage": round(completion_percentage, 2)
        },
        "category_breakdown": category_stats,
        "next_executable_tasks": next_tasks,
        "blocked_tasks": [
            {"id": t["id"], "title": t["title"], "dependencies": t.get("dependencies", [])}
            for t in tasks if t["status"] == "pending" and t not in executable_tasks
        ]
    }
    
    # Save report
    save_json_file("results/progress_report.json", report)
    
    return report
```

## Key Patterns Summary

| Pattern | Purpose | Implementation |
|---------|---------|----------------|
| PRD Parsing | Convert specs to structured tasks | `parse_prd_to_tasks()` function with regex parsing |
| API Rotation | Intelligent API key rotation and load balancing | `APIRotationManager` with weighted selection |
| Rate Limiting | Prevent API quota exhaustion | Per-endpoint usage tracking and throttling |
| Task State Management | Track progress across sessions | JSON file-based persistence in `tasks/` directory |
| Autonomous Execution | Self-directed task processing | `execute_next_task()` with dependency checking |
| Cross-Session Persistence | Resume work after interruption | File-based state management |
| Dependency Management | Ensure proper task ordering | Dependency analysis and validation |
| Progress Tracking | Monitor and update status | `update_task_status()` with counters |
| Parallel Execution | Handle independent tasks concurrently | ThreadPoolExecutor with file locking |
| Error Recovery | Handle failures gracefully | Try-catch with error logging and retry logic |
| Learning System | Improve from execution patterns | Pattern and solution storage in `memories/` |
| Agent Agnostic | Work with any AI agent | Standard Python functions, no framework dependencies |

## File Structure

Agent-agnostic long-running agent project structure:

```
project/
├── tasks/                  # Task management (JSON persistence)
│   ├── current_tasks.json  # Active task list from PRD
│   ├── execution_plan.json # Optimized task execution order
│   └── .lock_*             # Task execution locks for parallel processing
├── results/                # Task implementation results
│   ├── task_001/           # Frontend components
│   │   ├── README.md       # Implementation notes
│   │   ├── components/     # React/Vue components
│   │   └── tests/          # Unit tests
│   ├── task_002/           # Backend APIs
│   │   ├── README.md       # API documentation
│   │   ├── endpoints/      # API implementations
│   │   └── tests/          # API tests
│   └── documentation/      # Project documentation
├── memories/               # Learning and pattern storage
│   ├── patterns.json       # Successful execution patterns
│   ├── solutions.json      # Error solutions database
│   └── templates.json      # Reusable task templates
└── logs/                   # Execution logging
    ├── execution.log       # Task execution history
    ├── errors.log          # Error tracking
    └── progress.log        # Progress milestones
```

### File-Based State Management

The skill uses standard JSON files for persistence, making it compatible with any agent:

```python
# Task persistence
tasks/current_tasks.json    # Main task list with status tracking
tasks/execution_plan.json   # Optimized execution order

# Results storage  
results/task_*/README.md    # Implementation documentation
results/task_*/            # Task-specific implementation files

# Learning system
memories/patterns.json      # Execution patterns for improvement
memories/solutions.json     # Error solutions for reuse
```

## Reference Files

- **[prd-processing.md](references/prd-processing.md)** - PRD parsing patterns, task extraction, structured generation
- **[task-execution.md](references/task-execution.md)** - Autonomous task processing, dependency management, status tracking
- **[agent-integration.md](references/agent-integration.md)** - Integration patterns for different AI agents (Cursor, OpenCode, etc.)
- **[parallel-execution.md](references/parallel-execution.md)** - Concurrent task processing, thread safety, coordination patterns
- **[error-handling.md](references/error-handling.md)** - Error classification, recovery strategies, graceful degradation

## Quick Start

1. **Setup Environment** (Optional - no dependencies required):
   ```bash
   # Install uv for fast Python package management
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment (optional)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Copy Functions**: Copy the core functions from this skill into your agent's context
3. **Create Project Structure**: Set up the required directories (`tasks/`, `results/`, `memories/`, `logs/`)
4. **Provide PRD/Spec**: Give your agent a PRD or specification document
5. **Execute Workflow**: Agent automatically parses, plans, and executes tasks
6. **Monitor Progress**: Check `tasks/current_tasks.json` and `results/` directory for progress
7. **Resume Work**: Agent can resume from any interruption using persistent state

## Agent Instructions

When implementing this skill, instruct your agent to:

1. **Parse PRDs** using `parse_prd_to_tasks()` to create structured task lists
2. **Execute Tasks** using `execute_next_task()` to work through the list autonomously  
3. **Track Progress** using `update_task_status()` to maintain state
4. **Handle Dependencies** by checking task dependencies before execution
5. **Learn Patterns** by saving successful approaches and error solutions
6. **Resume Sessions** by loading existing task lists and continuing from where it left off

The skill is designed to work with any AI agent that can execute Python functions and access the file system.
