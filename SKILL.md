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

Create the basic directory structure for persistent state management:

```python
def setup_project_structure(project_name: str):
    """Create directory structure for long-running agent."""
    directories = [
        f"tasks/{project_name}",
        f"results/{project_name}", 
        f"memories/{project_name}",
        f"logs/{project_name}"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
```

### Step 2: Implement PRD Processing

Parse requirements documents into structured, executable task lists:

```python
def parse_prd_to_tasks(prd_content: str, project_name: str) -> Dict:
    """Parse PRD into structured task list with dependencies."""
    # See references/prd-processing.md for full implementation
    
    tasks = {
        "project_name": project_name,
        "created_at": datetime.now().isoformat(),
        "total_tasks": 0,
        "completed_tasks": 0,
        "tasks": []
    }
    
    # Extract sections, analyze dependencies, categorize tasks
    # Returns structured JSON with full task metadata
    return tasks
```

**Full Implementation**: See [references/prd-processing.md](references/prd-processing.md)

### Step 3: Set Up API Rotation and Management

Configure intelligent API rotation for external service calls:

```python
def setup_api_rotation(api_configs: List[Dict]):
    """Setup API rotation with multiple endpoints."""
    # See references/api-rotation.md for full implementation
    
    global api_manager
    api_manager = APIRotationManager()
    
    for config in api_configs:
        api_manager.add_endpoint(
            name=config["name"],
            base_url=config["base_url"], 
            api_key=config["api_key"],
            rate_limit=config.get("rate_limit", 60),
            quota_limit=config.get("quota_limit", 1000)
        )
    
    print(f"🔄 API rotation configured with {len(api_configs)} endpoints")
```

**Full Implementation**: See [references/api-rotation.md](references/api-rotation.md)

### Step 4: Implement State Management

Create persistent state management for cross-session continuity:

```python
def save_task_list(task_list: Dict, file_path: str = None):
    """Save task list to persistent storage."""
    # See references/state-management.md for full implementation
    
    if not file_path:
        file_path = f"tasks/{task_list['project_name']}/current_tasks.json"
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(task_list, f, indent=2)

def load_task_list(project_name: str = None, file_path: str = None) -> Dict:
    """Load task list from persistent storage."""
    # Implementation details in references/state-management.md
    pass
```

**Full Implementation**: See [references/state-management.md](references/state-management.md)

### Step 5: Implement Task Execution Engine

Execute tasks autonomously with dependency management:

```python
def execute_next_task(project_name: str) -> Dict:
    """Execute the next available task with dependency checking."""
    # See references/task-execution.md for full implementation
    
    task_list = load_task_list(project_name)
    
    # Find next executable task (dependencies met, status pending)
    next_task = find_next_executable_task(task_list)
    
    if not next_task:
        return {"status": "no_tasks_available"}
    
    # Execute task by category with API rotation support
    result = execute_task_by_category(next_task)
    
    # Update task status and save state
    update_task_status(next_task["id"], "completed" if result["success"] else "failed")
    
    return result
```

**Full Implementation**: See [references/task-execution.md](references/task-execution.md)

### Step 6: Set Up Learning and Memory System

Implement pattern recognition and continuous improvement:

```python
def save_execution_pattern(task: Dict, execution_result: Dict, pattern_file: str = "memories/patterns.json"):
    """Save successful execution patterns for learning."""
    # See references/learning-system.md for full implementation
    
    pattern = {
        "task_category": task["category"],
        "task_type": task.get("type", "general"),
        "execution_approach": execution_result.get("approach"),
        "success_factors": execution_result.get("success_factors", []),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save pattern for future reference
    patterns = load_json_file(pattern_file, [])
    patterns.append(pattern)
    save_json_file(pattern_file, patterns)
```

**Full Implementation**: See [references/learning-system.md](references/learning-system.md)

### Step 7: Agent Integration

Integrate with your specific AI agent framework:

```python
# For any agent framework (Cursor, OpenCode, Claude, etc.)
def run_long_running_agent(prd_content: str, project_name: str):
    """Main entry point for long-running agent workflow."""
    
    # 1. Setup
    setup_project_structure(project_name)
    setup_api_rotation(load_api_config())
    
    # 2. Parse PRD
    task_list = parse_prd_to_tasks(prd_content, project_name)
    save_task_list(task_list)
    
    # 3. Execute tasks
    while has_pending_tasks(project_name):
        result = execute_next_task(project_name)
        
        if result["status"] == "no_tasks_available":
            break
            
        # Learn from execution
        if result.get("success"):
            save_execution_pattern(result["task"], result)
    
    # 4. Generate summary
    return generate_project_summary(project_name)
```

## Agent Framework Integration

### For Cursor, OpenCode, and other AI Agents:

1. **Load this skill** when starting a new project or resuming work
2. **Call `run_long_running_agent()`** with your PRD content
3. **Monitor progress** through the generated task files
4. **Resume anytime** by calling `execute_next_task()` 

### Example Workflow:

```python
# Start new project
prd = "Your PRD content here..."
summary = run_long_running_agent(prd, "ecommerce-platform")

# Resume existing project  
result = execute_next_task("ecommerce-platform")

# Check status
status = get_project_status("ecommerce-platform")
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

```
project-name/
├── tasks/project-name/
│   ├── current_tasks.json    # Current task list and status
│   └── task_history.json     # Completed task history
├── results/project-name/
│   ├── task_001/            # Individual task outputs
│   └── task_002/
├── memories/project-name/
│   ├── patterns.json        # Learned execution patterns
│   └── solutions.json       # Error solutions
└── logs/project-name/
    └── execution.log        # Detailed execution logs
```

## Reference Files

For detailed implementations, see:

- **[prd-processing.md](references/prd-processing.md)** - PRD parsing patterns, task extraction, structured generation
- **[api-rotation.md](references/api-rotation.md)** - API rotation, rate limiting, load balancing, error handling
- **[task-execution.md](references/task-execution.md)** - Autonomous task processing, dependency management, status tracking
- **[state-management.md](references/state-management.md)** - File-based persistence, cross-session continuity, data integrity
- **[agent-integration.md](references/agent-integration.md)** - Integration patterns for different AI agents (Cursor, OpenCode, etc.)
- **[parallel-execution.md](references/parallel-execution.md)** - Concurrent task processing, thread safety, coordination patterns
- **[error-handling.md](references/error-handling.md)** - Error classification, recovery strategies, graceful degradation
- **[learning-system.md](references/learning-system.md)** - Pattern recognition, continuous improvement, memory management

## Quick Start

1. **Parse your PRD**: `tasks = parse_prd_to_tasks(prd_content, "my-project")`
2. **Start execution**: `run_long_running_agent(prd_content, "my-project")`  
3. **Monitor progress**: Check files in `tasks/my-project/`
4. **Resume anytime**: `execute_next_task("my-project")`

## Agent Instructions

This skill works with any AI agent that can:
- Read and write files
- Execute Python functions
- Maintain state across conversations
- Handle JSON data structures

Simply load this skill and call the main functions with your PRD content to begin autonomous task execution with full persistence and recovery capabilities.