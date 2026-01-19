# Long Running Agent Skill

An open Agent Skills standard-compliant skill for building autonomous, long-running AI agents that work with **any AI agent framework** (Cursor, OpenCode, Claude, etc.). Parse PRDs/specifications, generate structured task lists, and execute tasks autonomously with state persistence and recovery.

## 🌟 Universal Compatibility

This skill follows the [Agent Skills open standard](https://agentskills.io) and works with:
- **Cursor** - IDE-integrated development workflows
- **OpenCode** - Collaborative development environments  
- **Claude** - Conversation-based agent interactions
- **Custom AI Agents** - Any agent with file system access
- **Future Platforms** - Framework-agnostic design ensures compatibility

## Installation

### Quick Installation from GitHub

```bash
# One-line installation (recommended)
curl -sSL https://raw.githubusercontent.com/bowen31337/long-running-agent-skill/main/scripts/install.sh | bash

# Or specify custom directory
curl -sSL https://raw.githubusercontent.com/bowen31337/long-running-agent-skill/main/scripts/install.sh | bash -s my-project
```

### Manual Installation

```bash
# Clone from GitHub
git clone https://github.com/bowen31337/long-running-agent-skill.git
cd long-running-agent-skill

# Install uv for fast Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment (optional - skill has no dependencies)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Agent Skills Compatible Installation

```bash
# For Agent Skills compatible platforms
skills install long-running-agent

# Or add from GitHub URL directly in compatible agents
# GitHub URL: https://github.com/bowen31337/long-running-agent-skill
```

### Manual Integration

```bash
# Copy core functions into your agent's context
cp long-running-agent/examples/universal_example.py ./

# If you need development tools
uv pip install -e ".[dev]"

# Customize for your specific agent framework
```

## When to Use This Skill

This skill activates when you need to:

- Parse PRDs or specifications into executable task lists
- Build agents that execute tasks autonomously with dependency management
- Implement cross-session state persistence using standard JSON files
- Create learning agents that improve from execution patterns
- Handle complex multi-step workflows with proper error recovery
- Build agent-agnostic systems that work with any AI agent framework

## Quick Start

Once installed, ask any AI agent to help you build a long-running agent:

```
"Create an agent that can parse this PRD and execute the tasks autonomously"

"Build an agent-agnostic system that processes specifications into task lists"

"Implement a long-running agent that persists state across sessions using JSON files"

"Create an autonomous agent that works with Cursor, OpenCode, and other AI frameworks"
```

### Universal Example Usage

```python
# Works with ANY AI agent framework - no dependencies!
from long_running_agent import universal_long_running_agent_workflow

# Example PRD content
prd_content = """
# E-commerce Platform

## User Registration
Implement user registration with email verification
- Email validation and uniqueness check
- Password strength requirements
- Email verification flow

## Product Catalog  
Create product browsing and search functionality
- Product listing with pagination
- Search with filters
- Product detail pages
"""

# Execute with any AI agent
success = universal_long_running_agent_workflow(prd_content, "E-commerce MVP")

# The agent automatically:
# ✅ Parses PRD into structured JSON task list with dependencies
# ✅ Executes tasks autonomously (frontend, backend, database, etc.)
# ✅ Persists state using standard JSON files (no framework lock-in)
# ✅ Handles errors gracefully with retry logic
# ✅ Learns from patterns for continuous improvement
# ✅ Works across Cursor, OpenCode, Claude, and custom agents
```

### Framework-Specific Integration

```python
# Cursor IDE Integration
from agent_integration import cursor_long_running_agent_workflow
cursor_long_running_agent_workflow(prd_content, "Cursor Project")

# OpenCode Collaborative Integration  
from agent_integration import opencode_integration_workflow
opencode_integration_workflow(prd_content)

# Claude Conversation Integration
from agent_integration import claude_long_running_session  
claude_long_running_session(prd_content, session_id="project-123")

# Generic Agent Integration
from agent_integration import generic_agent_integration
generic_agent_integration("MyCustomAgent", prd_content, custom_config)
```

## 📁 Skill Contents

### Core Instructions ([SKILL.md](long-running-agent/SKILL.md))

Agent-agnostic implementation guidance covering six core systems:

1. **PRD/Spec Processing** - Parse requirements into structured, executable task lists
2. **Task Execution Engine** - Autonomous task processing with dependency management  
3. **State Management** - JSON-based persistence for workflow states and task tracking
4. **Error Handling** - Classification, recovery strategies, and graceful degradation
5. **Cross-Session Persistence** - Resume work across interruptions and restarts
6. **Learning & Memory** - Pattern recognition and improvement over time

### Reference Files

Detailed implementation patterns in `references/`:

| File | Contents |
|------|----------|
| **[prd-processing.md](references/prd-processing.md)** | PRD parsing patterns, task extraction, structured generation |
| **[task-execution.md](references/task-execution.md)** | Autonomous task processing, dependency management, status tracking |
| **[agent-integration.md](references/agent-integration.md)** | Integration patterns for Cursor, OpenCode, Claude, and custom agents |
| **[parallel-execution.md](references/parallel-execution.md)** | Concurrent task processing, thread safety, coordination patterns |
| **[error-handling.md](references/error-handling.md)** | Error classification, recovery strategies, graceful degradation |

### Examples

| File | Purpose |
|------|---------|
| **[universal_example.py](scripts/universal_example.py)** | Complete working example for any AI agent framework |
| **[complete_example.py](scripts/complete_example.py)** | Legacy DeepAgents example (for reference) |

## 🔧 Key Implementation Patterns

### Universal PRD Parsing

Parse any PRD format into structured tasks:

```python
def parse_prd_to_tasks(prd_content: str, project_name: str) -> Dict:
    """Parse PRD into structured task list with dependencies."""
    # Automatically detects headers, extracts acceptance criteria
    # Analyzes dependencies (auth before frontend, backend before UI)
    # Categories tasks (frontend, backend, database, auth, testing)
    # Returns JSON with full task metadata
```

### Intelligent API Rotation & Rate Limiting

Manages multiple API endpoints with automatic rotation, rate limiting, and load balancing:

```python
# Setup API rotation with multiple endpoints
api_configs = [
    {"name": "primary_api", "base_url": "https://api.example.com", "api_key": "key1", "rate_limit": 100},
    {"name": "backup_api", "base_url": "https://backup.example.com", "api_key": "key2", "rate_limit": 60}
]
setup_api_rotation(api_configs)

# Make API calls with automatic rotation and error handling
result = make_api_call(request_function, endpoint, data)
# Automatically handles: rate limits, quota management, failover, load balancing
```

### Agent-Agnostic State Management

JSON-based persistence that works everywhere:

```python
# Save task list (works with any agent)
def save_task_list(task_list: Dict, file_path: str = "tasks/current_tasks.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(task_list, f, indent=2)

# Load task list (universal compatibility)
def load_task_list(file_path: str = "tasks/current_tasks.json") -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)
```

### Autonomous Task Execution

Self-directed task processing with dependency management:

```python
def execute_next_task() -> Dict:
    """Find and execute next available task."""
    executable_tasks = get_executable_tasks()  # Dependencies met
    if not executable_tasks:
        return {"success": False, "message": "No executable tasks available"}
    
    task = executable_tasks[0]  # Highest priority
    result = execute_task_by_category(task)  # Category-specific implementation
    update_task_status(task["id"], "completed" if result["success"] else "failed")
```

### Cross-Platform Error Handling

Graceful degradation across different agent environments:

```python
def execute_task_by_category(task: Dict) -> Dict:
    try:
        if task["category"] == "frontend":
            return implement_frontend_task(task)
        elif task["category"] == "backend":
            return implement_backend_task(task)
        # ... other categories
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Learning and Memory System

Continuous improvement across sessions:

```python
def save_execution_pattern(task: Dict, result: Dict):
    """Save successful patterns for future learning."""
    patterns = load_json_file("memories/patterns.json", default=[])
    patterns.append({
        "category": task["category"],
        "approach": result["approach"],
        "success": result["success"],
        "timestamp": datetime.now().isoformat()
    })
    save_json_file("memories/patterns.json", patterns)
```

## 📂 Universal Project Structure

Agent-agnostic file structure that works with any AI agent:

```
my-project/
├── tasks/                     # Task management (JSON persistence)
│   ├── current_tasks.json     # Active task list from PRD parsing
│   ├── execution_plan.json    # Optimized task execution order
│   └── .lock_*                # Task locks for parallel processing
├── results/                   # Task implementation results
│   ├── task_001/              # Frontend components
│   │   ├── README.md          # Implementation documentation
│   │   ├── components/        # React/Vue/Angular components
│   │   └── tests/             # Unit tests
│   ├── task_002/              # Backend APIs
│   │   ├── README.md          # API documentation
│   │   ├── routes/            # API route definitions
│   │   ├── controllers/       # Request handlers
│   │   ├── services/          # Business logic
│   │   └── tests/             # API tests
│   ├── task_003/              # Database schemas
│   └── progress_report.json   # Overall project progress
├── memories/                  # Learning and pattern storage
│   ├── patterns.json          # Successful execution patterns
│   ├── solutions.json         # Error solutions database
│   └── templates.json         # Reusable task templates
└── logs/                      # Execution logging
    ├── execution.log          # Task execution history
    ├── errors.log             # Error tracking and analysis
    └── progress.log           # Progress milestones
```

### File Format Standards

All persistence uses standard JSON for maximum compatibility:

- **Task Lists**: Standard JSON with task metadata, dependencies, status
- **Results**: Markdown documentation + implementation files
- **Memory**: JSON arrays for patterns, solutions, and templates
- **Logs**: Plain text logs for debugging and analysis

## 🚀 Complete Usage Examples

### Basic PRD-to-Implementation Workflow

```python
# Universal workflow - works with any AI agent
def main():
    # Sample PRD content
    prd_content = """
    # User Management System
    
    ## User Registration
    Implement user registration with email verification
    - Email validation and uniqueness check
    - Password strength requirements
    - Email verification flow
    
    ## User Authentication
    Secure login system with session management
    - Email/password authentication
    - JWT token generation
    - Session timeout handling
    """
    
    # Execute with universal workflow
    success = universal_long_running_agent_workflow(
        prd_content, 
        "User Management MVP"
    )
    
    if success:
        print("✅ Project completed successfully!")
        # Check results in ./results/ directory
        # Review progress in ./results/progress_report.json

if __name__ == "__main__":
    main()
```

### Agent-Specific Integration Examples

```python
# Cursor IDE Integration
def cursor_example():
    """Example for Cursor IDE users."""
    prd_content = load_prd_from_file("requirements.md")
    
    # Cursor-optimized workflow with workspace integration
    result = cursor_long_running_agent_workflow(prd_content, "Cursor Project")
    
    # Results automatically organized in Cursor workspace
    # Progress notifications in Cursor status bar
    return result

# OpenCode Collaborative Integration
def opencode_example():
    """Example for OpenCode collaborative development."""
    prd_content = get_prd_from_team_discussion()
    
    # OpenCode workflow with team notifications
    result = opencode_integration_workflow(prd_content)
    
    # Creates feature branches for each task
    # Notifies team members of progress
    # Integrates with OpenCode's collaboration features
    return result

# Claude Conversation Integration
def claude_example():
    """Example for Claude conversation-based interaction."""
    session_id = "user-project-2026"
    
    # Claude session with conversation state management
    result = claude_long_running_session(prd_content, session_id)
    
    # Maintains conversation context across sessions
    # Provides detailed progress explanations
    # Allows for interactive guidance and adjustments
    return result
```

### Advanced Features

```python
# Parallel task execution (when supported by agent)
def parallel_execution_example():
    """Execute independent tasks concurrently."""
    
    # Parse PRD and identify independent tasks
    task_list = parse_prd_to_tasks(prd_content, "Parallel Project")
    save_task_list(task_list)
    
    # Execute tasks in parallel (thread-safe)
    result = execute_parallel_tasks(max_workers=3)
    
    print(f"Parallel execution: {result['message']}")
    return result

# Learning from previous projects
def learning_example():
    """Demonstrate learning and pattern reuse."""
    
    # Load patterns from previous successful projects
    patterns = load_similar_patterns("frontend", "user registration")
    
    # Apply learned patterns to new tasks
    for pattern in patterns:
        print(f"Applying pattern: {pattern['execution_approach']}")
    
    # Execute with improved accuracy based on learning
    result = execute_next_task()
    
    # Save new patterns for future projects
    if result["success"]:
        save_execution_pattern(task, result, "success")
```

## 🎯 Agent Skills Compliance

This skill fully complies with the [Agent Skills open standard](https://agentskills.io):

- ✅ **Standard Format**: Proper YAML frontmatter with required `name` and `description`
- ✅ **Optional Fields**: Includes `license` and `compatibility` information
- ✅ **Progressive Disclosure**: Main instructions under 500 lines, details in references
- ✅ **Self-Documenting**: Clear instructions readable by humans and agents
- ✅ **Portable**: No framework dependencies, works with any compatible agent
- ✅ **Extensible**: Modular design allows customization for specific use cases

### Compatibility Requirements

```yaml
compatibility: Requires file system access, JSON processing, and ability to execute tasks over extended periods. Compatible with any AI agent that supports file operations and persistent state management.
```

## 🤝 Contributing

This skill follows the Agent Skills open standard. Contributions welcome:

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** with multiple agent frameworks (Cursor, OpenCode, etc.)
4. **Submit** a pull request with Agent Skills compliance validation

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

## 🔗 Links

- **[Agent Skills Standard](https://agentskills.io)** - Open standard specification
- **[Skill Documentation](SKILL.md)** - Complete implementation guide
- **[Universal Example](scripts/universal_example.py)** - Working code for any agent
- **[Integration Patterns](references/agent-integration.md)** - Framework-specific guides
