# Agent Integration Patterns

## Overview

This reference covers integration patterns for using the long-running-agent skill with different AI agent frameworks including Cursor, OpenCode, Claude, and others.

## Universal Integration Principles

The long-running-agent skill is designed to be framework-agnostic by using:

1. **Standard Python Functions** - No framework-specific dependencies
2. **JSON File Persistence** - Universal file-based state management
3. **Clear Function Interfaces** - Well-defined inputs and outputs
4. **Error Handling** - Graceful degradation and recovery
5. **Progress Transparency** - Clear status reporting and logging

## Integration with Cursor

### Setup Instructions

1. **Install uv** (recommended for fast Python package management):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Copy Core Functions**: Copy all functions from the SKILL.md into your Cursor workspace
3. **Create Project Structure**: Set up the required directories
4. **Initialize Agent**: Configure Cursor to use the long-running-agent workflow

### Environment Setup with uv

```bash
# Create virtual environment (optional - no dependencies required)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development tools (optional)
uv pip install black ruff mypy pytest
```

### Cursor-Specific Integration

```python
# Cursor integration example
def cursor_long_running_agent_workflow(prd_content: str, project_name: str):
    """
    Cursor-specific workflow for long-running agent.
    
    This function integrates with Cursor's file management and execution capabilities.
    """
    print(f"🚀 Starting long-running agent workflow for: {project_name}")
    
    # Step 1: Parse PRD into tasks
    print("📋 Parsing PRD into structured tasks...")
    task_list = parse_prd_to_tasks(prd_content, project_name)
    
    if not save_task_list(task_list):
        print("❌ Failed to save task list")
        return False
    
    print(f"✅ Generated {task_list['total_tasks']} tasks")
    
    # Step 2: Execute tasks autonomously
    print("🔄 Beginning autonomous task execution...")
    completed_count = 0
    
    while True:
        result = execute_next_task()
        
        if not result["success"]:
            if "No executable tasks" in result["message"]:
                print(f"🎉 All tasks completed! ({completed_count} tasks)")
                break
            else:
                print(f"⚠️ Error: {result['message']}")
                # Continue with next task or break based on error type
                continue
        else:
            completed_count += 1
            print(f"✅ Task completed: {result['message']}")
            
            # Show progress every 5 tasks
            if completed_count % 5 == 0:
                report = get_progress_report()
                progress = report["overall_progress"]["completion_percentage"]
                print(f"📊 Progress: {progress}% complete")
    
    # Step 3: Generate final report
    print("📈 Generating final progress report...")
    final_report = get_progress_report()
    
    print(f"""
🎯 Project Completion Summary:
- Project: {final_report['project_name']}
- Total Tasks: {final_report['overall_progress']['total_tasks']}
- Completed: {final_report['overall_progress']['completed_tasks']}
- Failed: {final_report['overall_progress']['failed_tasks']}
- Success Rate: {final_report['overall_progress']['completion_percentage']}%

📁 Results saved to: ./results/
📊 Full report: ./results/progress_report.json
""")
    
    return True

# Usage in Cursor
def main():
    prd_content = """
    # Your PRD content here
    """
    
    cursor_long_running_agent_workflow(prd_content, "My Project")

if __name__ == "__main__":
    main()
```

### Cursor File Integration

```python
# Cursor-specific file operations
def cursor_create_project_structure(project_root: str = "./"):
    """Create project structure optimized for Cursor workspace."""
    
    directories = [
        f"{project_root}/tasks",
        f"{project_root}/results", 
        f"{project_root}/memories",
        f"{project_root}/logs",
        f"{project_root}/src",           # Cursor workspace convention
        f"{project_root}/tests",         # Cursor testing integration
        f"{project_root}/.cursor",       # Cursor-specific settings
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create Cursor-specific configuration
    cursor_config = {
        "long_running_agent": {
            "enabled": True,
            "auto_execute": False,
            "progress_notifications": True,
            "task_directory": "tasks/",
            "results_directory": "results/"
        }
    }
    
    with open(f"{project_root}/.cursor/long_running_agent.json", "w") as f:
        json.dump(cursor_config, f, indent=2)
    
    print("✅ Cursor project structure created")
```

## Integration with OpenCode

### OpenCode-Specific Patterns

```python
def opencode_integration_workflow(prd_content: str):
    """
    OpenCode-specific integration for long-running agent.
    
    Optimized for OpenCode's collaborative development environment.
    """
    
    # OpenCode collaborative features
    def notify_team_progress(task_result: Dict):
        """Notify team members of task completion."""
        if task_result["success"]:
            print(f"🔔 Team Notification: Task {task_result.get('task_id')} completed")
            # Integration with OpenCode's notification system
            
    def create_opencode_branch(task_id: str, task_title: str):
        """Create feature branch for task in OpenCode."""
        branch_name = f"feature/{task_id}-{task_title.lower().replace(' ', '-')}"
        print(f"🌿 Creating branch: {branch_name}")
        # Integration with OpenCode's git workflow
        
    # Enhanced execution with OpenCode features
    def execute_task_with_opencode_integration(task: Dict) -> Dict:
        """Execute task with OpenCode-specific integrations."""
        
        # Create feature branch
        create_opencode_branch(task["id"], task["title"])
        
        # Execute task
        result = execute_task_by_category(task)
        
        # Notify team
        notify_team_progress(result)
        
        return result
    
    # Main workflow
    task_list = parse_prd_to_tasks(prd_content, "OpenCode Project")
    save_task_list(task_list)
    
    # Execute with OpenCode integrations
    while True:
        executable_tasks = get_executable_tasks()
        if not executable_tasks:
            break
            
        task = executable_tasks[0]
        update_task_status(task["id"], "in_progress")
        
        result = execute_task_with_opencode_integration(task)
        
        if result["success"]:
            update_task_status(task["id"], "completed", result["output"])
        else:
            update_task_status(task["id"], "failed", "", "", result.get("error", ""))
```

## Integration with Claude (Anthropic)

### Claude-Specific Implementation

```python
def claude_long_running_session(prd_content: str, session_id: str = None):
    """
    Claude-specific long-running agent implementation.
    
    Designed for Claude's conversation-based interaction model.
    """
    
    if session_id is None:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🤖 Starting Claude long-running session: {session_id}")
    
    # Claude conversation state management
    conversation_state = {
        "session_id": session_id,
        "started_at": datetime.now().isoformat(),
        "current_task": None,
        "completed_tasks": [],
        "conversation_history": []
    }
    
    def save_conversation_state():
        """Save conversation state for Claude session continuity."""
        with open(f"memories/claude_session_{session_id}.json", "w") as f:
            json.dump(conversation_state, f, indent=2)
    
    def load_conversation_state():
        """Load previous conversation state."""
        try:
            with open(f"memories/claude_session_{session_id}.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return conversation_state
    
    # Load previous state if resuming
    conversation_state = load_conversation_state()
    
    # Parse PRD if new session
    if not conversation_state.get("tasks_generated", False):
        print("📋 Parsing PRD into tasks...")
        task_list = parse_prd_to_tasks(prd_content, f"Claude Project {session_id}")
        save_task_list(task_list)
        conversation_state["tasks_generated"] = True
        conversation_state["conversation_history"].append({
            "action": "prd_parsed",
            "timestamp": datetime.now().isoformat(),
            "task_count": task_list["total_tasks"]
        })
    
    # Execute tasks with conversation tracking
    def execute_with_conversation_tracking():
        """Execute tasks while maintaining conversation context."""
        
        while True:
            result = execute_next_task()
            
            if not result["success"]:
                if "No executable tasks" in result["message"]:
                    print("🎉 All tasks in this session completed!")
                    conversation_state["conversation_history"].append({
                        "action": "session_completed",
                        "timestamp": datetime.now().isoformat()
                    })
                    break
                else:
                    print(f"⚠️ Task execution issue: {result['message']}")
                    conversation_state["conversation_history"].append({
                        "action": "task_error",
                        "timestamp": datetime.now().isoformat(),
                        "error": result["message"]
                    })
            else:
                print(f"✅ Completed: {result['message']}")
                conversation_state["completed_tasks"].append({
                    "task_info": result["message"],
                    "timestamp": datetime.now().isoformat()
                })
                conversation_state["conversation_history"].append({
                    "action": "task_completed",
                    "timestamp": datetime.now().isoformat(),
                    "task_info": result["message"]
                })
            
            # Save state after each task
            save_conversation_state()
    
    # Execute tasks
    execute_with_conversation_tracking()
    
    # Generate session summary
    print(f"""
🤖 Claude Session Summary:
- Session ID: {session_id}
- Tasks Completed: {len(conversation_state['completed_tasks'])}
- Session Duration: {datetime.now().isoformat()} (started: {conversation_state['started_at']})
- Conversation History: {len(conversation_state['conversation_history'])} events

💾 Session state saved for future resumption.
""")
    
    return conversation_state
```

## Generic Agent Integration

### Universal Integration Template

```python
def generic_agent_integration(agent_name: str, prd_content: str, config: Dict = None):
    """
    Generic integration template for any AI agent.
    
    This template can be adapted for any agent framework.
    """
    
    if config is None:
        config = {
            "auto_execute": True,
            "progress_notifications": True,
            "error_handling": "graceful",
            "parallel_execution": False,
            "learning_enabled": True
        }
    
    print(f"🤖 Initializing long-running agent for: {agent_name}")
    
    # Step 1: Setup
    project_name = f"{agent_name}_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Step 2: Parse PRD
    task_list = parse_prd_to_tasks(prd_content, project_name)
    save_task_list(task_list)
    
    print(f"📋 Generated {task_list['total_tasks']} tasks for {agent_name}")
    
    # Step 3: Execution strategy based on config
    if config.get("parallel_execution", False):
        print("🔄 Using parallel execution strategy...")
        result = execute_parallel_tasks(max_workers=3)
        print(f"⚡ Parallel execution result: {result['message']}")
    else:
        print("🔄 Using sequential execution strategy...")
        
        task_count = 0
        while True:
            result = execute_next_task()
            
            if not result["success"]:
                if "No executable tasks" in result["message"]:
                    break
                else:
                    if config.get("error_handling") == "strict":
                        print(f"❌ Stopping execution due to error: {result['message']}")
                        break
                    else:
                        print(f"⚠️ Continuing despite error: {result['message']}")
                        continue
            else:
                task_count += 1
                if config.get("progress_notifications", True):
                    print(f"✅ Task {task_count} completed: {result['message']}")
    
    # Step 4: Learning (if enabled)
    if config.get("learning_enabled", True):
        print("🧠 Saving execution patterns for future improvement...")
        # Save patterns based on successful execution
        
    # Step 5: Final report
    final_report = get_progress_report()
    print(f"""
🎯 {agent_name} Execution Complete:
- Project: {final_report['project_name']}
- Success Rate: {final_report['overall_progress']['completion_percentage']}%
- Total Tasks: {final_report['overall_progress']['total_tasks']}
- Completed: {final_report['overall_progress']['completed_tasks']}
- Failed: {final_report['overall_progress']['failed_tasks']}
""")
    
    return final_report

# Usage examples for different agents
def integrate_with_cursor(prd_content: str):
    return generic_agent_integration("Cursor", prd_content, {
        "auto_execute": True,
        "progress_notifications": True,
        "parallel_execution": True,
        "learning_enabled": True
    })

def integrate_with_opencode(prd_content: str):
    return generic_agent_integration("OpenCode", prd_content, {
        "auto_execute": True,
        "progress_notifications": True,
        "parallel_execution": False,  # OpenCode prefers sequential for collaboration
        "learning_enabled": True
    })

def integrate_with_custom_agent(agent_name: str, prd_content: str, custom_config: Dict):
    return generic_agent_integration(agent_name, prd_content, custom_config)
```

## Best Practices for Agent Integration

### 1. State Management
- Always use the provided JSON persistence functions
- Create agent-specific state directories if needed
- Implement proper error handling for file operations

### 2. Progress Reporting
- Provide regular progress updates appropriate for the agent's UI
- Use the built-in progress reporting functions
- Adapt notification style to agent's communication patterns

### 3. Error Handling
- Implement graceful degradation for agent-specific errors
- Use the learning system to improve error handling over time
- Provide clear error messages and recovery suggestions

### 4. Performance Optimization
- Use parallel execution when the agent supports it
- Implement appropriate batching for large task lists
- Monitor resource usage and adjust execution strategy

### 5. User Experience
- Provide clear status updates and progress indicators
- Allow for user intervention and manual overrides
- Maintain transparency about what the agent is doing

The long-running-agent skill is designed to be highly adaptable to different agent frameworks while maintaining consistency in core functionality.