# DeepAgents Integration Patterns

## Overview

This reference covers proper integration with DeepAgents API, including backend configuration, memory management, and compliance with DeepAgents best practices.

## Core DeepAgents Setup

```python
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
import json
from datetime import datetime

def create_long_running_agent(project_dir: str = "./", model: str = "anthropic:claude-sonnet-4-5-20250929"):
    """Create a DeepAgents-based long-running agent with proper configuration."""
    
    # Create persistent store for cross-session memory
    store = InMemoryStore()
    
    # Configure composite backend for different data types
    backend = CompositeBackend(
        default=lambda rt: StateBackend(rt),  # Default for temporary state
        routes={
            "/tasks/": lambda rt: StoreBackend(rt),      # Persistent task storage
            "/memories/": lambda rt: StoreBackend(rt),   # Cross-session memory
            "/results/": lambda rt: FilesystemBackend(rt), # Task implementation results
            "/logs/": lambda rt: FilesystemBackend(rt),    # Execution logs
        }
    )
    
    # Define project-specific tools
    tools = [
        parse_prd_to_tasks,
        execute_next_task,
        update_task_status,
        check_task_dependencies,
        generate_progress_report,
        save_project_context,
        load_project_patterns
    ]
    
    # Create agent with full DeepAgents integration
    agent = create_deep_agent(
        model=model,
        tools=tools,
        store=store,
        backend=backend,
        memory=[
            "~/.deepagents/AGENTS.md",           # Global agent preferences
            f"{project_dir}/.deepagents/AGENTS.md", # Project-specific context
        ],
        system_prompt=get_long_running_agent_prompt(),
        # Optional: Add interrupt configuration for human oversight
        interrupt_before=["execute_next_task", "update_task_status"],
        interrupt_after=["parse_prd_to_tasks"]
    )
    
    return agent

def get_long_running_agent_prompt() -> str:
    """Get the system prompt for long-running agent."""
    return """You are a long-running autonomous agent specialized in processing PRDs/specifications and executing tasks over extended periods.

CORE CAPABILITIES:
1. Parse PRDs and specifications into structured, executable task lists
2. Execute tasks autonomously with proper dependency management
3. Maintain persistent state across sessions and interruptions
4. Learn from execution patterns and improve over time
5. Handle errors gracefully with appropriate recovery strategies

WORKFLOW PROCESS:
1. PRD PROCESSING:
   - Parse PRD content using parse_prd_to_tasks tool
   - Generate structured JSON task list with dependencies
   - Save task list to /tasks/current_tasks.json using write_file
   - Validate task structure and dependencies

2. TASK EXECUTION:
   - Use execute_next_task to find and execute available tasks
   - Check dependencies with check_task_dependencies before execution
   - Update progress with update_task_status throughout execution
   - Save implementation results to /results/[task_id]/ directory

3. STATE MANAGEMENT:
   - All task data persists in /tasks/ directory (StoreBackend)
   - Project learnings and patterns go to /memories/ directory (StoreBackend)
   - Implementation results saved to /results/ directory (FilesystemBackend)
   - Execution logs maintained in /logs/ directory (FilesystemBackend)

4. MEMORY INTEGRATION:
   - Read .deepagents/AGENTS.md files for project context
   - Save successful patterns to /memories/ for future reference
   - Learn from past executions to improve task parsing and execution

5. ERROR HANDLING:
   - Classify errors (transient, configuration, logic, dependency)
   - Apply appropriate recovery strategies (retry, manual intervention, degradation)
   - Log failures and update task status appropriately
   - Learn from failures to prevent similar issues

TOOL USAGE GUIDELINES:
- Always use filesystem tools (read_file, write_file, ls, etc.) for state management
- Coordinate tool usage to maintain data consistency
- Use memory tools to save and load project patterns
- Generate regular progress reports for transparency

PERSISTENCE RULES:
- Task lists must be saved to /tasks/current_tasks.json
- All progress updates must be persisted immediately
- Cross-session state must be maintained in appropriate backend routes
- Memory patterns should be saved for future project improvements

You have access to all standard DeepAgents filesystem tools automatically. Use them extensively for state management and persistence."""
```

## Backend Configuration Patterns

### Basic Backend Setup

```python
def create_basic_backend():
    """Create basic backend configuration for simple projects."""
    return CompositeBackend(
        default=lambda rt: StateBackend(rt),
        routes={
            "/tasks/": lambda rt: StoreBackend(rt),
            "/memories/": lambda rt: StoreBackend(rt)
        }
    )

def create_advanced_backend(project_root: str):
    """Create advanced backend configuration with filesystem integration."""
    return CompositeBackend(
        default=lambda rt: StateBackend(rt),
        routes={
            # Persistent storage for task management
            "/tasks/": lambda rt: StoreBackend(rt),
            "/memories/": lambda rt: StoreBackend(rt),
            
            # Filesystem storage for implementation artifacts
            "/results/": lambda rt: FilesystemBackend(f"{project_root}/results"),
            "/logs/": lambda rt: FilesystemBackend(f"{project_root}/logs"),
            "/docs/": lambda rt: FilesystemBackend(f"{project_root}/docs"),
            
            # Temporary storage for work-in-progress
            "/temp/": lambda rt: StateBackend(rt),
        }
    )
```

### Memory Store Configuration

```python
def create_persistent_memory_store():
    """Create persistent memory store for cross-session data."""
    # In production, use persistent store like Redis or database
    return InMemoryStore()

def create_shared_memory_store(redis_url: str):
    """Create shared memory store for multi-agent coordination."""
    # Example for production deployment
    # from langgraph.store.redis import RedisStore
    # return RedisStore(redis_url)
    return InMemoryStore()  # Fallback for development
```

## Memory Integration Patterns

### Project Context Management

```python
@tool
def save_project_context(context_data: str, context_type: str = "general") -> str:
    """Save project context to DeepAgents memory system."""
    
    context_map = {
        "patterns": "/memories/project_patterns.md",
        "solutions": "/memories/error_solutions.md", 
        "templates": "/memories/task_templates.json",
        "learnings": "/memories/project_learnings.md",
        "general": "/memories/project_context.md"
    }
    
    file_path = context_map.get(context_type, "/memories/project_context.md")
    
    return f"""
    Save project context to {file_path}:
    
    Context Type: {context_type}
    Timestamp: {datetime.now().isoformat()}
    Content: {context_data}
    
    Use write_file tool to append this context to {file_path}
    """

@tool
def load_project_patterns(pattern_type: str = "all") -> str:
    """Load relevant project patterns from memory."""
    
    return f"""
    Load project patterns for {pattern_type}:
    
    1. Use read_file to load relevant memory files:
       - /memories/project_patterns.md (implementation patterns)
       - /memories/error_solutions.md (error handling patterns)
       - /memories/task_templates.json (task structure templates)
       - /memories/project_learnings.md (general learnings)
    
    2. Filter patterns relevant to current context
    3. Apply patterns to improve current task execution
    """

@tool
def update_agent_memory(learning_data: Dict) -> str:
    """Update agent memory with new learnings from task execution."""
    
    return f"""
    Update agent memory with new learning:
    
    Learning Data: {json.dumps(learning_data, indent=2)}
    
    1. Categorize learning by type (pattern, solution, template, general)
    2. Append to appropriate memory file in /memories/
    3. Update .deepagents/AGENTS.md with relevant project context
    4. Log memory update in /logs/memory_updates.log
    """
```

### AGENTS.md File Management

```python
def create_project_agents_file(project_dir: str, project_info: Dict):
    """Create project-specific AGENTS.md file for context."""
    
    agents_content = f"""# {project_info.get('name', 'Project')} - Agent Context

## Project Overview
{project_info.get('description', 'Long-running agent project')}

## Task Execution Patterns
- PRD parsing approach: {project_info.get('prd_format', 'markdown')}
- Implementation style: {project_info.get('implementation_style', 'modular')}
- Testing strategy: {project_info.get('testing_strategy', 'comprehensive')}

## Dependency Management
- Dependency resolution: {project_info.get('dependency_strategy', 'topological')}
- Parallel execution: {project_info.get('parallel_execution', 'enabled')}

## Error Handling
- Retry strategy: {project_info.get('retry_strategy', 'exponential_backoff')}
- Failure escalation: {project_info.get('failure_escalation', 'manual_intervention')}

## Memory and Learning
- Pattern recognition: enabled
- Cross-session learning: enabled
- Template generation: enabled

## File Organization
- Task storage: /tasks/
- Results storage: /results/
- Memory storage: /memories/
- Logs storage: /logs/

## Quality Standards
- Code coverage minimum: {project_info.get('coverage_min', '80%')}
- Documentation required: {project_info.get('docs_required', 'yes')}
- Review process: {project_info.get('review_process', 'automated')}

Last updated: {datetime.now().isoformat()}
"""
    
    # Save to project directory
    agents_file_path = f"{project_dir}/.deepagents/AGENTS.md"
    return f"Create directory and save content to {agents_file_path}"
```

## Subagent Integration Patterns

### Task-Specific Subagents

```python
from deepagents import CompiledSubAgent

def create_specialized_subagents():
    """Create specialized subagents for different task types."""
    
    # Frontend implementation subagent
    frontend_agent = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[implement_frontend_task, run_frontend_tests],
        system_prompt="""You are a frontend implementation specialist.
        Focus on creating high-quality UI components, pages, and user interactions.
        Always include proper testing and documentation."""
    )
    
    # Backend implementation subagent  
    backend_agent = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929", 
        tools=[implement_backend_task, run_backend_tests],
        system_prompt="""You are a backend implementation specialist.
        Focus on creating robust APIs, services, and data processing logic.
        Ensure proper security, validation, and error handling."""
    )
    
    # Testing and validation subagent
    testing_agent = create_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[run_tests, validate_implementation, generate_test_reports],
        system_prompt="""You are a testing and validation specialist.
        Focus on comprehensive testing, quality assurance, and validation.
        Ensure all implementations meet quality standards."""
    )
    
    return [
        CompiledSubAgent(
            name="frontend-specialist",
            description="Specialized agent for frontend implementation tasks",
            runnable=frontend_agent
        ),
        CompiledSubAgent(
            name="backend-specialist", 
            description="Specialized agent for backend implementation tasks",
            runnable=backend_agent
        ),
        CompiledSubAgent(
            name="testing-specialist",
            description="Specialized agent for testing and validation tasks", 
            runnable=testing_agent
        )
    ]

def create_coordinated_agent_system():
    """Create main coordinator agent with specialized subagents."""
    
    subagents = create_specialized_subagents()
    
    coordinator = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[
            parse_prd_to_tasks,
            coordinate_task_execution,
            manage_dependencies,
            generate_progress_report
        ],
        subagents=subagents,
        store=InMemoryStore(),
        backend=create_advanced_backend("./"),
        memory=[
            "~/.deepagents/AGENTS.md",
            "./.deepagents/AGENTS.md"
        ],
        system_prompt="""You are a project coordinator agent managing specialized subagents.
        
        COORDINATION RESPONSIBILITIES:
        - Parse PRDs and create task lists
        - Assign tasks to appropriate specialist subagents
        - Manage dependencies and execution order
        - Monitor progress and handle escalations
        - Generate reports and maintain project state
        
        SUBAGENT DELEGATION:
        - Use frontend-specialist for UI/UX tasks
        - Use backend-specialist for API/service tasks  
        - Use testing-specialist for validation tasks
        - Coordinate between subagents for integrated features
        
        Always maintain overall project coherence while leveraging specialist expertise."""
    )
    
    return coordinator
```

## Error Handling and Recovery

### DeepAgents-Specific Error Patterns

```python
@tool
def handle_deepagents_errors(error_type: str, error_details: str) -> str:
    """Handle DeepAgents-specific errors with appropriate recovery."""
    
    error_handlers = {
        "backend_routing": """
        Backend routing error - check CompositeBackend configuration:
        1. Verify route patterns match file paths
        2. Ensure backend types (StateBackend, StoreBackend) are appropriate
        3. Check for conflicting route definitions
        4. Test with simple path first
        """,
        
        "memory_access": """
        Memory access error - check store and memory configuration:
        1. Verify InMemoryStore is properly initialized
        2. Check AGENTS.md file paths and permissions
        3. Ensure memory files exist and are readable
        4. Test memory access with simple read/write
        """,
        
        "tool_coordination": """
        Tool coordination error - check tool interactions:
        1. Verify tool dependencies and call order
        2. Check for tool parameter validation
        3. Ensure proper error propagation between tools
        4. Test tools individually before coordination
        """,
        
        "subagent_communication": """
        Subagent communication error - check subagent configuration:
        1. Verify CompiledSubAgent setup and naming
        2. Check subagent tool availability
        3. Ensure proper message passing between agents
        4. Test subagent individually before integration
        """
    }
    
    handler = error_handlers.get(error_type, "Generic error handling")
    
    return f"""
    DeepAgents Error Handling for {error_type}:
    
    {handler}
    
    Error Details: {error_details}
    
    Recovery Actions:
    1. Log error details to /logs/deepagents_errors.log
    2. Apply specific recovery strategy above
    3. Test recovery with minimal configuration
    4. Gradually restore full functionality
    5. Update error handling patterns in /memories/error_solutions.md
    """
```

## Configuration Validation

```python
@tool
def validate_deepagents_setup() -> str:
    """Validate DeepAgents configuration for long-running agent."""
    
    return """
    DeepAgents Setup Validation Checklist:
    
    1. CORE CONFIGURATION:
       ✓ create_deep_agent() used (not raw StateGraph)
       ✓ Proper model string format
       ✓ Required tools are defined and imported
       ✓ System prompt is comprehensive
    
    2. BACKEND CONFIGURATION:
       ✓ CompositeBackend properly configured
       ✓ Route patterns match usage patterns
       ✓ Backend types appropriate for data types
       ✓ No conflicting route definitions
    
    3. MEMORY INTEGRATION:
       ✓ InMemoryStore initialized
       ✓ AGENTS.md files exist and accessible
       ✓ Memory routes configured in backend
       ✓ Memory tools integrated
    
    4. TOOL INTEGRATION:
       ✓ All tools use @tool decorator
       ✓ Tool parameters properly typed
       ✓ Tool coordination logic correct
       ✓ Error handling in tools
    
    5. PERSISTENCE SETUP:
       ✓ File paths match backend routes
       ✓ Directory structure exists
       ✓ Permissions allow read/write
       ✓ Backup and recovery possible
    
    Run validation and report any configuration issues.
    """
```

## Usage Examples

### Basic Agent Creation

```python
# Create basic long-running agent
agent = create_long_running_agent(
    project_dir="./my_project",
    model="anthropic:claude-sonnet-4-5-20250929"
)

# Configure for specific project
config = {"configurable": {"thread_id": "project-123"}}
```

### Advanced Multi-Agent Setup

```python
# Create coordinated multi-agent system
coordinator = create_coordinated_agent_system()

# Execute with proper configuration
result = coordinator.invoke({
    "messages": [{
        "role": "user",
        "content": "Process this PRD and coordinate implementation across specialized teams"
    }]
}, config={"configurable": {"thread_id": "enterprise-project-456"}})
```

### Memory and Context Management

```python
# Save project patterns for future use
agent.invoke({
    "messages": [{
        "role": "user", 
        "content": "Save the successful authentication implementation pattern to memory for future projects"
    }]
})

# Load and apply previous patterns
agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Load similar authentication patterns from memory and apply to current task"
    }]
})
```