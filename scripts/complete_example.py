#!/usr/bin/env python3
"""
Complete Example: Long-Running Agent (Legacy DeepAgents Reference)
================================================================

⚠️  LEGACY REFERENCE: This example shows the old DeepAgents-specific implementation.
    For new projects, use universal_example.py which works with any AI agent framework.

This example demonstrates a complete PRD-to-implementation workflow using DeepAgents.
The agent can:
1. Parse PRD/specifications into structured task lists
2. Execute tasks autonomously with dependency management
3. Persist state across sessions
4. Learn from execution patterns

Setup with uv (recommended):
    # Install uv if not already installed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Install DeepAgents (for this legacy example only)
    uv pip install deepagents
    
    # Run the example
    uv run python complete_example.py

Traditional usage:
    pip install deepagents
    python complete_example.py

Note: For new projects, use universal_example.py instead - it's agent-agnostic!
"""

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from typing import Dict, List, Any
import json
import re
from datetime import datetime
from enum import Enum
import os

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress" 
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
def parse_prd_to_tasks(prd_content: str, project_name: str = "Generated Project") -> str:
    """Parse PRD/specification content into structured task list JSON."""
    try:
        tasks = []
        task_id_counter = 1
        
        # Split content into sections by headers
        sections = re.split(r'\n(?=#{1,3}\s)', prd_content)
        
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # Extract title from header
            header_line = lines[0]
            title_match = re.match(r'^#{1,3}\s+(.+)', header_line)
            if not title_match:
                continue
                
            title = title_match.group(1).strip()
            description_lines = lines[1:] if len(lines) > 1 else []
            
            # Parse description and extract acceptance criteria
            acceptance_criteria = []
            remaining_description = []
            
            for line in description_lines:
                line = line.strip()
                if re.match(r'^[-*]\s+|^\d+\.\s+', line):
                    criteria = re.sub(r'^[-*]\s+|^\d+\.\s+', '', line)
                    acceptance_criteria.append(criteria)
                else:
                    remaining_description.append(line)
            
            # Determine priority and effort
            content_lower = (title + ' ' + ' '.join(description_lines)).lower()
            
            priority = "medium"
            if any(word in content_lower for word in ['critical', 'urgent', 'high priority']):
                priority = "high"
            elif any(word in content_lower for word in ['nice to have', 'optional', 'low priority']):
                priority = "low"
            
            effort = "medium"
            if any(word in content_lower for word in ['simple', 'basic', 'quick']):
                effort = "small"
            elif any(word in content_lower for word in ['complex', 'advanced', 'large', 'system']):
                effort = "large"
            
            # Categorize task
            category = "general"
            if any(word in content_lower for word in ['ui', 'interface', 'frontend', 'component']):
                category = "frontend"
            elif any(word in content_lower for word in ['api', 'backend', 'server', 'database']):
                category = "backend"
            elif any(word in content_lower for word in ['test', 'testing', 'validation']):
                category = "testing"
            elif any(word in content_lower for word in ['auth', 'login', 'security']):
                category = "authentication"
            
            task = {
                "id": f"task_{task_id_counter:03d}",
                "title": title,
                "description": '\\n'.join(remaining_description).strip() or title,
                "category": category,
                "priority": priority,
                "estimated_effort": effort,
                "acceptance_criteria": acceptance_criteria,
                "dependencies": [],
                "status": TaskStatus.PENDING.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "result": "",
                "progress_notes": "",
                "error_details": ""
            }
            
            tasks.append(task)
            task_id_counter += 1
        
        # Analyze dependencies (backend before frontend, auth before user features, etc.)
        tasks = analyze_task_dependencies(tasks)
        
        task_list = {
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
        
        return json.dumps(task_list, indent=2)
        
    except Exception as e:
        return f"Error parsing PRD: {str(e)}"

def analyze_task_dependencies(tasks: List[Dict]) -> List[Dict]:
    """Analyze and set task dependencies based on categories and content."""
    
    dependency_rules = [
        ("authentication", "frontend"),
        ("backend", "frontend"),
        ("database", "backend"),
        ("api", "frontend")
    ]
    
    for i, task in enumerate(tasks):
        dependencies = []
        
        for j, other_task in enumerate(tasks[:i]):
            for prereq, dependent in dependency_rules:
                if (prereq in other_task['category'] or prereq in other_task['title'].lower()) and \\
                   (dependent in task['category'] or dependent in task['title'].lower()):
                    dependencies.append(other_task['id'])
        
        task['dependencies'] = list(set(dependencies))
    
    return tasks

@tool
def execute_next_task() -> str:
    """Find and execute the next available task."""
    return """
    To execute the next task:
    
    1. Use read_file to load /tasks/current_tasks.json
    2. Find tasks with status='pending' and all dependencies completed
    3. Select highest priority task (high > medium > low)
    4. Update task status to 'in_progress' using update_task_status
    5. Implement the task based on its category:
       - Frontend: Create UI components and interfaces
       - Backend: Implement APIs and business logic
       - Database: Create schemas and migrations
       - Authentication: Implement auth flows and security
       - Testing: Create test suites and validation
    6. Save implementation to /results/[task_id]/ directory
    7. Update task status to 'completed' with results
    8. Continue with next available task
    """

@tool
def update_task_status(task_id: str, status: str, result: str = "", progress_notes: str = "", error_details: str = "") -> str:
    """Update task status and progress information."""
    
    valid_statuses = [s.value for s in TaskStatus]
    if status not in valid_statuses:
        return f"Invalid status. Valid options: {valid_statuses}"
    
    return f"""
    To update task {task_id}:
    
    1. Use read_file to load /tasks/current_tasks.json
    2. Find task with id '{task_id}'
    3. Update fields:
       - status: '{status}'
       - updated_at: current timestamp
       - result: '{result}'
       - progress_notes: '{progress_notes}'
       - error_details: '{error_details}'
    4. Update counters based on status change
    5. Use write_file to save updated task list
    6. Log status change to /logs/task_execution.log
    """

@tool
def check_task_dependencies(task_id: str) -> str:
    """Check if task dependencies are satisfied."""
    return f"""
    To check dependencies for {task_id}:
    
    1. Use read_file to load /tasks/current_tasks.json
    2. Find task with id '{task_id}'
    3. For each dependency in task.dependencies:
       - Find dependency task by ID
       - Check if status is 'completed'
    4. Return true if all dependencies completed, false otherwise
    5. List any pending dependencies
    """

@tool
def generate_progress_report() -> str:
    """Generate comprehensive progress report."""
    return """
    To generate progress report:
    
    1. Use read_file to load /tasks/current_tasks.json
    2. Calculate statistics:
       - Total tasks by status
       - Completion percentage
       - Tasks by category and priority
       - Average completion time
    3. Identify:
       - Next executable tasks
       - Blocked tasks and reasons
       - Critical path and bottlenecks
    4. Save report to /results/progress_report.json
    5. Create summary in /results/progress_summary.md
    """

@tool
def implement_task(task_id: str, task_details: Dict) -> str:
    """Implement a specific task based on its category and requirements."""
    
    category = task_details.get('category', 'general')
    
    implementation_guides = {
        "frontend": """
        Frontend Implementation:
        1. Create component structure in /results/{task_id}/components/
        2. Implement React/Vue components with proper props and state
        3. Add styling (CSS/SCSS/styled-components)
        4. Include responsive design and accessibility
        5. Add unit tests for component logic
        6. Create documentation and usage examples
        """,
        
        "backend": """
        Backend Implementation:
        1. Create API structure in /results/{task_id}/api/
        2. Implement route handlers and middleware
        3. Add input validation and error handling
        4. Create business logic and services
        5. Add database integration if needed
        6. Include comprehensive testing
        """,
        
        "authentication": """
        Authentication Implementation:
        1. Create auth structure in /results/{task_id}/auth/
        2. Implement authentication flows (login, register, logout)
        3. Add JWT token management
        4. Create password hashing and validation
        5. Add session management
        6. Include security testing
        """,
        
        "database": """
        Database Implementation:
        1. Create schema in /results/{task_id}/database/
        2. Design database tables and relationships
        3. Create migration scripts
        4. Add indexes and constraints
        5. Include seed data
        6. Add database tests
        """,
        
        "testing": """
        Testing Implementation:
        1. Create test structure in /results/{task_id}/tests/
        2. Write unit tests for core functionality
        3. Add integration tests for workflows
        4. Create end-to-end tests for user journeys
        5. Add performance and security tests
        6. Generate test coverage reports
        """
    }
    
    guide = implementation_guides.get(category, implementation_guides["backend"])
    
    return f"""
    Implementing task {task_id} ({category}):
    
    {guide}
    
    Task Details:
    - Title: {task_details.get('title', 'Unknown')}
    - Description: {task_details.get('description', 'No description')}
    - Acceptance Criteria: {task_details.get('acceptance_criteria', [])}
    
    Use write_file to create implementation files and update task status when complete.
    """

# ============================================================================
# AGENT SETUP
# ============================================================================

def create_long_running_agent(project_dir: str = "./example_project"):
    """Create the complete long-running agent setup."""
    
    # Create project directories
    os.makedirs(f"{project_dir}/.deepagents", exist_ok=True)
    os.makedirs(f"{project_dir}/tasks", exist_ok=True)
    os.makedirs(f"{project_dir}/results", exist_ok=True)
    os.makedirs(f"{project_dir}/memories", exist_ok=True)
    os.makedirs(f"{project_dir}/logs", exist_ok=True)
    
    # Create persistent store
    store = InMemoryStore()
    
    # Configure backend with proper routing
    backend = CompositeBackend(
        default=lambda rt: StateBackend(rt),
        routes={
            "/tasks/": lambda rt: StoreBackend(rt),
            "/memories/": lambda rt: StoreBackend(rt),
            "/results/": lambda rt: FilesystemBackend(f"{project_dir}/results"),
            "/logs/": lambda rt: FilesystemBackend(f"{project_dir}/logs"),
        }
    )
    
    # Create AGENTS.md file
    agents_content = f"""# Long-Running Agent Project Context

## Project Information
- Type: PRD-driven development project
- Agent: Long-running autonomous implementation agent
- Created: {datetime.now().isoformat()}

## Workflow Patterns
- PRD parsing: Structured markdown with automatic task extraction
- Task execution: Dependency-aware autonomous implementation
- State persistence: Cross-session with DeepAgents backends
- Error handling: Graceful degradation with retry logic

## Implementation Standards
- Frontend: React components with TypeScript
- Backend: RESTful APIs with proper validation
- Database: Relational design with migrations
- Testing: Comprehensive unit and integration tests
- Documentation: Inline comments and README files

## Quality Requirements
- Code coverage: Minimum 80%
- Error handling: All edge cases covered
- Security: Input validation and authentication
- Performance: Optimized for production use

## File Organization
- Tasks: /tasks/current_tasks.json
- Results: /results/[task_id]/
- Memory: /memories/[pattern_files]
- Logs: /logs/[execution_logs]

Last updated: {datetime.now().isoformat()}
"""
    
    with open(f"{project_dir}/.deepagents/AGENTS.md", "w") as f:
        f.write(agents_content)
    
    # Create agent with all tools
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[
            parse_prd_to_tasks,
            execute_next_task,
            update_task_status,
            check_task_dependencies,
            generate_progress_report,
            implement_task
        ],
        store=store,
        backend=backend,
        memory=[
            "~/.deepagents/AGENTS.md",
            f"{project_dir}/.deepagents/AGENTS.md"
        ],
        system_prompt="""You are a long-running autonomous agent specialized in processing PRDs and executing implementation tasks.

WORKFLOW:
1. Parse PRD content into structured task lists using parse_prd_to_tasks
2. Save task list to /tasks/current_tasks.json using write_file
3. Continuously execute tasks using execute_next_task
4. Check dependencies with check_task_dependencies before execution
5. Update progress with update_task_status throughout execution
6. Save implementation results to /results/[task_id]/ directories
7. Generate regular progress reports for transparency

EXECUTION RULES:
- Only execute tasks whose dependencies are completed
- Update task status immediately when starting/completing tasks
- Save detailed implementation results and progress notes
- Handle errors gracefully and mark tasks appropriately
- Use filesystem tools extensively for state management
- Learn from execution patterns and save to /memories/

IMPLEMENTATION APPROACH:
- Frontend tasks: Create React components with proper structure
- Backend tasks: Implement APIs with validation and error handling  
- Database tasks: Design schemas with proper relationships
- Authentication tasks: Implement secure auth flows
- Testing tasks: Create comprehensive test suites

PERSISTENCE:
- All task data goes to /tasks/ directory (StoreBackend)
- Implementation results go to /results/ directory (FilesystemBackend)
- Learning patterns go to /memories/ directory (StoreBackend)
- Execution logs go to /logs/ directory (FilesystemBackend)

You have access to all DeepAgents filesystem tools automatically. Use them extensively."""
    )
    
    return agent

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def main():
    """Demonstrate complete PRD-to-implementation workflow."""
    
    # Sample PRD content
    sample_prd = """
# E-commerce User Management System

## User Registration
Implement user registration functionality with email verification
- Email validation and uniqueness check
- Password strength requirements (8+ chars, mixed case, numbers)
- Email verification flow with secure tokens
- User profile creation with basic information

## User Authentication  
Secure login system with session management
- Email/password authentication
- JWT token generation and validation
- Session timeout and refresh logic
- Remember me functionality
- Password reset via email

## User Profile Management
Allow users to view and update their profile information
- Display current profile information
- Edit profile form with validation
- Profile picture upload and management
- Account settings and preferences
- Account deactivation option

## Admin User Management
Administrative interface for managing users
- View all users with filtering and search
- User account status management (active/inactive/banned)
- User role assignment (user/admin/moderator)
- Bulk user operations
- User activity audit logs
"""
    
    print("🚀 Creating Long-Running Agent...")
    agent = create_long_running_agent("./example_project")
    
    print("📋 Processing PRD and starting autonomous execution...")
    
    # Configuration for persistent session
    config = {"configurable": {"thread_id": "ecommerce-user-mgmt-001"}}
    
    # Start the workflow
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""Process this PRD and begin autonomous task execution:

{sample_prd}

Please:
1. Parse the PRD into a structured task list
2. Save the task list to /tasks/current_tasks.json  
3. Begin executing tasks autonomously
4. Update progress as you work through each task
5. Generate a progress report when you've made significant progress

Work through the tasks systematically, respecting dependencies and priorities."""
        }]
    }, config=config)
    
    print("\\n✅ Initial processing complete!")
    print("\\n📊 Agent Response:")
    print(result["messages"][-1].content)
    
    # Continue execution in subsequent sessions
    print("\\n🔄 Continuing execution (simulating session resumption)...")
    
    continuation_result = agent.invoke({
        "messages": [{
            "role": "user", 
            "content": "Continue executing the remaining tasks and provide a progress update."
        }]
    }, config=config)
    
    print("\\n📈 Continuation Response:")
    print(continuation_result["messages"][-1].content)
    
    # Generate final report
    print("\\n📋 Generating final progress report...")
    
    report_result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Generate a comprehensive progress report showing all completed tasks, remaining work, and project status."
        }]
    }, config=config)
    
    print("\\n📊 Final Report:")
    print(report_result["messages"][-1].content)
    
    print("\\n🎉 Example workflow complete!")
    print("\\nCheck the following directories for results:")
    print("- ./example_project/tasks/ - Task lists and status")
    print("- ./example_project/results/ - Implementation results")  
    print("- ./example_project/memories/ - Learned patterns")
    print("- ./example_project/logs/ - Execution logs")

if __name__ == "__main__":
    main()