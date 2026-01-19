# Task Execution Patterns

## Overview

This reference covers autonomous task execution patterns using DeepAgents, including dependency management, status tracking, and cross-session persistence.

## Core Task Execution Engine

```python
from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@tool
def execute_next_task() -> str:
    """Find and execute the next available task based on dependencies and priority."""
    
    try:
        # This tool coordinates with DeepAgents filesystem tools
        instruction = """
        1. Use read_file to load /tasks/current_tasks.json
        2. Find next executable task (status=pending, dependencies met)
        3. Update task status to in_progress
        4. Execute the task using appropriate implementation approach
        5. Update task with results and completion status
        6. Save updated task list back to /tasks/current_tasks.json
        
        Priority order: high -> medium -> low
        Within same priority: consider dependencies and creation order
        """
        
        return instruction
        
    except Exception as e:
        return f"Error in task execution coordination: {str(e)}"

@tool
def check_task_dependencies(task_id: str) -> str:
    """Check if all dependencies for a task are completed."""
    
    return f"""
    To check dependencies for {task_id}:
    1. Use read_file to load /tasks/current_tasks.json
    2. Find task with id '{task_id}'
    3. Check each dependency ID in the dependencies list
    4. Verify all dependency tasks have status 'completed'
    5. Return true if all dependencies met, false otherwise
    
    If dependencies not met, list which tasks are still pending/in-progress.
    """

@tool
def update_task_status(task_id: str, status: str, result: str = "", progress_notes: str = "", error_details: str = "") -> str:
    """Update task status and save progress information."""
    
    if status not in [s.value for s in TaskStatus]:
        return f"Invalid status '{status}'. Valid statuses: {[s.value for s in TaskStatus]}"
    
    update_instruction = f"""
    To update task {task_id}:
    1. Use read_file to load /tasks/current_tasks.json
    2. Find task with id '{task_id}'
    3. Update the following fields:
       - status: '{status}'
       - updated_at: current timestamp
       - result: '{result}' (if provided)
       - progress_notes: '{progress_notes}' (if provided)
       - error_details: '{error_details}' (if status is failed)
    4. If status is 'completed', increment completed_tasks count
    5. If status is 'failed', increment failed_tasks count  
    6. If status is 'in_progress', increment in_progress_tasks count
    7. Update corresponding pending_tasks count
    8. Use write_file to save updated task list to /tasks/current_tasks.json
    
    Also log the status change to /logs/task_execution.log
    """
    
    return update_instruction

@tool
def get_executable_tasks() -> str:
    """Get list of tasks that can be executed (dependencies met, status pending)."""
    
    return """
    To get executable tasks:
    1. Use read_file to load /tasks/current_tasks.json
    2. Filter tasks where:
       - status = 'pending'
       - All dependencies have status = 'completed' (or no dependencies)
    3. Sort by priority (high, medium, low) then by created_at
    4. Return list of executable task IDs and titles
    """

@tool
def execute_specific_task(task_id: str, implementation_approach: str = "auto") -> str:
    """Execute a specific task with given approach."""
    
    return f"""
    To execute task {task_id}:
    
    1. PREPARATION:
       - Use read_file to load task details from /tasks/current_tasks.json
       - Update status to 'in_progress' using update_task_status
       - Create results directory: /results/{task_id}/
    
    2. IMPLEMENTATION:
       Based on task category and description, choose implementation approach:
       
       FRONTEND TASKS:
       - Create component files (.tsx, .jsx, .vue)
       - Implement UI logic and styling
       - Add tests and documentation
       
       BACKEND TASKS:
       - Create API endpoints and handlers
       - Implement business logic
       - Add database models/migrations
       - Write unit tests
       
       DATABASE TASKS:
       - Create migration scripts
       - Update schema definitions
       - Add seed data if needed
       
       TESTING TASKS:
       - Write test cases
       - Implement test automation
       - Create test data and fixtures
       
       DOCUMENTATION TASKS:
       - Write technical documentation
       - Create user guides
       - Update README files
    
    3. VALIDATION:
       - Verify implementation meets acceptance criteria
       - Run relevant tests
       - Check for integration issues
    
    4. COMPLETION:
       - Save implementation files to /results/{task_id}/
       - Update task status to 'completed' with detailed results
       - Log completion in /logs/task_execution.log
    
    Use appropriate DeepAgents tools (write_file, execute, etc.) for implementation.
    """
```

## Task Implementation Patterns

### Frontend Task Implementation

```python
@tool
def implement_frontend_task(task_id: str, task_details: Dict) -> str:
    """Implement frontend-specific tasks (UI components, pages, etc.)."""
    
    return f"""
    Frontend implementation for {task_id}:
    
    1. COMPONENT STRUCTURE:
       - Create /results/{task_id}/components/ directory
       - Implement main component file
       - Add supporting utilities and hooks
       - Create style files (CSS/SCSS/styled-components)
    
    2. IMPLEMENTATION STEPS:
       - Parse task requirements and acceptance criteria
       - Design component interface and props
       - Implement core functionality
       - Add error handling and loading states
       - Implement responsive design
       - Add accessibility features
    
    3. TESTING:
       - Create unit tests for component logic
       - Add integration tests for user interactions
       - Test responsive behavior
       - Validate accessibility compliance
    
    4. DOCUMENTATION:
       - Document component API and usage
       - Add code comments for complex logic
       - Create usage examples
    
    Example structure:
    /results/{task_id}/
    ├── components/
    │   ├── ComponentName.tsx
    │   ├── ComponentName.test.tsx
    │   ├── ComponentName.stories.tsx
    │   └── index.ts
    ├── styles/
    │   └── ComponentName.styles.ts
    ├── hooks/
    │   └── useComponentLogic.ts
    └── README.md
    """

@tool
def implement_backend_task(task_id: str, task_details: Dict) -> str:
    """Implement backend-specific tasks (APIs, services, etc.)."""
    
    return f"""
    Backend implementation for {task_id}:
    
    1. API STRUCTURE:
       - Create /results/{task_id}/api/ directory
       - Implement route handlers
       - Add middleware and validation
       - Create service layer logic
    
    2. DATABASE INTEGRATION:
       - Create/update database models
       - Add migration scripts
       - Implement data access layer
       - Add database indexes and constraints
    
    3. BUSINESS LOGIC:
       - Implement core business rules
       - Add input validation and sanitization
       - Handle error cases and edge conditions
       - Add logging and monitoring
    
    4. TESTING:
       - Unit tests for business logic
       - Integration tests for API endpoints
       - Database integration tests
       - Performance and load testing
    
    5. SECURITY:
       - Add authentication/authorization
       - Implement input validation
       - Add rate limiting
       - Secure sensitive data
    
    Example structure:
    /results/{task_id}/
    ├── api/
    │   ├── routes/
    │   ├── controllers/
    │   ├── middleware/
    │   └── validators/
    ├── services/
    ├── models/
    ├── migrations/
    ├── tests/
    └── README.md
    """
```

## Dependency Management

```python
@tool
def resolve_task_dependencies(task_id: str) -> str:
    """Analyze and resolve task dependencies before execution."""
    
    return f"""
    Dependency resolution for {task_id}:
    
    1. DEPENDENCY ANALYSIS:
       - Load task details from /tasks/current_tasks.json
       - Check each dependency task status
       - Identify blocking dependencies
       - Calculate dependency chain depth
    
    2. DEPENDENCY TYPES:
       
       HARD DEPENDENCIES (must be completed):
       - API endpoints before frontend integration
       - Database models before business logic
       - Authentication before protected features
       
       SOFT DEPENDENCIES (preferred order):
       - Basic features before advanced features
       - Core components before specialized components
       - Setup tasks before implementation tasks
    
    3. RESOLUTION STRATEGIES:
       
       IF ALL DEPENDENCIES MET:
       - Proceed with task execution
       
       IF DEPENDENCIES PENDING:
       - Execute dependency tasks first (recursive)
       - Update task status to 'blocked'
       - Schedule for later execution
       
       IF CIRCULAR DEPENDENCIES:
       - Break circular dependencies by identifying optional links
       - Restructure tasks to remove cycles
       - Log dependency resolution changes
    
    4. DEPENDENCY TRACKING:
       - Log dependency resolution in /logs/dependency_resolution.log
       - Update task metadata with dependency status
       - Track dependency execution time and success rates
    """

@tool
def optimize_task_execution_order() -> str:
    """Optimize the order of task execution based on dependencies and priorities."""
    
    return """
    Task execution optimization:
    
    1. LOAD AND ANALYZE:
       - Read all pending tasks from /tasks/current_tasks.json
       - Build dependency graph
       - Calculate task priorities and effort estimates
    
    2. OPTIMIZATION ALGORITHMS:
       
       TOPOLOGICAL SORT:
       - Order tasks respecting dependencies
       - Ensure no task executes before its dependencies
       
       PRIORITY WEIGHTING:
       - High priority tasks get preference
       - Balance priority vs dependency constraints
       
       PARALLEL EXECUTION OPPORTUNITIES:
       - Identify tasks that can run concurrently
       - Group independent tasks for parallel processing
       
       EFFORT BALANCING:
       - Mix large and small tasks for steady progress
       - Avoid clustering all large tasks together
    
    3. EXECUTION PLAN:
       - Generate optimized execution sequence
       - Save plan to /tasks/execution_plan.json
       - Include parallel execution groups
       - Estimate total completion time
    
    4. ADAPTIVE REPLANNING:
       - Replan when tasks fail or change
       - Adjust for actual vs estimated execution times
       - Update plan based on new task additions
    """
```

## Progress Tracking and Reporting

```python
@tool
def generate_progress_report() -> str:
    """Generate comprehensive progress report for current project."""
    
    return """
    Progress report generation:
    
    1. TASK STATISTICS:
       - Total tasks: count by status
       - Completion percentage
       - Average task completion time
       - Success/failure rates
    
    2. TIMELINE ANALYSIS:
       - Tasks completed per day/week
       - Velocity trends
       - Estimated completion date
       - Milestone progress
    
    3. DEPENDENCY INSIGHTS:
       - Critical path analysis
       - Bottleneck identification
       - Dependency resolution efficiency
    
    4. QUALITY METRICS:
       - Task rework rates
       - Error frequency by category
       - Test coverage and pass rates
    
    5. REPORT OUTPUT:
       Save comprehensive report to /results/progress_report.json
       Include charts and visualizations in /results/charts/
       Generate executive summary in /results/executive_summary.md
    """

@tool
def track_task_metrics(task_id: str, metrics: Dict) -> str:
    """Track detailed metrics for task execution."""
    
    return f"""
    Metrics tracking for {task_id}:
    
    EXECUTION METRICS:
    - Start time and end time
    - Actual vs estimated effort
    - Number of iterations/revisions
    - Error count and types
    
    QUALITY METRICS:
    - Code coverage percentage
    - Test pass rate
    - Performance benchmarks
    - Security scan results
    
    PROCESS METRICS:
    - Dependency wait time
    - Review and approval time
    - Deployment success rate
    - User acceptance feedback
    
    Save metrics to /logs/task_metrics.json with timestamp and task context.
    """
```

## Error Handling and Recovery

```python
@tool
def handle_task_failure(task_id: str, error_details: str, recovery_strategy: str = "auto") -> str:
    """Handle task failure with appropriate recovery strategy."""
    
    return f"""
    Failure handling for {task_id}:
    
    1. ERROR CLASSIFICATION:
       
       TRANSIENT ERRORS (retry):
       - Network timeouts
       - Temporary resource unavailability
       - Rate limiting
       
       CONFIGURATION ERRORS (fix and retry):
       - Missing environment variables
       - Incorrect API endpoints
       - Permission issues
       
       LOGIC ERRORS (redesign):
       - Incorrect implementation approach
       - Misunderstood requirements
       - Design flaws
       
       DEPENDENCY ERRORS (wait and retry):
       - Dependency task failures
       - External service outages
       - Database connectivity issues
    
    2. RECOVERY STRATEGIES:
       
       AUTO RETRY:
       - Exponential backoff for transient errors
       - Maximum retry count: 3
       - Log retry attempts
       
       MANUAL INTERVENTION:
       - Mark task as 'failed' with detailed error info
       - Create follow-up task for manual resolution
       - Notify relevant stakeholders
       
       GRACEFUL DEGRADATION:
       - Implement simplified version
       - Mark as partial completion
       - Plan enhancement in future iteration
    
    3. FAILURE TRACKING:
       - Log failure details in /logs/task_failures.json
       - Update task status and error information
       - Analyze failure patterns for prevention
       - Update error handling strategies based on patterns
    """

@tool
def implement_task_rollback(task_id: str, rollback_reason: str) -> str:
    """Rollback task changes if implementation causes issues."""
    
    return f"""
    Rollback procedure for {task_id}:
    
    1. ROLLBACK SCOPE:
       - Identify all files and changes made during task execution
       - Check for dependent tasks that might be affected
       - Assess impact on overall project state
    
    2. ROLLBACK EXECUTION:
       - Revert code changes to pre-task state
       - Remove created files and directories
       - Restore database to previous state (if applicable)
       - Update task status to 'cancelled' or 'failed'
    
    3. CLEANUP:
       - Remove task results from /results/{task_id}/
       - Update task dependencies for affected tasks
       - Log rollback details and reasons
    
    4. RECOVERY PLANNING:
       - Analyze rollback reasons
       - Create new task with corrected approach
       - Update project timeline and estimates
    """
```

## Integration with DeepAgents Memory

```python
@tool
def learn_from_task_execution(task_id: str, execution_results: Dict) -> str:
    """Learn from task execution to improve future performance."""
    
    return f"""
    Learning from {task_id} execution:
    
    1. PATTERN EXTRACTION:
       - Successful implementation patterns
       - Common error patterns and solutions
       - Effective dependency resolution strategies
       - Optimal task breakdown approaches
    
    2. KNOWLEDGE STORAGE:
       Save learnings to /memories/:
       - /memories/implementation_patterns.md
       - /memories/error_solutions.md
       - /memories/dependency_strategies.md
       - /memories/task_templates.json
    
    3. TEMPLATE GENERATION:
       - Create reusable task templates
       - Document best practices
       - Build solution libraries
       - Update estimation models
    
    4. CONTINUOUS IMPROVEMENT:
       - Update parsing accuracy based on results
       - Refine dependency detection algorithms
       - Improve error prediction models
       - Enhance automation capabilities
    """
```

## Usage Examples

### Basic Task Execution Flow

```python
# Execute next available task
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Execute the next available task from the task list"
    }]
})

# The agent will:
# 1. Load /tasks/current_tasks.json
# 2. Find executable task (dependencies met)
# 3. Update status to in_progress
# 4. Implement the task
# 5. Save results and update status to completed
```

### Specific Task Execution

```python
# Execute specific task by ID
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Execute task_001 (User Registration API) using backend implementation approach"
    }]
})
```

### Progress Monitoring

```python
# Generate progress report
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Generate a progress report showing current project status and completion estimates"
    }]
})
```