#!/usr/bin/env python3
"""
Universal Long-Running Agent Example
===================================

This example demonstrates the agent-agnostic long-running agent that works with
any AI agent framework (Cursor, OpenCode, Claude, etc.).

The agent can:
1. Parse PRD/specifications into structured task lists
2. Execute tasks autonomously with dependency management
3. Manage API rotation with intelligent load balancing and rate limiting
4. Persist state across sessions using standard JSON files
5. Learn from execution patterns for continuous improvement

Setup with uv (recommended):
    # Install uv if not already installed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Create virtual environment (optional - no external dependencies)
    uv venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    
    # Run the example
    uv run python universal_example.py

Traditional usage:
    python universal_example.py

Dependencies:
    None! This example uses only Python standard library for maximum compatibility.
"""

import json
import re
import os
import threading
import concurrent.futures
import time
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

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
            
            # Find available endpoints
            available_endpoints = [ep for ep in self.endpoints if ep.can_make_request()]
            
            if not available_endpoints:
                print("⚠️ All API endpoints are rate limited or at quota")
                return None
            
            # Select endpoint using weighted selection based on remaining capacity
            return self._select_optimal_endpoint(available_endpoints)
    
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
        
        if not weights or sum(weights) == 0:
            return available_endpoints[0]
        
        # Weighted random selection
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for (endpoint, weight) in top_endpoints:
            current_weight += weight
            if rand_val <= current_weight:
                return endpoint
        
        return top_endpoints[0][0]  # Fallback
    
    def simulate_api_request(self, endpoint: APIEndpoint, request_type: str) -> Dict:
        """Simulate an API request for demonstration purposes."""
        
        # Record the request
        endpoint.record_request(success=True)
        self.usage_stats["total_requests"] += 1
        self.usage_stats["successful_requests"] += 1
        
        # Simulate response based on request type
        if request_type == "completion":
            return {
                "success": True,
                "data": {
                    "choices": [{"text": f"Generated response from {endpoint.name}"}],
                    "usage": {"tokens": 150}
                },
                "endpoint_used": endpoint.name
            }
        elif request_type == "embedding":
            return {
                "success": True,
                "data": {
                    "embeddings": [[0.1, 0.2, 0.3] * 100],  # Mock embedding
                    "usage": {"tokens": 50}
                },
                "endpoint_used": endpoint.name
            }
        else:
            return {
                "success": True,
                "data": {"result": f"API response from {endpoint.name}"},
                "endpoint_used": endpoint.name
            }
    
    def make_api_request(self, request_type: str = "completion") -> Dict:
        """Make an API request with automatic rotation."""
        
        endpoint = self.get_best_endpoint()
        
        if not endpoint:
            return {
                "success": False,
                "error": "No API endpoints available",
                "usage_stats": self._get_usage_summary()
            }
        
        print(f"🔄 Using endpoint: {endpoint.name} (usage: {endpoint.current_usage}/{endpoint.rate_limit})")
        
        # Simulate the API request
        result = self.simulate_api_request(endpoint, request_type)
        
        return result
    
    def _get_usage_summary(self) -> Dict:
        """Get current usage summary across all endpoints."""
        return {
            "total_requests": self.usage_stats["total_requests"],
            "success_rate": (self.usage_stats["successful_requests"] / 
                           max(1, self.usage_stats["total_requests"]) * 100),
            "active_endpoints": len([ep for ep in self.endpoints if ep.status == APIStatus.ACTIVE]),
            "total_endpoints": len(self.endpoints)
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

# Global API rotation manager instance
api_manager = APIRotationManager()

# ============================================================================
# CORE FUNCTIONS (Agent-Agnostic)
# ============================================================================

def parse_prd_to_tasks(prd_content: str, project_name: str = "Generated Project") -> Dict:
    """Parse PRD/specification content into structured task list."""
    tasks = []
    task_id_counter = 1
    
    # Split content into sections by headers
    sections = re.split(r'\\n(?=#{2,3}\\s)', prd_content)
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\\n')
        if not lines:
            continue
            
        # Extract title from header
        header_match = re.match(r'^#{2,3}\\s+(.+)', lines[0])
        if not header_match:
            continue
            
        title = header_match.group(1).strip()
        description_lines = lines[1:] if len(lines) > 1 else []
        
        # Extract acceptance criteria (bullet points)
        acceptance_criteria = []
        remaining_description = []
        
        for line in description_lines:
            line = line.strip()
            if re.match(r'^[-*]\\s+|^\\d+\\.\\s+', line):
                criteria = re.sub(r'^[-*]\\s+|^\\d+\\.\\s+', '', line)
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
    dependency_rules = [
        ('authentication', 'frontend'),
        ('backend', 'frontend'),
        ('database', 'backend'),
        ('database', 'authentication'),
    ]
    
    for i, task in enumerate(tasks):
        dependencies = []
        
        for j, other_task in enumerate(tasks[:i]):
            for prereq_category, dependent_category in dependency_rules:
                if (prereq_category in other_task['category'] and 
                    dependent_category in task['category']):
                    dependencies.append(other_task['id'])
        
        task['dependencies'] = list(set(dependencies))
    
    return tasks

# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def save_task_list(task_list: Dict, file_path: str = "tasks/current_tasks.json") -> bool:
    """Save task list to persistent storage."""
    try:
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
                
                update_task_counters(task_list, old_status, status)
                break
        
        return save_task_list(task_list)
        
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False

def update_task_counters(task_list: Dict, old_status: str, new_status: str):
    """Update task counters when status changes."""
    # Initialize counters if missing
    for counter in ["pending_tasks", "in_progress_tasks", "completed_tasks", "failed_tasks"]:
        if counter not in task_list:
            task_list[counter] = 0
    
    # Decrement old status counter
    if old_status == "pending":
        task_list["pending_tasks"] = max(0, task_list["pending_tasks"] - 1)
    elif old_status == "in_progress":
        task_list["in_progress_tasks"] = max(0, task_list["in_progress_tasks"] - 1)
    elif old_status == "completed":
        task_list["completed_tasks"] = max(0, task_list["completed_tasks"] - 1)
    elif old_status == "failed":
        task_list["failed_tasks"] = max(0, task_list["failed_tasks"] - 1)
    
    # Increment new status counter
    if new_status == "pending":
        task_list["pending_tasks"] += 1
    elif new_status == "in_progress":
        task_list["in_progress_tasks"] += 1
    elif new_status == "completed":
        task_list["completed_tasks"] += 1
    elif new_status == "failed":
        task_list["failed_tasks"] += 1

# ============================================================================
# TASK EXECUTION ENGINE
# ============================================================================

def get_executable_tasks() -> List[Dict]:
    """Get list of tasks that can be executed."""
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
    
    # Sort by priority then by creation time
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
            return {"success": True, "message": f"Task {task_id} ({task['title']}) completed", "result": result}
        else:
            update_task_status(task_id, "failed", "", "", result.get("error", "Unknown error"))
            return {"success": False, "message": f"Task {task_id} failed", "error": result.get("error")}
            
    except Exception as e:
        update_task_status(task_id, "failed", "", "", str(e))
        return {"success": False, "message": f"Task {task_id} failed with exception", "error": str(e)}

def execute_task_by_category(task: Dict) -> Dict:
    """Execute a task based on its category."""
    category = task.get("category", "general")
    task_id = task["id"]
    
    # Create results directory
    results_dir = f"results/{task_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Category-specific implementation
    if category == "frontend":
        return implement_frontend_task(task, results_dir)
    elif category == "backend":
        return implement_backend_task(task, results_dir)
    elif category == "database":
        return implement_database_task(task, results_dir)
    elif category == "authentication":
        return implement_auth_task(task, results_dir)
    elif category == "testing":
        return implement_testing_task(task, results_dir)
    elif category == "documentation":
        return implement_documentation_task(task, results_dir)
    else:
        return implement_general_task(task, results_dir)

def implement_frontend_task(task: Dict, results_dir: str) -> Dict:
    """Implement frontend-specific tasks."""
    
    # Create frontend structure
    os.makedirs(f"{results_dir}/components", exist_ok=True)
    os.makedirs(f"{results_dir}/styles", exist_ok=True)
    os.makedirs(f"{results_dir}/tests", exist_ok=True)
    
    # Generate component implementation
    component_name = task["title"].replace(" ", "")
    
    component_code = f"""
import React from 'react';
import './{component_name}.css';

interface {component_name}Props {{
  // Define props based on task requirements
}}

const {component_name}: React.FC<{component_name}Props> = (props) => {{
  return (
    <div className="{component_name.lower()}">
      <h2>{task["title"]}</h2>
      <p>{task["description"]}</p>
      {/* Implementation based on acceptance criteria */}
    </div>
  );
}};

export default {component_name};
"""
    
    # Save component file
    with open(f"{results_dir}/components/{component_name}.tsx", "w") as f:
        f.write(component_code)
    
    # Generate CSS
    css_code = f"""
.{component_name.lower()} {{
  /* Styles for {task["title"]} */
  padding: 1rem;
  border: 1px solid #ddd;
  border-radius: 4px;
}}

.{component_name.lower()} h2 {{
  margin-top: 0;
  color: #333;
}}
"""
    
    with open(f"{results_dir}/styles/{component_name}.css", "w") as f:
        f.write(css_code)
    
    # Generate test file
    test_code = f"""
import {{ render, screen }} from '@testing-library/react';
import {component_name} from '../components/{component_name}';

describe('{component_name}', () => {{
  test('renders component title', () => {{
    render(<{component_name} />);
    expect(screen.getByText('{task["title"]}')).toBeInTheDocument();
  }});
  
  // Add more tests based on acceptance criteria
}});
"""
    
    with open(f"{results_dir}/tests/{component_name}.test.tsx", "w") as f:
        f.write(test_code)
    
    # Create README
    readme_content = create_task_readme(task, "Frontend Component", [
        f"components/{component_name}.tsx - Main component implementation",
        f"styles/{component_name}.css - Component styles", 
        f"tests/{component_name}.test.tsx - Unit tests"
    ])
    
    with open(f"{results_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    return {
        "success": True,
        "output": f"Frontend component '{component_name}' implemented successfully",
        "notes": f"Created React component with styles and tests in {results_dir}",
        "files_created": [
            f"components/{component_name}.tsx",
            f"styles/{component_name}.css",
            f"tests/{component_name}.test.tsx",
            "README.md"
        ]
    }

def implement_backend_task(task: Dict, results_dir: str) -> Dict:
    """Implement backend-specific tasks."""
    
    # Create backend structure
    os.makedirs(f"{results_dir}/routes", exist_ok=True)
    os.makedirs(f"{results_dir}/controllers", exist_ok=True)
    os.makedirs(f"{results_dir}/services", exist_ok=True)
    os.makedirs(f"{results_dir}/tests", exist_ok=True)
    
    # Generate API implementation
    endpoint_name = task["title"].lower().replace(" ", "_")
    
    # Route definition
    route_code = f"""
from flask import Blueprint, request, jsonify
from .controllers.{endpoint_name}_controller import {endpoint_name.title().replace('_', '')}Controller

{endpoint_name}_bp = Blueprint('{endpoint_name}', __name__)
controller = {endpoint_name.title().replace('_', '')}Controller()

@{endpoint_name}_bp.route('/{endpoint_name}', methods=['GET'])
def get_{endpoint_name}():
    \"\"\"
    {task["description"]}
    \"\"\"
    try:
        result = controller.get_{endpoint_name}()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({{"error": str(e)}}), 500

@{endpoint_name}_bp.route('/{endpoint_name}', methods=['POST'])
def create_{endpoint_name}():
    \"\"\"
    Create new {endpoint_name}
    \"\"\"
    try:
        data = request.get_json()
        result = controller.create_{endpoint_name}(data)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({{"error": str(e)}}), 400
"""
    
    with open(f"{results_dir}/routes/{endpoint_name}_routes.py", "w") as f:
        f.write(route_code)
    
    # Controller implementation
    controller_code = f"""
from ..services.{endpoint_name}_service import {endpoint_name.title().replace('_', '')}Service

class {endpoint_name.title().replace('_', '')}Controller:
    def __init__(self):
        self.service = {endpoint_name.title().replace('_', '')}Service()
    
    def get_{endpoint_name}(self):
        \"\"\"
        {task["description"]}
        \"\"\"
        return self.service.get_all()
    
    def create_{endpoint_name}(self, data):
        \"\"\"
        Create new {endpoint_name}
        \"\"\"
        # Validate input data
        if not data:
            raise ValueError("No data provided")
        
        return self.service.create(data)
"""
    
    with open(f"{results_dir}/controllers/{endpoint_name}_controller.py", "w") as f:
        f.write(controller_code)
    
    # Service implementation
    service_code = f"""
class {endpoint_name.title().replace('_', '')}Service:
    def __init__(self):
        # Initialize database connection or other dependencies
        pass
    
    def get_all(self):
        \"\"\"
        Retrieve all {endpoint_name} records
        \"\"\"
        # Implementation based on task requirements
        return []
    
    def create(self, data):
        \"\"\"
        Create new {endpoint_name} record
        \"\"\"
        # Validate and create record
        # Implementation based on acceptance criteria
        return {{"id": 1, "message": "{endpoint_name} created successfully"}}
"""
    
    with open(f"{results_dir}/services/{endpoint_name}_service.py", "w") as f:
        f.write(service_code)
    
    # Generate tests
    test_code = f"""
import pytest
from unittest.mock import Mock
from ..controllers.{endpoint_name}_controller import {endpoint_name.title().replace('_', '')}Controller

class Test{endpoint_name.title().replace('_', '')}Controller:
    def setup_method(self):
        self.controller = {endpoint_name.title().replace('_', '')}Controller()
    
    def test_get_{endpoint_name}(self):
        result = self.controller.get_{endpoint_name}()
        assert isinstance(result, list)
    
    def test_create_{endpoint_name}(self):
        test_data = {{"name": "test"}}
        result = self.controller.create_{endpoint_name}(test_data)
        assert "id" in result
        assert result["message"] == "{endpoint_name} created successfully"
"""
    
    with open(f"{results_dir}/tests/test_{endpoint_name}.py", "w") as f:
        f.write(test_code)
    
    # Create README
    readme_content = create_task_readme(task, "Backend API", [
        f"routes/{endpoint_name}_routes.py - API route definitions",
        f"controllers/{endpoint_name}_controller.py - Request handling logic",
        f"services/{endpoint_name}_service.py - Business logic implementation",
        f"tests/test_{endpoint_name}.py - Unit tests"
    ])
    
    with open(f"{results_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    return {
        "success": True,
        "output": f"Backend API for '{task['title']}' implemented successfully",
        "notes": f"Created REST API with routes, controllers, services, and tests in {results_dir}",
        "files_created": [
            f"routes/{endpoint_name}_routes.py",
            f"controllers/{endpoint_name}_controller.py",
            f"services/{endpoint_name}_service.py",
            f"tests/test_{endpoint_name}.py",
            "README.md"
        ]
    }

def implement_general_task(task: Dict, results_dir: str) -> Dict:
    """Implement general tasks."""
    
    readme_content = create_task_readme(task, "General Implementation", [
        "README.md - Task implementation documentation"
    ])
    
    with open(f"{results_dir}/README.md", "w") as f:
        f.write(readme_content)
    
    return {
        "success": True,
        "output": f"General task '{task['title']}' documented and structured",
        "notes": f"Created implementation structure and documentation in {results_dir}",
        "files_created": ["README.md"]
    }

# Placeholder implementations for other categories
def implement_database_task(task: Dict, results_dir: str) -> Dict:
    return implement_general_task(task, results_dir)

def implement_auth_task(task: Dict, results_dir: str) -> Dict:
    return implement_general_task(task, results_dir)

def implement_testing_task(task: Dict, results_dir: str) -> Dict:
    return implement_general_task(task, results_dir)

def implement_documentation_task(task: Dict, results_dir: str) -> Dict:
    return implement_general_task(task, results_dir)

def create_task_readme(task: Dict, implementation_type: str, files_created: List[str]) -> str:
    """Create README content for task implementation."""
    
    criteria_list = "\\n".join(f"- {criteria}" for criteria in task.get("acceptance_criteria", []))
    files_list = "\\n".join(f"- {file}" for file in files_created)
    
    return f"""# {task["title"]} - {implementation_type}

## Task Information
- **ID**: {task["id"]}
- **Category**: {task.get("category", "general")}
- **Priority**: {task.get("priority", "medium")}
- **Estimated Effort**: {task.get("estimated_effort", "medium")}

## Description
{task["description"]}

## Acceptance Criteria
{criteria_list if criteria_list else "- No specific criteria provided"}

## Implementation Details
- **Implementation Type**: {implementation_type}
- **Status**: {task["status"]}
- **Created**: {task.get("created_at", "Unknown")}
- **Updated**: {task.get("updated_at", "Unknown")}

## Files Created
{files_list}

## Dependencies
{", ".join(task.get("dependencies", [])) if task.get("dependencies") else "None"}

## Results
{task.get("result", "Implementation completed successfully")}

## Progress Notes
{task.get("progress_notes", "Task completed as specified")}

---
*Generated by Long-Running Agent - Universal Example*
"""

# ============================================================================
# PROGRESS REPORTING
# ============================================================================

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
    os.makedirs("results", exist_ok=True)
    with open("results/progress_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

# ============================================================================
# MAIN WORKFLOW FUNCTION
# ============================================================================

def universal_long_running_agent_workflow(prd_content: str, project_name: str = "Universal Agent Project"):
    """
    Universal workflow that works with any AI agent.
    
    This function demonstrates the complete PRD-to-implementation workflow
    using only standard Python and JSON files for persistence.
    """
    
    print(f"🚀 Starting Universal Long-Running Agent Workflow")
    print(f"📋 Project: {project_name}")
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Step 1: Setup project structure
    print("📁 Setting up project structure...")
    directories = ["tasks", "results", "memories", "logs"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ Project structure created")
    
    # Step 2: Parse PRD into tasks
    print("\\n📋 Parsing PRD into structured tasks...")
    task_list = parse_prd_to_tasks(prd_content, project_name)
    
    if not save_task_list(task_list):
        print("❌ Failed to save task list")
        return False
    
    print(f"✅ Generated {task_list['total_tasks']} tasks:")
    for task in task_list["tasks"]:
        print(f"   - {task['id']}: {task['title']} ({task['category']}, {task['priority']} priority)")
    
    # Step 3: Execute tasks autonomously
    print("\\n🔄 Beginning autonomous task execution...")
    completed_count = 0
    failed_count = 0
    
    while True:
        result = execute_next_task()
        
        if not result["success"]:
            if "No executable tasks" in result["message"]:
                print(f"\\n🎉 All executable tasks completed!")
                break
            else:
                failed_count += 1
                print(f"⚠️ Task failed: {result['message']}")
                
                # Continue with next task (graceful degradation)
                if failed_count >= 3:
                    print("❌ Too many consecutive failures, stopping execution")
                    break
                continue
        else:
            completed_count += 1
            failed_count = 0  # Reset failure count on success
            print(f"✅ Task {completed_count} completed: {result['message']}")
            
            # Show progress every 3 tasks
            if completed_count % 3 == 0:
                report = get_progress_report()
                progress = report["overall_progress"]["completion_percentage"]
                print(f"📊 Progress Update: {progress}% complete ({completed_count} tasks done)")
    
    # Step 4: Generate final report
    print("\\n📈 Generating final progress report...")
    final_report = get_progress_report()
    
    print("\\n" + "=" * 60)
    print("🎯 FINAL PROJECT SUMMARY")
    print("=" * 60)
    print(f"Project: {final_report['project_name']}")
    print(f"Total Tasks: {final_report['overall_progress']['total_tasks']}")
    print(f"Completed: {final_report['overall_progress']['completed_tasks']}")
    print(f"Failed: {final_report['overall_progress']['failed_tasks']}")
    print(f"Success Rate: {final_report['overall_progress']['completion_percentage']}%")
    print(f"\\n📁 Results Location:")
    print(f"   - Task List: ./tasks/current_tasks.json")
    print(f"   - Implementation Results: ./results/")
    print(f"   - Progress Report: ./results/progress_report.json")
    
    # Show category breakdown
    if final_report["category_breakdown"]:
        print(f"\\n📊 Category Breakdown:")
        for category, stats in final_report["category_breakdown"].items():
            success_rate = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"   - {category.title()}: {stats['completed']}/{stats['total']} ({success_rate:.1f}%)")
    
    print("\\n🎉 Universal Long-Running Agent workflow completed!")
    return True

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def setup_api_rotation_demo():
    """Setup API rotation for demonstration."""
    global api_manager
    
    print("\n🔄 Setting up API rotation...")
    
    # Add demo API endpoints
    api_manager.add_endpoint(
        name="openai_primary",
        base_url="https://api.openai.com/v1",
        api_key="sk-demo-primary-key",
        rate_limit=100,
        quota_limit=10000
    )
    
    api_manager.add_endpoint(
        name="openai_backup",
        base_url="https://api.openai.com/v1",
        api_key="sk-demo-backup-key",
        rate_limit=60,
        quota_limit=5000
    )
    
    api_manager.add_endpoint(
        name="anthropic_claude",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-demo-key",
        rate_limit=50,
        quota_limit=8000
    )
    
    print("✅ API rotation setup complete")
    
    # Demonstrate API rotation
    print("\n📡 Testing API rotation...")
    for i in range(5):
        result = api_manager.make_api_request("completion")
        if result["success"]:
            print(f"  Request {i+1}: ✅ {result['endpoint_used']}")
        else:
            print(f"  Request {i+1}: ❌ {result['error']}")
    
    # Show API status
    status = api_manager.get_status_report()
    print(f"\n📊 API Status: {status['overall_stats']['total_requests']} total requests")
    for endpoint in status['endpoints']:
        print(f"  {endpoint['name']}: {endpoint['rate_usage']} rate, {endpoint['quota_usage']} quota")

def main():
    """Demonstrate the universal long-running agent with API rotation and a sample PRD."""
    
    print("🤖 Universal Long-Running Agent with API Rotation")
    print("=" * 55)
    
    # Setup API rotation demonstration
    setup_api_rotation_demo()
    
    # Sample PRD for demonstration
    sample_prd = """
# E-commerce Platform MVP

## User Registration System
Implement user registration with email verification and profile creation
- Email validation and uniqueness checking
- Password strength requirements (8+ characters, mixed case, numbers)
- Email verification flow with secure tokens
- User profile creation with basic information fields
- Account activation process

## Product Catalog Management
Create a product catalog system for browsing and searching products
- Product listing with pagination
- Search functionality with filters (category, price, rating)
- Product detail pages with images and descriptions
- Category-based navigation
- Product rating and review system

## Shopping Cart Functionality
Implement shopping cart for managing selected products
- Add/remove products from cart
- Quantity management with stock validation
- Cart persistence across sessions
- Price calculation with taxes and discounts
- Cart summary and checkout preparation

## User Authentication
Secure login system with session management
- Email/password authentication
- JWT token generation and validation
- Session timeout and refresh logic
- Password reset via email
- Remember me functionality

## Order Processing System
Complete order processing from cart to fulfillment
- Order creation from cart contents
- Payment integration (mock for MVP)
- Order status tracking
- Email notifications for order updates
- Order history for users

## Admin Dashboard
Administrative interface for managing the platform
- User management (view, activate, deactivate)
- Product management (CRUD operations)
- Order management and status updates
- Basic analytics and reporting
- Content management for site pages
"""
    
    print("🌟 Universal Long-Running Agent Example")
    print("=====================================")
    print("This example works with ANY AI agent framework!")
    print("(Cursor, OpenCode, Claude, Custom Agents, etc.)\\n")
    
    # Run the workflow
    success = universal_long_running_agent_workflow(
        sample_prd, 
        "E-commerce Platform MVP"
    )
    
    if success:
        print("\\n✨ Example completed successfully!")
        print("\\nTo integrate with your AI agent:")
        print("1. Copy the core functions into your agent's context")
        print("2. Call universal_long_running_agent_workflow() with your PRD")
        print("3. Monitor progress through the generated files")
        print("4. Customize implementation functions for your specific needs")
    else:
        print("\\n❌ Example encountered issues")

if __name__ == "__main__":
    main()