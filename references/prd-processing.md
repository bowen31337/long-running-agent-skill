# PRD Processing Patterns

## Overview

This reference covers patterns for parsing Product Requirements Documents (PRDs) and specifications into structured, executable task lists using DeepAgents.

## Core PRD Parsing Tool

```python
from langchain_core.tools import tool
from typing import Dict, List, Any
import json
import re
from datetime import datetime

@tool
def parse_prd_to_tasks(prd_content: str, project_name: str = "Untitled Project") -> str:
    """Parse PRD/specification content into structured task list JSON.
    
    Args:
        prd_content: The PRD or specification text
        project_name: Name for the project
    
    Returns:
        JSON string with structured task list
    """
    try:
        tasks = []
        task_id_counter = 1
        
        # Split content into sections
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
            
            # Parse description and extract details
            description = '\n'.join(description_lines).strip()
            
            # Extract acceptance criteria (lines starting with -, *, or numbered)
            acceptance_criteria = []
            remaining_description = []
            
            for line in description_lines:
                line = line.strip()
                if re.match(r'^[-*]\s+|^\d+\.\s+', line):
                    # This is a bullet point or numbered item
                    criteria = re.sub(r'^[-*]\s+|^\d+\.\s+', '', line)
                    acceptance_criteria.append(criteria)
                else:
                    remaining_description.append(line)
            
            # Determine priority based on keywords
            priority = "medium"
            content_lower = (title + ' ' + description).lower()
            if any(word in content_lower for word in ['critical', 'urgent', 'high priority', 'must have']):
                priority = "high"
            elif any(word in content_lower for word in ['nice to have', 'optional', 'low priority', 'future']):
                priority = "low"
            
            # Estimate effort based on complexity indicators
            effort = "medium"
            if any(word in content_lower for word in ['simple', 'basic', 'quick', 'small']):
                effort = "small"
            elif any(word in content_lower for word in ['complex', 'advanced', 'large', 'system', 'integration']):
                effort = "large"
            
            # Create task object
            task = {
                "id": f"task_{task_id_counter:03d}",
                "title": title,
                "description": '\n'.join(remaining_description).strip() or title,
                "priority": priority,
                "estimated_effort": effort,
                "acceptance_criteria": acceptance_criteria,
                "dependencies": [],  # Will be populated in dependency analysis
                "status": "pending",
                "assigned_to": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "tags": extract_tags(title + ' ' + description),
                "category": categorize_task(title, description)
            }
            
            tasks.append(task)
            task_id_counter += 1
        
        # Analyze dependencies
        tasks = analyze_task_dependencies(tasks)
        
        # Create task list structure
        task_list = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total_tasks": len(tasks),
            "completed_tasks": 0,
            "in_progress_tasks": 0,
            "pending_tasks": len(tasks),
            "failed_tasks": 0,
            "tasks": tasks,
            "metadata": {
                "source": "prd_parsing",
                "version": "1.0",
                "parsing_method": "regex_based"
            }
        }
        
        return json.dumps(task_list, indent=2)
        
    except Exception as e:
        error_response = {
            "error": f"Failed to parse PRD: {str(e)}",
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "tasks": []
        }
        return json.dumps(error_response, indent=2)

def extract_tags(content: str) -> List[str]:
    """Extract relevant tags from content."""
    tags = []
    content_lower = content.lower()
    
    # Technology tags
    tech_keywords = {
        'frontend': ['ui', 'interface', 'frontend', 'client', 'web', 'mobile'],
        'backend': ['api', 'server', 'backend', 'database', 'service'],
        'auth': ['login', 'authentication', 'auth', 'security', 'password'],
        'data': ['database', 'data', 'storage', 'migration', 'model'],
        'testing': ['test', 'testing', 'validation', 'verify'],
        'deployment': ['deploy', 'deployment', 'infrastructure', 'devops'],
        'documentation': ['document', 'docs', 'readme', 'guide']
    }
    
    for tag, keywords in tech_keywords.items():
        if any(keyword in content_lower for keyword in keywords):
            tags.append(tag)
    
    return list(set(tags))

def categorize_task(title: str, description: str) -> str:
    """Categorize task based on content."""
    content = (title + ' ' + description).lower()
    
    if any(word in content for word in ['ui', 'interface', 'frontend', 'design']):
        return 'frontend'
    elif any(word in content for word in ['api', 'backend', 'server', 'database']):
        return 'backend'
    elif any(word in content for word in ['test', 'testing', 'validation']):
        return 'testing'
    elif any(word in content for word in ['deploy', 'infrastructure', 'devops']):
        return 'deployment'
    elif any(word in content for word in ['document', 'docs', 'readme']):
        return 'documentation'
    else:
        return 'general'

def analyze_task_dependencies(tasks: List[Dict]) -> List[Dict]:
    """Analyze and set task dependencies based on content and categories."""
    
    # Common dependency patterns
    dependency_rules = [
        # Backend usually comes before frontend
        ('backend', 'frontend'),
        # Authentication needed before user features
        ('auth', 'user'),
        # Database setup before data operations
        ('database', 'data'),
        # Core features before advanced features
        ('basic', 'advanced'),
        # Setup before implementation
        ('setup', 'implementation'),
        # Models before views
        ('model', 'view'),
        # API before client
        ('api', 'client')
    ]
    
    for i, task in enumerate(tasks):
        dependencies = []
        
        # Check against previous tasks for dependencies
        for j, other_task in enumerate(tasks[:i]):
            # Check category-based dependencies
            for prereq, dependent in dependency_rules:
                if (prereq in other_task['category'] or prereq in other_task['title'].lower()) and \
                   (dependent in task['category'] or dependent in task['title'].lower()):
                    dependencies.append(other_task['id'])
        
        task['dependencies'] = list(set(dependencies))
    
    return tasks
```

## Advanced PRD Parsing Patterns

### Multi-Format Support

```python
@tool
def parse_structured_prd(prd_content: str, format_type: str = "auto") -> str:
    """Parse PRD with support for multiple formats (Markdown, JIRA, etc.)."""
    
    if format_type == "auto":
        format_type = detect_prd_format(prd_content)
    
    if format_type == "jira":
        return parse_jira_format(prd_content)
    elif format_type == "user_stories":
        return parse_user_stories_format(prd_content)
    elif format_type == "technical_spec":
        return parse_technical_spec_format(prd_content)
    else:
        return parse_prd_to_tasks(prd_content)

def detect_prd_format(content: str) -> str:
    """Auto-detect PRD format based on content patterns."""
    if re.search(r'As a .+ I want .+ So that', content, re.IGNORECASE):
        return "user_stories"
    elif re.search(r'Epic:|Story:|Task:', content):
        return "jira"
    elif re.search(r'## Technical Requirements|### API Specification', content):
        return "technical_spec"
    else:
        return "markdown"
```

### Template-Based Parsing

```python
@tool
def parse_prd_with_template(prd_content: str, template_name: str) -> str:
    """Parse PRD using predefined templates for consistent extraction."""
    
    templates = {
        "feature_spec": {
            "sections": ["overview", "requirements", "acceptance_criteria", "technical_notes"],
            "task_indicators": ["requirement", "feature", "story"],
            "priority_keywords": {
                "high": ["critical", "must have", "p0"],
                "medium": ["should have", "p1", "important"],
                "low": ["nice to have", "p2", "optional"]
            }
        },
        "epic_breakdown": {
            "sections": ["epic", "stories", "tasks", "subtasks"],
            "hierarchy_levels": ["#", "##", "###", "####"],
            "dependency_indicators": ["depends on", "requires", "after"]
        }
    }
    
    template = templates.get(template_name, templates["feature_spec"])
    
    # Use template to guide parsing
    return parse_with_template_rules(prd_content, template)
```

## Integration with DeepAgents Memory

```python
@tool
def save_prd_parsing_patterns(prd_content: str, task_list: str, success_metrics: Dict) -> str:
    """Save successful PRD parsing patterns to memory for future use."""
    
    pattern = {
        "prd_characteristics": analyze_prd_structure(prd_content),
        "parsing_approach": "regex_based",
        "task_count": len(json.loads(task_list)["tasks"]),
        "success_metrics": success_metrics,
        "timestamp": datetime.now().isoformat()
    }
    
    # Save to DeepAgents memory
    return f"Save pattern to /memories/prd_patterns.json: {json.dumps(pattern, indent=2)}"

@tool
def load_similar_prd_patterns(prd_content: str) -> str:
    """Load similar PRD parsing patterns from memory to improve accuracy."""
    
    # Analyze current PRD
    current_structure = analyze_prd_structure(prd_content)
    
    # Load and compare with saved patterns
    return "Use read_file to load /memories/prd_patterns.json and find similar patterns"
```

## Error Handling and Validation

```python
@tool
def validate_task_list(task_list_json: str) -> str:
    """Validate generated task list for completeness and consistency."""
    
    try:
        task_list = json.loads(task_list_json)
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Check required fields
        required_fields = ["project_name", "tasks", "total_tasks"]
        for field in required_fields:
            if field not in task_list:
                validation_results["errors"].append(f"Missing required field: {field}")
                validation_results["valid"] = False
        
        # Validate tasks
        if "tasks" in task_list:
            for i, task in enumerate(task_list["tasks"]):
                task_errors = validate_single_task(task, i)
                validation_results["errors"].extend(task_errors)
                if task_errors:
                    validation_results["valid"] = False
        
        # Check for circular dependencies
        if "tasks" in task_list:
            circular_deps = detect_circular_dependencies(task_list["tasks"])
            if circular_deps:
                validation_results["errors"].extend(circular_deps)
                validation_results["valid"] = False
        
        return json.dumps(validation_results, indent=2)
        
    except json.JSONDecodeError as e:
        return json.dumps({
            "valid": False,
            "errors": [f"Invalid JSON: {str(e)}"],
            "warnings": [],
            "suggestions": []
        })

def validate_single_task(task: Dict, index: int) -> List[str]:
    """Validate individual task structure."""
    errors = []
    required_task_fields = ["id", "title", "description", "status"]
    
    for field in required_task_fields:
        if field not in task:
            errors.append(f"Task {index}: Missing required field '{field}'")
    
    # Validate task ID format
    if "id" in task and not re.match(r'^task_\d{3}$', task["id"]):
        errors.append(f"Task {index}: Invalid ID format '{task['id']}' (expected: task_XXX)")
    
    return errors
```

## Usage Examples

### Basic PRD Processing

```python
# Simple PRD parsing
prd_text = """
# User Management System

## User Registration
Allow new users to create accounts with email verification

## User Login  
Secure authentication with session management

## Profile Management
Users can update their profile information
"""

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": f"Parse this PRD into tasks: {prd_text}"
    }]
})
```

### Advanced PRD with Templates

```python
# Using template-based parsing
complex_prd = """
Epic: E-commerce Platform

Story 1: As a customer, I want to browse products so that I can find items to purchase
- Display product catalog
- Implement search functionality  
- Add filtering options

Story 2: As a customer, I want to add items to cart so that I can purchase multiple items
- Shopping cart functionality
- Quantity management
- Price calculations
"""

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": f"Parse this user story format PRD: {complex_prd}"
    }]
})
```