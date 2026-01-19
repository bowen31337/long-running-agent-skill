# Learning and Memory System Patterns

## Overview

This reference covers the learning and memory system that enables long-running agents to improve their performance over time by recognizing patterns, storing solutions, and adapting execution strategies based on historical data.

## Core Learning Architecture

### Pattern Recognition System

```python
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

class ExecutionPatternLearner:
    """Learn from task execution patterns to improve future performance."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.patterns_file = f"memories/{project_name}/patterns.json"
        self.solutions_file = f"memories/{project_name}/solutions.json"
        self.performance_file = f"memories/{project_name}/performance.json"
        
        # Ensure memory directory exists
        os.makedirs(f"memories/{project_name}", exist_ok=True)
    
    def save_execution_pattern(self, task: Dict, execution_result: Dict):
        """Save successful execution patterns for learning."""
        
        pattern = {
            "pattern_id": f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "task_category": task["category"],
            "task_type": task.get("type", "general"),
            "task_complexity": self._assess_complexity(task),
            "execution_approach": execution_result.get("approach", "standard"),
            "success_factors": execution_result.get("success_factors", []),
            "execution_time": execution_result.get("execution_time", 0),
            "dependencies_resolved": task.get("dependencies", []),
            "tools_used": execution_result.get("tools_used", []),
            "api_calls_made": execution_result.get("api_calls_made", 0),
            "timestamp": datetime.now().isoformat(),
            "success": execution_result.get("success", False)
        }
        
        # Load existing patterns
        patterns = self._load_json_file(self.patterns_file, [])
        patterns.append(pattern)
        
        # Keep only last 1000 patterns to prevent file bloat
        if len(patterns) > 1000:
            patterns = patterns[-1000:]
        
        self._save_json_file(self.patterns_file, patterns)
        print(f"📚 Saved execution pattern: {pattern['pattern_id']}")
    
    def save_error_solution(self, error_info: Dict, solution: Dict):
        """Save error solutions for future reference."""
        
        solution_entry = {
            "solution_id": f"solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "error_type": error_info.get("error_type", "unknown"),
            "error_message": error_info.get("error_message", ""),
            "error_context": error_info.get("context", {}),
            "solution_approach": solution.get("approach", ""),
            "solution_steps": solution.get("steps", []),
            "success_rate": solution.get("success_rate", 0.0),
            "applicable_categories": solution.get("applicable_categories", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Load existing solutions
        solutions = self._load_json_file(self.solutions_file, [])
        solutions.append(solution_entry)
        
        # Keep only last 500 solutions
        if len(solutions) > 500:
            solutions = solutions[-500:]
        
        self._save_json_file(self.solutions_file, solutions)
        print(f"🔧 Saved error solution: {solution_entry['solution_id']}")
    
    def get_similar_patterns(self, task: Dict, limit: int = 5) -> List[Dict]:
        """Find similar execution patterns for a given task."""
        
        patterns = self._load_json_file(self.patterns_file, [])
        
        # Filter successful patterns only
        successful_patterns = [p for p in patterns if p.get("success", False)]
        
        # Score patterns by similarity
        scored_patterns = []
        for pattern in successful_patterns:
            similarity_score = self._calculate_pattern_similarity(task, pattern)
            if similarity_score > 0.3:  # Minimum similarity threshold
                scored_patterns.append((pattern, similarity_score))
        
        # Sort by similarity score (highest first)
        scored_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Return top patterns
        return [pattern for pattern, score in scored_patterns[:limit]]
    
    def get_error_solutions(self, error_type: str, error_message: str = None) -> List[Dict]:
        """Get relevant error solutions based on error type and message."""
        
        solutions = self._load_json_file(self.solutions_file, [])
        
        relevant_solutions = []
        for solution in solutions:
            # Exact error type match
            if solution["error_type"] == error_type:
                relevance_score = 1.0
                
                # Boost score if error message is similar
                if error_message and solution["error_message"]:
                    message_similarity = self._calculate_text_similarity(
                        error_message.lower(), 
                        solution["error_message"].lower()
                    )
                    relevance_score += message_similarity * 0.5
                
                relevant_solutions.append((solution, relevance_score))
        
        # Sort by relevance score
        relevant_solutions.sort(key=lambda x: x[1], reverse=True)
        
        return [solution for solution, score in relevant_solutions[:3]]
    
    def analyze_performance_trends(self) -> Dict:
        """Analyze performance trends across task categories."""
        
        patterns = self._load_json_file(self.patterns_file, [])
        performance_data = self._load_json_file(self.performance_file, {})
        
        # Group patterns by category
        category_stats = defaultdict(list)
        for pattern in patterns:
            category = pattern["task_category"]
            category_stats[category].append(pattern)
        
        # Calculate trends for each category
        trends = {}
        for category, category_patterns in category_stats.items():
            # Sort by timestamp
            category_patterns.sort(key=lambda x: x["timestamp"])
            
            # Calculate success rate trend
            recent_patterns = category_patterns[-20:]  # Last 20 executions
            success_rate = sum(1 for p in recent_patterns if p.get("success", False)) / len(recent_patterns)
            
            # Calculate average execution time
            exec_times = [p["execution_time"] for p in recent_patterns if p["execution_time"] > 0]
            avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
            
            # Identify most successful approaches
            approaches = Counter(p["execution_approach"] for p in recent_patterns if p.get("success", False))
            
            trends[category] = {
                "success_rate": success_rate,
                "average_execution_time": avg_exec_time,
                "total_executions": len(category_patterns),
                "recent_executions": len(recent_patterns),
                "best_approaches": approaches.most_common(3),
                "improvement_trend": self._calculate_improvement_trend(category_patterns)
            }
        
        # Update performance file
        performance_data["last_analysis"] = datetime.now().isoformat()
        performance_data["trends"] = trends
        self._save_json_file(self.performance_file, performance_data)
        
        return trends
    
    def get_execution_recommendations(self, task: Dict) -> Dict:
        """Get execution recommendations based on learned patterns."""
        
        similar_patterns = self.get_similar_patterns(task)
        
        if not similar_patterns:
            return {
                "approach": "standard",
                "confidence": 0.0,
                "recommendations": ["No similar patterns found, using standard approach"]
            }
        
        # Analyze successful patterns
        approaches = Counter(p["execution_approach"] for p in similar_patterns)
        success_factors = []
        tools_used = Counter()
        
        for pattern in similar_patterns:
            success_factors.extend(pattern.get("success_factors", []))
            for tool in pattern.get("tools_used", []):
                tools_used[tool] += 1
        
        # Generate recommendations
        recommended_approach = approaches.most_common(1)[0][0]
        common_success_factors = Counter(success_factors).most_common(5)
        recommended_tools = tools_used.most_common(3)
        
        confidence = len(similar_patterns) / 10.0  # Max confidence at 10 similar patterns
        confidence = min(confidence, 1.0)
        
        recommendations = []
        recommendations.append(f"Use '{recommended_approach}' approach (used in {approaches[recommended_approach]} similar tasks)")
        
        if common_success_factors:
            recommendations.append("Key success factors:")
            for factor, count in common_success_factors:
                recommendations.append(f"  - {factor} (appeared in {count} successful executions)")
        
        if recommended_tools:
            recommendations.append("Recommended tools:")
            for tool, count in recommended_tools:
                recommendations.append(f"  - {tool} (used in {count} similar tasks)")
        
        return {
            "approach": recommended_approach,
            "confidence": confidence,
            "recommendations": recommendations,
            "similar_patterns_count": len(similar_patterns)
        }
    
    def _assess_complexity(self, task: Dict) -> str:
        """Assess task complexity based on various factors."""
        
        complexity_score = 0
        
        # Factor 1: Description length
        description = task.get("description", "")
        if len(description) > 200:
            complexity_score += 2
        elif len(description) > 100:
            complexity_score += 1
        
        # Factor 2: Number of dependencies
        dependencies = task.get("dependencies", [])
        complexity_score += len(dependencies)
        
        # Factor 3: Estimated effort
        effort = task.get("effort", 1)
        if effort > 5:
            complexity_score += 3
        elif effort > 2:
            complexity_score += 1
        
        # Factor 4: Category complexity
        category = task.get("category", "general")
        complex_categories = ["backend", "database", "integration", "auth"]
        if category in complex_categories:
            complexity_score += 2
        
        # Classify complexity
        if complexity_score >= 7:
            return "high"
        elif complexity_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _calculate_pattern_similarity(self, task: Dict, pattern: Dict) -> float:
        """Calculate similarity score between a task and a pattern."""
        
        similarity_score = 0.0
        
        # Category match (high weight)
        if task.get("category") == pattern.get("task_category"):
            similarity_score += 0.4
        
        # Type match (medium weight)
        if task.get("type") == pattern.get("task_type"):
            similarity_score += 0.3
        
        # Complexity match (low weight)
        task_complexity = self._assess_complexity(task)
        if task_complexity == pattern.get("task_complexity"):
            similarity_score += 0.2
        
        # Description similarity (medium weight)
        task_desc = task.get("description", "").lower()
        pattern_desc = pattern.get("task_category", "").lower()
        desc_similarity = self._calculate_text_similarity(task_desc, pattern_desc)
        similarity_score += desc_similarity * 0.1
        
        return similarity_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity based on common words."""
        
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_improvement_trend(self, patterns: List[Dict]) -> str:
        """Calculate improvement trend for a category."""
        
        if len(patterns) < 10:
            return "insufficient_data"
        
        # Split into first half and second half
        mid_point = len(patterns) // 2
        first_half = patterns[:mid_point]
        second_half = patterns[mid_point:]
        
        # Calculate success rates
        first_half_success = sum(1 for p in first_half if p.get("success", False)) / len(first_half)
        second_half_success = sum(1 for p in second_half if p.get("success", False)) / len(second_half)
        
        # Determine trend
        improvement = second_half_success - first_half_success
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _load_json_file(self, file_path: str, default: Any = None) -> Any:
        """Load JSON file with error handling."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return default if default is not None else {}
    
    def _save_json_file(self, file_path: str, data: Any):
        """Save JSON file with error handling."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

# Global learning system instance
learning_system = None

def initialize_learning_system(project_name: str):
    """Initialize the learning system for a project."""
    global learning_system
    learning_system = ExecutionPatternLearner(project_name)
    print(f"🧠 Learning system initialized for project: {project_name}")

def save_execution_pattern(task: Dict, execution_result: Dict):
    """Save execution pattern for learning."""
    if learning_system:
        learning_system.save_execution_pattern(task, execution_result)

def save_error_solution(error_info: Dict, solution: Dict):
    """Save error solution for future reference."""
    if learning_system:
        learning_system.save_error_solution(error_info, solution)

def get_execution_recommendations(task: Dict) -> Dict:
    """Get execution recommendations based on learned patterns."""
    if learning_system:
        return learning_system.get_execution_recommendations(task)
    
    return {
        "approach": "standard",
        "confidence": 0.0,
        "recommendations": ["Learning system not initialized"]
    }
```

## Advanced Learning Patterns

### Adaptive Execution Strategy

```python
class AdaptiveExecutionEngine:
    """Execution engine that adapts based on learned patterns."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.learner = ExecutionPatternLearner(project_name)
    
    def execute_task_adaptively(self, task: Dict) -> Dict:
        """Execute task using adaptive strategy based on learned patterns."""
        
        # Get recommendations from learning system
        recommendations = self.learner.get_execution_recommendations(task)
        
        print(f"🎯 Execution recommendations for {task['id']}:")
        print(f"   Approach: {recommendations['approach']} (confidence: {recommendations['confidence']:.2f})")
        
        for rec in recommendations['recommendations']:
            print(f"   {rec}")
        
        # Execute with recommended approach
        start_time = time.time()
        
        try:
            if recommendations['approach'] == 'incremental':
                result = self._execute_incrementally(task)
            elif recommendations['approach'] == 'parallel':
                result = self._execute_in_parallel(task)
            elif recommendations['approach'] == 'api_heavy':
                result = self._execute_with_api_focus(task)
            else:
                result = self._execute_standard(task)
            
            execution_time = time.time() - start_time
            
            # Record successful execution
            execution_result = {
                "success": True,
                "approach": recommendations['approach'],
                "execution_time": execution_time,
                "success_factors": self._identify_success_factors(task, result),
                "tools_used": result.get("tools_used", []),
                "api_calls_made": result.get("api_calls_made", 0)
            }
            
            self.learner.save_execution_pattern(task, execution_result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Try to find solution for this error
            error_solutions = self.learner.get_error_solutions(
                error_type=type(e).__name__,
                error_message=str(e)
            )
            
            if error_solutions:
                print(f"🔧 Found {len(error_solutions)} potential solutions for this error")
                
                # Try the best solution
                best_solution = error_solutions[0]
                try:
                    result = self._apply_error_solution(task, best_solution)
                    
                    # Record successful recovery
                    execution_result = {
                        "success": True,
                        "approach": f"{recommendations['approach']}_recovered",
                        "execution_time": time.time() - start_time,
                        "success_factors": ["error_recovery"] + self._identify_success_factors(task, result),
                        "recovery_used": True
                    }
                    
                    self.learner.save_execution_pattern(task, execution_result)
                    
                    return result
                    
                except Exception as recovery_error:
                    print(f"❌ Recovery attempt failed: {recovery_error}")
            
            # Record failed execution
            execution_result = {
                "success": False,
                "approach": recommendations['approach'],
                "execution_time": execution_time,
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
            
            self.learner.save_execution_pattern(task, execution_result)
            
            raise e
    
    def _execute_incrementally(self, task: Dict) -> Dict:
        """Execute task in small increments."""
        # Break task into smaller pieces and execute step by step
        pass
    
    def _execute_in_parallel(self, task: Dict) -> Dict:
        """Execute task components in parallel where possible."""
        # Identify parallelizable components and execute concurrently
        pass
    
    def _execute_with_api_focus(self, task: Dict) -> Dict:
        """Execute task with emphasis on API efficiency."""
        # Use API rotation and batch API calls where possible
        pass
    
    def _execute_standard(self, task: Dict) -> Dict:
        """Execute task using standard approach."""
        # Standard sequential execution
        pass
    
    def _identify_success_factors(self, task: Dict, result: Dict) -> List[str]:
        """Identify factors that contributed to successful execution."""
        factors = []
        
        # Analyze result to identify success factors
        if result.get("files_created"):
            factors.append("file_creation")
        
        if result.get("api_calls_made", 0) > 0:
            factors.append("api_integration")
        
        if result.get("dependencies_resolved"):
            factors.append("dependency_resolution")
        
        if task.get("category") == "frontend" and "component" in result.get("output", "").lower():
            factors.append("component_creation")
        
        return factors
    
    def _apply_error_solution(self, task: Dict, solution: Dict) -> Dict:
        """Apply a learned error solution to recover from failure."""
        
        print(f"🔧 Applying solution: {solution['solution_approach']}")
        
        # Execute solution steps
        for step in solution.get("solution_steps", []):
            print(f"   Executing: {step}")
            # Apply each solution step
        
        # Retry task execution
        return self._execute_standard(task)
```

### Performance Analytics

```python
class PerformanceAnalyzer:
    """Analyze and report on agent performance over time."""
    
    def __init__(self, project_name: str):
        self.project_name = project_name
        self.learner = ExecutionPatternLearner(project_name)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        
        trends = self.learner.analyze_performance_trends()
        
        report = {
            "project_name": self.project_name,
            "generated_at": datetime.now().isoformat(),
            "overall_metrics": self._calculate_overall_metrics(),
            "category_performance": trends,
            "recommendations": self._generate_improvement_recommendations(trends),
            "learning_insights": self._extract_learning_insights()
        }
        
        # Save report
        report_file = f"memories/{self.project_name}/performance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report generated: {report_file}")
        
        return report
    
    def _calculate_overall_metrics(self) -> Dict:
        """Calculate overall performance metrics."""
        
        patterns = self.learner._load_json_file(self.learner.patterns_file, [])
        
        if not patterns:
            return {"status": "no_data"}
        
        total_executions = len(patterns)
        successful_executions = sum(1 for p in patterns if p.get("success", False))
        success_rate = successful_executions / total_executions
        
        # Calculate average execution time
        exec_times = [p["execution_time"] for p in patterns if p["execution_time"] > 0]
        avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        # Calculate category distribution
        categories = Counter(p["task_category"] for p in patterns)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_exec_time,
            "category_distribution": dict(categories.most_common())
        }
    
    def _generate_improvement_recommendations(self, trends: Dict) -> List[str]:
        """Generate recommendations for improving performance."""
        
        recommendations = []
        
        for category, trend_data in trends.items():
            success_rate = trend_data["success_rate"]
            improvement_trend = trend_data["improvement_trend"]
            
            if success_rate < 0.7:
                recommendations.append(f"Focus on improving {category} tasks (current success rate: {success_rate:.1%})")
            
            if improvement_trend == "declining":
                recommendations.append(f"Investigate declining performance in {category} category")
            
            if trend_data["average_execution_time"] > 300:  # 5 minutes
                recommendations.append(f"Optimize execution time for {category} tasks")
        
        if not recommendations:
            recommendations.append("Performance is stable across all categories")
        
        return recommendations
    
    def _extract_learning_insights(self) -> List[str]:
        """Extract insights from learning patterns."""
        
        patterns = self.learner._load_json_file(self.learner.patterns_file, [])
        
        if len(patterns) < 10:
            return ["Insufficient data for learning insights"]
        
        insights = []
        
        # Analyze approach effectiveness
        approaches = Counter(p["execution_approach"] for p in patterns if p.get("success", False))
        if approaches:
            best_approach = approaches.most_common(1)[0]
            insights.append(f"Most successful approach: {best_approach[0]} ({best_approach[1]} successes)")
        
        # Analyze success factors
        all_success_factors = []
        for pattern in patterns:
            if pattern.get("success", False):
                all_success_factors.extend(pattern.get("success_factors", []))
        
        if all_success_factors:
            common_factors = Counter(all_success_factors).most_common(3)
            insights.append("Key success factors:")
            for factor, count in common_factors:
                insights.append(f"  - {factor} (appeared in {count} successful executions)")
        
        return insights

def generate_learning_report(project_name: str) -> Dict:
    """Generate learning and performance report for a project."""
    analyzer = PerformanceAnalyzer(project_name)
    return analyzer.generate_performance_report()
```

## Integration with Task Execution

### Learning-Enhanced Task Execution

```python
def execute_task_with_learning(task: Dict, project_name: str) -> Dict:
    """Execute task with learning system integration."""
    
    # Initialize learning system if not already done
    if not learning_system:
        initialize_learning_system(project_name)
    
    # Get execution recommendations
    recommendations = get_execution_recommendations(task)
    
    # Execute with adaptive strategy
    adaptive_engine = AdaptiveExecutionEngine(project_name)
    
    try:
        result = adaptive_engine.execute_task_adaptively(task)
        
        # Learn from successful execution
        if result.get("success", False):
            print(f"✅ Task {task['id']} completed successfully with learning")
        
        return result
        
    except Exception as e:
        # Learn from failure
        error_info = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "context": {
                "task_category": task.get("category"),
                "task_type": task.get("type"),
                "approach_used": recommendations.get("approach", "standard")
            }
        }
        
        print(f"❌ Task {task['id']} failed, recording for learning: {e}")
        
        # This error will be available for future solution matching
        return {
            "success": False,
            "error": str(e),
            "task_id": task["id"],
            "learning_recorded": True
        }

def continuous_learning_loop(project_name: str):
    """Run continuous learning analysis for ongoing improvement."""
    
    print(f"🧠 Starting continuous learning analysis for {project_name}")
    
    # Generate performance report
    report = generate_learning_report(project_name)
    
    # Print key insights
    print("\n📊 Performance Insights:")
    for insight in report.get("learning_insights", []):
        print(f"   {insight}")
    
    print("\n💡 Improvement Recommendations:")
    for rec in report.get("recommendations", []):
        print(f"   {rec}")
    
    return report
```

## Usage Examples

### Basic Learning Integration

```python
# Initialize learning system
initialize_learning_system("ecommerce-platform")

# Execute tasks with learning
task_list = load_task_list("ecommerce-platform")
for task in task_list["tasks"]:
    if task["status"] == "pending":
        result = execute_task_with_learning(task, "ecommerce-platform")
        
        if result["success"]:
            update_task_status(task["id"], "completed")
        else:
            update_task_status(task["id"], "failed")

# Generate learning report
learning_report = continuous_learning_loop("ecommerce-platform")
```

### Advanced Learning Configuration

```python
# Custom learning configuration
class CustomLearningConfig:
    """Custom configuration for learning system."""
    
    def __init__(self):
        self.pattern_retention_limit = 2000  # Keep more patterns
        self.solution_retention_limit = 1000  # Keep more solutions
        self.similarity_threshold = 0.2  # Lower threshold for more matches
        self.confidence_boost_factor = 1.5  # Boost confidence calculations
    
    def apply_to_learner(self, learner: ExecutionPatternLearner):
        """Apply custom configuration to learner."""
        # Override default limits and thresholds
        pass

# Use custom configuration
config = CustomLearningConfig()
learner = ExecutionPatternLearner("advanced-project")
config.apply_to_learner(learner)
```

This learning system enables long-running agents to continuously improve their performance by recognizing successful patterns, learning from failures, and adapting execution strategies based on historical data.