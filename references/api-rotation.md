# API Rotation and Management Patterns

## Overview

This reference covers intelligent API rotation, rate limiting, and load balancing patterns for long-running agents that need to make extensive API calls while avoiding rate limits and quota exhaustion.

## Core API Rotation System

### APIEndpoint Class

```python
from dataclasses import dataclass, field
from enum import Enum
import time

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
    priority: int = 1  # Higher number = higher priority
    
    def can_make_request(self) -> bool:
        """Check if endpoint can handle a request."""
        if self.status != APIStatus.ACTIVE:
            return False
            
        current_time = time.time()
        
        # Reset rate limit counter if minute has passed
        if current_time - self.last_request_time >= 60:
            self.current_usage = 0
            self.last_request_time = current_time
        
        # Reset daily usage if day has passed
        if current_time - self.last_reset_time >= 86400:
            self.daily_usage = 0
            self.last_reset_time = current_time
            if self.status == APIStatus.QUOTA_EXCEEDED:
                self.status = APIStatus.ACTIVE
        
        return (self.current_usage < self.rate_limit and 
                self.daily_usage < self.quota_limit)
    
    def record_request(self, success: bool = True, response_time: float = 0):
        """Record a request and update usage counters."""
        self.current_usage += 1
        self.daily_usage += 1
        self.last_request_time = time.time()
        
        if success:
            # Decay error count on successful requests
            self.error_count = max(0, self.error_count - 1)
        else:
            self.error_count += 1
            
            # Disable endpoint if too many consecutive errors
            if self.error_count >= 5:
                self.status = APIStatus.ERROR
                print(f"❌ Endpoint {self.name} disabled due to excessive errors")
    
    def get_capacity_score(self) -> float:
        """Get endpoint capacity score (0-1, higher is better)."""
        if not self.can_make_request():
            return 0.0
        
        # Rate limit capacity
        rate_capacity = (self.rate_limit - self.current_usage) / self.rate_limit
        
        # Quota capacity
        quota_capacity = (self.quota_limit - self.daily_usage) / self.quota_limit
        
        # Error penalty (fewer errors = higher score)
        error_penalty = max(0, 1 - (self.error_count / 10))
        
        # Combined score with weights
        return (rate_capacity * 0.4 + quota_capacity * 0.4 + error_penalty * 0.2) * self.priority
```

## Advanced Rotation Strategies

### Weighted Round-Robin Selection

```python
class APIRotationManager:
    """Advanced API rotation manager with multiple selection strategies."""
    
    def __init__(self, strategy: str = "weighted"):
        self.endpoints: List[APIEndpoint] = []
        self.strategy = strategy  # "round_robin", "weighted", "priority", "least_used"
        self.current_index = 0
        self.lock = threading.Lock()
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rotations": 0,
            "rate_limit_hits": 0,
            "quota_exceeded_count": 0,
            "average_response_time": 0
        }
    
    def get_best_endpoint(self) -> Optional[APIEndpoint]:
        """Get the best endpoint using configured strategy."""
        with self.lock:
            available = [ep for ep in self.endpoints if ep.can_make_request()]
            
            if not available:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(available)
            elif self.strategy == "weighted":
                return self._weighted_selection(available)
            elif self.strategy == "priority":
                return self._priority_selection(available)
            elif self.strategy == "least_used":
                return self._least_used_selection(available)
            else:
                return available[0]  # Fallback
    
    def _weighted_selection(self, available: List[APIEndpoint]) -> APIEndpoint:
        """Select endpoint based on capacity scores."""
        scored_endpoints = [(ep, ep.get_capacity_score()) for ep in available]
        scored_endpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Use weighted random selection from top 50%
        top_half = scored_endpoints[:max(1, len(scored_endpoints) // 2)]
        weights = [score for _, score in top_half]
        
        if sum(weights) == 0:
            return available[0]
        
        # Weighted random selection
        import random
        total_weight = sum(weights)
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for (endpoint, weight) in top_half:
            current_weight += weight
            if rand_val <= current_weight:
                return endpoint
        
        return top_half[0][0]
    
    def _priority_selection(self, available: List[APIEndpoint]) -> APIEndpoint:
        """Select highest priority available endpoint."""
        return max(available, key=lambda ep: ep.priority)
    
    def _least_used_selection(self, available: List[APIEndpoint]) -> APIEndpoint:
        """Select endpoint with lowest current usage."""
        return min(available, key=lambda ep: ep.current_usage)
    
    def _round_robin_selection(self, available: List[APIEndpoint]) -> APIEndpoint:
        """Simple round-robin selection."""
        endpoint = available[self.current_index % len(available)]
        self.current_index += 1
        return endpoint
```

## Rate Limiting and Quota Management

### Intelligent Rate Limiting

```python
class RateLimiter:
    """Advanced rate limiting with predictive throttling."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = threading.Lock()
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without hitting rate limits."""
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            return len(self.request_times) < self.requests_per_minute
    
    def wait_if_needed(self) -> float:
        """Wait if necessary to avoid rate limiting."""
        with self.lock:
            if not self.can_make_request():
                # Calculate wait time until oldest request expires
                oldest_request = min(self.request_times)
                wait_time = 60 - (time.time() - oldest_request)
                
                if wait_time > 0:
                    print(f"⏳ Rate limit protection: waiting {wait_time:.1f}s")
                    time.sleep(wait_time + 0.1)  # Small buffer
                    return wait_time
        
        return 0
    
    def record_request(self):
        """Record a request timestamp."""
        with self.lock:
            self.request_times.append(time.time())

class QuotaManager:
    """Manage daily/monthly API quotas across endpoints."""
    
    def __init__(self):
        self.quotas = {}  # endpoint_name -> quota_info
        self.usage_file = "memories/api_quotas.json"
    
    def set_quota(self, endpoint_name: str, daily_limit: int, monthly_limit: int = None):
        """Set quota limits for an endpoint."""
        self.quotas[endpoint_name] = {
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit or daily_limit * 30,
            "daily_usage": 0,
            "monthly_usage": 0,
            "last_reset": time.time()
        }
        self._save_quotas()
    
    def can_use_endpoint(self, endpoint_name: str) -> bool:
        """Check if endpoint is within quota limits."""
        if endpoint_name not in self.quotas:
            return True
        
        quota_info = self.quotas[endpoint_name]
        current_time = time.time()
        
        # Reset daily usage if day has passed
        if current_time - quota_info["last_reset"] >= 86400:
            quota_info["daily_usage"] = 0
            quota_info["last_reset"] = current_time
        
        # Reset monthly usage if month has passed (approximately)
        if current_time - quota_info["last_reset"] >= 86400 * 30:
            quota_info["monthly_usage"] = 0
        
        return (quota_info["daily_usage"] < quota_info["daily_limit"] and
                quota_info["monthly_usage"] < quota_info["monthly_limit"])
    
    def record_usage(self, endpoint_name: str, request_count: int = 1):
        """Record API usage for quota tracking."""
        if endpoint_name in self.quotas:
            self.quotas[endpoint_name]["daily_usage"] += request_count
            self.quotas[endpoint_name]["monthly_usage"] += request_count
            self._save_quotas()
    
    def _save_quotas(self):
        """Save quota information to persistent storage."""
        os.makedirs(os.path.dirname(self.usage_file), exist_ok=True)
        with open(self.usage_file, 'w') as f:
            json.dump(self.quotas, f, indent=2)
    
    def _load_quotas(self):
        """Load quota information from persistent storage."""
        try:
            with open(self.usage_file, 'r') as f:
                self.quotas = json.load(f)
        except FileNotFoundError:
            self.quotas = {}
```

## Load Balancing Strategies

### Geographic Load Balancing

```python
class GeographicAPIManager(APIRotationManager):
    """API manager with geographic load balancing."""
    
    def __init__(self, user_region: str = "us-east"):
        super().__init__()
        self.user_region = user_region
        self.region_preferences = {
            "us-east": ["us-east-1", "us-east-2", "us-west-1", "eu-west-1"],
            "us-west": ["us-west-1", "us-west-2", "us-east-1", "eu-west-1"],
            "eu-west": ["eu-west-1", "eu-central-1", "us-east-1", "us-west-1"],
            "ap-southeast": ["ap-southeast-1", "ap-northeast-1", "us-west-1", "eu-west-1"]
        }
    
    def add_regional_endpoint(self, name: str, base_url: str, api_key: str, 
                             region: str, **kwargs):
        """Add endpoint with regional information."""
        endpoint = APIEndpoint(name=name, base_url=base_url, api_key=api_key, **kwargs)
        endpoint.region = region
        
        # Set priority based on region preference
        preferred_regions = self.region_preferences.get(self.user_region, [])
        if region in preferred_regions:
            endpoint.priority = len(preferred_regions) - preferred_regions.index(region)
        else:
            endpoint.priority = 1
        
        self.endpoints.append(endpoint)
        print(f"🌍 Added regional endpoint: {name} ({region}) - priority: {endpoint.priority}")
    
    def get_regional_status(self) -> Dict:
        """Get status report grouped by region."""
        regional_status = {}
        
        for endpoint in self.endpoints:
            region = getattr(endpoint, 'region', 'unknown')
            if region not in regional_status:
                regional_status[region] = {
                    "endpoints": [],
                    "active_count": 0,
                    "total_usage": 0
                }
            
            regional_status[region]["endpoints"].append({
                "name": endpoint.name,
                "status": endpoint.status.value,
                "usage": f"{endpoint.daily_usage}/{endpoint.quota_limit}",
                "can_request": endpoint.can_make_request()
            })
            
            if endpoint.status == APIStatus.ACTIVE:
                regional_status[region]["active_count"] += 1
            
            regional_status[region]["total_usage"] += endpoint.daily_usage
        
        return regional_status
```

### Performance-Based Selection

```python
class PerformanceTrackingManager(APIRotationManager):
    """API manager that tracks performance metrics for optimal selection."""
    
    def __init__(self):
        super().__init__()
        self.performance_metrics = {}  # endpoint_name -> metrics
    
    def record_performance(self, endpoint_name: str, response_time: float, 
                          success: bool, error_type: str = None):
        """Record performance metrics for an endpoint."""
        if endpoint_name not in self.performance_metrics:
            self.performance_metrics[endpoint_name] = {
                "response_times": [],
                "success_count": 0,
                "error_count": 0,
                "error_types": {},
                "last_success": 0,
                "consecutive_failures": 0
            }
        
        metrics = self.performance_metrics[endpoint_name]
        
        # Record response time
        metrics["response_times"].append(response_time)
        # Keep only last 100 response times
        if len(metrics["response_times"]) > 100:
            metrics["response_times"] = metrics["response_times"][-100:]
        
        # Record success/failure
        if success:
            metrics["success_count"] += 1
            metrics["last_success"] = time.time()
            metrics["consecutive_failures"] = 0
        else:
            metrics["error_count"] += 1
            metrics["consecutive_failures"] += 1
            
            if error_type:
                metrics["error_types"][error_type] = metrics["error_types"].get(error_type, 0) + 1
    
    def get_performance_score(self, endpoint_name: str) -> float:
        """Calculate performance score for endpoint selection."""
        if endpoint_name not in self.performance_metrics:
            return 0.5  # Neutral score for new endpoints
        
        metrics = self.performance_metrics[endpoint_name]
        
        # Success rate (0-1)
        total_requests = metrics["success_count"] + metrics["error_count"]
        success_rate = metrics["success_count"] / max(1, total_requests)
        
        # Average response time score (lower is better, normalized to 0-1)
        if metrics["response_times"]:
            avg_response_time = sum(metrics["response_times"]) / len(metrics["response_times"])
            # Assume 5 seconds is very slow, 0.1 seconds is very fast
            response_score = max(0, 1 - (avg_response_time - 0.1) / 4.9)
        else:
            response_score = 0.5
        
        # Recency score (more recent success = higher score)
        recency_score = 0.5
        if metrics["last_success"] > 0:
            time_since_success = time.time() - metrics["last_success"]
            # Recent success within 1 hour gets full score
            recency_score = max(0, 1 - (time_since_success / 3600))
        
        # Consecutive failure penalty
        failure_penalty = max(0, 1 - (metrics["consecutive_failures"] / 10))
        
        # Combined performance score
        return (success_rate * 0.4 + response_score * 0.3 + 
                recency_score * 0.2 + failure_penalty * 0.1)
    
    def select_optimal_endpoint(self, available: List[APIEndpoint]) -> APIEndpoint:
        """Select optimal endpoint based on performance metrics."""
        if not available:
            return None
        
        # Calculate performance scores
        scored_endpoints = []
        for endpoint in available:
            capacity_score = endpoint.get_capacity_score()
            performance_score = self.get_performance_score(endpoint.name)
            
            # Combined score
            total_score = capacity_score * 0.6 + performance_score * 0.4
            scored_endpoints.append((endpoint, total_score))
        
        # Sort by score (highest first)
        scored_endpoints.sort(key=lambda x: x[1], reverse=True)
        
        # Use weighted random selection from top performers
        top_performers = scored_endpoints[:max(1, len(scored_endpoints) // 2)]
        
        if len(top_performers) == 1:
            return top_performers[0][0]
        
        # Weighted random selection
        weights = [score for _, score in top_performers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return available[0]
        
        rand_val = random.uniform(0, total_weight)
        current_weight = 0
        
        for (endpoint, weight) in top_performers:
            current_weight += weight
            if rand_val <= current_weight:
                return endpoint
        
        return top_performers[0][0]
```

## Error Handling and Recovery

### API-Specific Error Classification

```python
class APIErrorHandler:
    """Handle API-specific errors with appropriate recovery strategies."""
    
    ERROR_PATTERNS = {
        "rate_limit": [
            "rate limit", "429", "too many requests", "quota exceeded",
            "requests per minute", "throttled"
        ],
        "auth_error": [
            "unauthorized", "401", "invalid api key", "authentication failed",
            "forbidden", "403", "access denied"
        ],
        "server_error": [
            "500", "502", "503", "504", "internal server error",
            "bad gateway", "service unavailable", "gateway timeout"
        ],
        "network_error": [
            "connection", "timeout", "network", "dns", "unreachable",
            "connection reset", "connection refused"
        ],
        "quota_error": [
            "quota", "billing", "usage limit", "monthly limit",
            "daily limit", "exceeded limit"
        ]
    }
    
    @classmethod
    def classify_error(cls, error_message: str) -> str:
        """Classify API error for appropriate handling."""
        error_lower = error_message.lower()
        
        for error_type, patterns in cls.ERROR_PATTERNS.items():
            if any(pattern in error_lower for pattern in patterns):
                return error_type
        
        return "unknown"
    
    @classmethod
    def get_recovery_strategy(cls, error_type: str, endpoint: APIEndpoint) -> Dict:
        """Get recovery strategy for specific error type."""
        
        strategies = {
            "rate_limit": {
                "action": "rotate_and_wait",
                "wait_time": 60,
                "disable_endpoint": False,
                "retry_count": 3
            },
            "auth_error": {
                "action": "disable_endpoint",
                "wait_time": 0,
                "disable_endpoint": True,
                "retry_count": 0
            },
            "server_error": {
                "action": "retry_with_backoff",
                "wait_time": 5,
                "disable_endpoint": False,
                "retry_count": 3
            },
            "network_error": {
                "action": "retry_with_backoff", 
                "wait_time": 2,
                "disable_endpoint": False,
                "retry_count": 5
            },
            "quota_error": {
                "action": "disable_until_reset",
                "wait_time": 3600,  # 1 hour
                "disable_endpoint": True,
                "retry_count": 0
            },
            "unknown": {
                "action": "retry_with_caution",
                "wait_time": 10,
                "disable_endpoint": False,
                "retry_count": 2
            }
        }
        
        return strategies.get(error_type, strategies["unknown"])
    
    @classmethod
    def handle_api_error(cls, error: Exception, endpoint: APIEndpoint, 
                        api_manager: APIRotationManager) -> Dict:
        """Handle API error with appropriate recovery strategy."""
        
        error_type = cls.classify_error(str(error))
        strategy = cls.get_recovery_strategy(error_type, endpoint)
        
        print(f"🚨 API Error on {endpoint.name}: {error_type}")
        print(f"🔧 Recovery strategy: {strategy['action']}")
        
        # Apply recovery strategy
        if strategy["disable_endpoint"]:
            if error_type == "quota_error":
                endpoint.status = APIStatus.QUOTA_EXCEEDED
            else:
                endpoint.status = APIStatus.ERROR
        
        if strategy["wait_time"] > 0:
            print(f"⏳ Waiting {strategy['wait_time']}s before retry...")
            time.sleep(strategy["wait_time"])
        
        # Record error for learning
        endpoint.record_request(success=False)
        
        return {
            "error_type": error_type,
            "strategy_applied": strategy["action"],
            "endpoint_disabled": strategy["disable_endpoint"],
            "wait_time": strategy["wait_time"],
            "retry_recommended": strategy["retry_count"] > 0
        }
```

## Integration with Task Execution

### API-Aware Task Execution

```python
def setup_task_api_rotation(api_configs: List[Dict]):
    """Setup API rotation for task execution."""
    
    # Create performance-tracking manager
    global api_manager
    api_manager = PerformanceTrackingManager()
    
    # Add all configured endpoints
    for config in api_configs:
        api_manager.add_endpoint(
            name=config["name"],
            base_url=config["base_url"],
            api_key=config["api_key"],
            rate_limit=config.get("rate_limit", 60),
            quota_limit=config.get("quota_limit", 1000)
        )
        
        # Set priority if specified
        if "priority" in config:
            for endpoint in api_manager.endpoints:
                if endpoint.name == config["name"]:
                    endpoint.priority = config["priority"]
                    break
    
    print(f"🔄 API rotation configured with {len(api_configs)} endpoints")
    
    # Save API configuration for persistence
    save_json_file("memories/api_config.json", api_configs)

def execute_task_with_api_calls(task: Dict, api_calls_needed: List[str]) -> Dict:
    """Execute task that requires multiple API calls with rotation."""
    
    task_id = task["id"]
    print(f"🔄 Executing {task_id} with API rotation ({len(api_calls_needed)} calls needed)")
    
    # Track API usage for this task
    task_api_stats = {
        "calls_made": 0,
        "calls_successful": 0,
        "calls_failed": 0,
        "endpoints_used": set(),
        "total_response_time": 0
    }
    
    results = []
    
    for i, api_call in enumerate(api_calls_needed):
        print(f"📡 Making API call {i+1}/{len(api_calls_needed)}: {api_call}")
        
        start_time = time.time()
        
        # Make API call with rotation
        result = api_manager.make_api_request(
            make_http_request,
            api_call,
            method="GET"
        )
        
        response_time = time.time() - start_time
        task_api_stats["total_response_time"] += response_time
        task_api_stats["calls_made"] += 1
        
        if result["success"]:
            task_api_stats["calls_successful"] += 1
            task_api_stats["endpoints_used"].add(result["endpoint_used"])
            results.append(result["data"])
            
            print(f"✅ API call successful ({response_time:.2f}s) via {result['endpoint_used']}")
        else:
            task_api_stats["calls_failed"] += 1
            print(f"❌ API call failed: {result['error']}")
            
            # Decide whether to continue or fail the task
            if task_api_stats["calls_failed"] > len(api_calls_needed) * 0.3:  # 30% failure rate
                return {
                    "success": False,
                    "error": f"Too many API failures ({task_api_stats['calls_failed']}/{task_api_stats['calls_made']})",
                    "task_id": task_id,
                    "api_stats": task_api_stats
                }
    
    # Calculate task success metrics
    success_rate = task_api_stats["calls_successful"] / task_api_stats["calls_made"] * 100
    avg_response_time = task_api_stats["total_response_time"] / task_api_stats["calls_made"]
    
    return {
        "success": True,
        "output": f"Task {task_id} completed with {task_api_stats['calls_successful']} successful API calls",
        "results": results,
        "api_stats": {
            **task_api_stats,
            "success_rate": success_rate,
            "average_response_time": avg_response_time,
            "endpoints_used": list(task_api_stats["endpoints_used"])
        }
    }

def make_http_request(url: str, method: str = "GET", **kwargs) -> Any:
    """Make HTTP request (placeholder - implement with requests library)."""
    import requests
    
    if method.upper() == "GET":
        response = requests.get(url, **kwargs)
    elif method.upper() == "POST":
        response = requests.post(url, **kwargs)
    elif method.upper() == "PUT":
        response = requests.put(url, **kwargs)
    elif method.upper() == "DELETE":
        response = requests.delete(url, **kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")
    
    response.raise_for_status()
    return response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
```

## Usage Examples

### Basic API Rotation Setup

```python
# Configure multiple API endpoints
api_configs = [
    {
        "name": "openai_primary",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-primary-key-here",
        "rate_limit": 100,
        "quota_limit": 10000,
        "priority": 3
    },
    {
        "name": "openai_backup",
        "base_url": "https://api.openai.com/v1", 
        "api_key": "sk-backup-key-here",
        "rate_limit": 60,
        "quota_limit": 5000,
        "priority": 2
    },
    {
        "name": "anthropic_claude",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "sk-ant-key-here",
        "rate_limit": 50,
        "quota_limit": 8000,
        "priority": 2
    }
]

# Setup API rotation
setup_task_api_rotation(api_configs)

# Execute tasks with API rotation
task_list = load_task_list()
for task in task_list["tasks"]:
    if task["status"] == "pending":
        # Determine if task needs API calls
        api_calls = analyze_task_api_requirements(task)
        
        if api_calls:
            result = execute_task_with_api_calls(task, api_calls)
        else:
            result = execute_task_by_category(task)
        
        # Update task status
        if result["success"]:
            update_task_status(task["id"], "completed", result["output"])
        else:
            update_task_status(task["id"], "failed", "", "", result["error"])
```

### Advanced Configuration with Regional Endpoints

```python
# Setup geographic API rotation
geo_manager = GeographicAPIManager(user_region="us-east")

regional_configs = [
    {
        "name": "openai_us_east",
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-us-east-key",
        "region": "us-east-1",
        "rate_limit": 120,
        "quota_limit": 15000
    },
    {
        "name": "openai_eu_west", 
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-eu-west-key",
        "region": "eu-west-1",
        "rate_limit": 100,
        "quota_limit": 12000
    },
    {
        "name": "anthropic_us",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "sk-ant-us-key",
        "region": "us-east-1", 
        "rate_limit": 80,
        "quota_limit": 10000
    }
]

for config in regional_configs:
    geo_manager.add_regional_endpoint(**config)

# Get regional status
regional_status = geo_manager.get_regional_status()
print("🌍 Regional API Status:")
for region, status in regional_status.items():
    print(f"  {region}: {status['active_count']} active endpoints, {status['total_usage']} total usage")
```

### Step 4: Implement State Management