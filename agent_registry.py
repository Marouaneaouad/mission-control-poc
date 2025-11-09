"""
Agent Registry System
Centralized configuration and management for all AI agents in the enterprise
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime

@dataclass
class AgentConfig:
    """Configuration for a single AI agent"""
    agent_id: str
    agent_name: str
    display_name: str
    agent_type: str  # "assistant", "specialist", "coach", "analyst"
    description: str
    model_id: str
    capabilities: List[str]
    slack_channels: List[str]
    department: str
    status: str  # "active", "maintenance", "archived"
    created_date: str
    avatar_emoji: str
    theme_color: str  # Hex color for UI theming
    
    # Cost configuration
    input_cost_per_mtok: float  # Cost per million input tokens
    output_cost_per_mtok: float  # Cost per million output tokens
    
    # Performance SLAs
    target_latency_ms: int
    target_success_rate: float
    
    # Alert thresholds
    daily_cost_threshold: Optional[float] = None
    monthly_cost_threshold: Optional[float] = None
    error_rate_threshold: float = 5.0  # Percentage


# Model Pricing Database
MODEL_PRICING = {
    "anthropic.claude-3-haiku-20240307-v1:0": {
        "name": "Claude 3 Haiku",
        "input": 0.25,
        "output": 1.25
    },
    "anthropic.claude-haiku-4-5-20251001-v1:0": {
        "name": "Claude 4.5 Haiku",
        "input": 1.0,
        "output": 5.0
    },
    "anthropic.claude-3-5-haiku-20241022-v1:0": {
        "name": "Claude 3.5 Haiku",
        "input": 1.0,
        "output": 5.0
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "name": "Claude 3.5 Sonnet v2",
        "input": 3.0,
        "output": 15.0
    },
    "anthropic.claude-sonnet-4-20250514": {
        "name": "Claude Sonnet 4",
        "input": 3.0,
        "output": 15.0
    },
    "unknown_model_id": {
        "name": "Unknown Model",
        "input": 1.0,
        "output": 5.0
    }
}


class AgentRegistry:
    """Centralized registry for managing all agents in the enterprise"""
    
    def __init__(self):
        self.agents: Dict[str, AgentConfig] = {}
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize the three POC agents"""
        
        # Patrick - Partnership Intelligence Agent
        patrick = AgentConfig(
            agent_id="patrick-ops-agent",
            agent_name="Patrick",
            display_name="Patrick - Partnership Intelligence",
            agent_type="assistant",
            description="Ellucian's internal Partnerships AI assistant. Provides information about partners, solutions, badges, and EPN program policies.",
            model_id="anthropic.claude-haiku-4-5-20251001-v1:0",
            capabilities=[
                "Partner lookup",
                "Solution verification",
                "Badge intelligence",
                "EPN policy guidance",
                "SaaS verification"
            ],
            slack_channels=["#partnerships", "#partner-support"],
            department="Partnerships",
            status="active",
            created_date="2024-10-15",
            avatar_emoji="🤝",
            theme_color="#2E86AB",
            input_cost_per_mtok=1.0,
            output_cost_per_mtok=5.0,
            target_latency_ms=5000,
            target_success_rate=98.0,
            daily_cost_threshold=50.0,
            monthly_cost_threshold=1500.0
        )
        
        # Marvin - Motivation Coach
        marvin = AgentConfig(
            agent_id="marvin-ops-agent",
            agent_name="Marvin",
            display_name="Marvin - Motivation Coach",
            agent_type="coach",
            description="Your official team Motivation Coach. Provides positive encouragement, emotional support, and helps team members bring their A-game.",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            capabilities=[
                "Motivational coaching",
                "Emotional support",
                "Positivity reinforcement",
                "Goal encouragement"
            ],
            slack_channels=["#wellness", "#team-motivation"],
            department="People & Culture",
            status="active",
            created_date="2024-11-01",
            avatar_emoji="💪",
            theme_color="#F18F01",
            input_cost_per_mtok=0.25,
            output_cost_per_mtok=1.25,
            target_latency_ms=3000,
            target_success_rate=95.0,
            daily_cost_threshold=20.0,
            monthly_cost_threshold=600.0
        )
        
        # Eva - Culinary Expert
        eva = AgentConfig(
            agent_id="eva-ops-agent",
            agent_name="Eva",
            display_name="Eva - Culinary Expert",
            agent_type="specialist",
            description="Award-winning chef AI. Helps employees discover true culinary art, provides recipes, and guides cooking techniques.",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            capabilities=[
                "Recipe creation",
                "Culinary guidance",
                "Ingredient recommendations",
                "Cooking techniques"
            ],
            slack_channels=["#food-and-recipes", "#break-room"],
            department="Employee Engagement",
            status="active",
            created_date="2024-11-01",
            avatar_emoji="👨‍🍳",
            theme_color="#C73E1D",
            input_cost_per_mtok=0.25,
            output_cost_per_mtok=1.25,
            target_latency_ms=4000,
            target_success_rate=96.0,
            daily_cost_threshold=15.0,
            monthly_cost_threshold=450.0
        )
        
        self.agents[patrick.agent_id] = patrick
        self.agents[marvin.agent_id] = marvin
        self.agents[eva.agent_id] = eva
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentConfig]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def get_active_agents(self) -> List[AgentConfig]:
        """Get only active agents"""
        return [agent for agent in self.agents.values() if agent.status == "active"]
    
    def add_agent(self, agent: AgentConfig):
        """Add a new agent to the registry"""
        self.agents[agent.agent_id] = agent
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
    
    def get_agent_by_name(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent by display name"""
        for agent in self.agents.values():
            if agent.agent_name.lower() == agent_name.lower():
                return agent
        return None
    
    def get_model_pricing(self, model_id: str) -> Dict[str, float]:
        """Get pricing for a specific model"""
        return MODEL_PRICING.get(model_id, MODEL_PRICING["unknown_model_id"])


def get_registry() -> AgentRegistry:
    """Singleton factory for agent registry"""
    if not hasattr(get_registry, "_instance"):
        get_registry._instance = AgentRegistry()
    return get_registry._instance
