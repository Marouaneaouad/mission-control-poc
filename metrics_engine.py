"""
Enterprise Metrics Engine
Advanced analytics and cost calculation for multi-agent observability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from agent_registry import get_registry, MODEL_PRICING


class MetricsEngine:
    """Advanced metrics calculation engine for multi-agent observability"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame containing log data
        Expected columns: agent_id, agentName, timestamp, inputTokens, outputTokens,
                         agentLatency, feedbackStatus, status, modelId, etc.
        """
        self.df = df
        self.registry = get_registry()
    
    def calculate_cost(self, row: pd.Series) -> float:
        """
        Calculate cost for a single interaction using dynamic model pricing.
        CRITICAL: Each query can use a different model, so we calculate cost per row.
        """
        model_id = row.get('modelId', 'unknown_model_id')
        
        # Handle cases where modelId might be missing or empty
        if pd.isna(model_id) or model_id == '' or model_id is None:
            model_id = 'unknown_model_id'
        
        # Get pricing for this specific model
        pricing = MODEL_PRICING.get(model_id, MODEL_PRICING['unknown_model_id'])
        
        input_tokens = row.get('inputTokens', 0)
        output_tokens = row.get('outputTokens', 0)
        
        # Handle NaN values
        if pd.isna(input_tokens):
            input_tokens = 0
        if pd.isna(output_tokens):
            output_tokens = 0
        
        input_cost = (float(input_tokens) / 1_000_000) * pricing['input']
        output_cost = (float(output_tokens) / 1_000_000) * pricing['output']
        
        return input_cost + output_cost
    
    def get_fleet_metrics(self) -> Dict:
        """Calculate enterprise-wide fleet metrics across all agents"""
        if self.df.empty:
            return self._empty_fleet_metrics()
        
        # Add cost column
        self.df['cost'] = self.df.apply(self.calculate_cost, axis=1)
        
        total_agents = self.df['agent_id'].nunique()
        total_queries = len(self.df)
        total_sessions = self.df['sessionId'].nunique()
        
        # Latency metrics (convert from ms to seconds)
        avg_latency_sec = self.df['agentLatency'].mean() / 1000
        p95_latency_sec = self.df['agentLatency'].quantile(0.95) / 1000
        
        # Feedback metrics - ONLY count positive and negative (exclude empty)
        feedback_data = self.df[
            (self.df['feedbackStatus'] == 'positive') | 
            (self.df['feedbackStatus'] == 'negative')
        ]
        total_feedback = len(feedback_data)
        positive_feedback = len(feedback_data[feedback_data['feedbackStatus'] == 'positive'])
        negative_feedback = len(feedback_data[feedback_data['feedbackStatus'] == 'negative'])
        positive_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        # Error metrics
        total_errors = (self.df['status'] != 'SUCCESS').sum()
        error_rate = (total_errors / total_queries * 100) if total_queries > 0 else 0
        
        # Cost metrics
        total_cost = self.df['cost'].sum()
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0
        
        # Token metrics
        total_input_tokens = self.df['inputTokens'].sum()
        total_output_tokens = self.df['outputTokens'].sum()
        
        # Calculate fleet health score (0-100)
        health_score = self._calculate_fleet_health(
            error_rate=error_rate,
            avg_latency_sec=avg_latency_sec,
            positive_rate=positive_rate
        )
        
        # Daily cost projection
        days_in_period = (self.df['timestamp'].max() - self.df['timestamp'].min()).days
        days_in_period = max(days_in_period, 1)
        daily_avg_cost = total_cost / days_in_period
        monthly_projection = daily_avg_cost * 30
        
        return {
            "total_agents": total_agents,
            "total_queries": total_queries,
            "total_sessions": total_sessions,
            "avg_latency_sec": avg_latency_sec,
            "p95_latency_sec": p95_latency_sec,
            "positive_feedback_rate": positive_rate,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "total_cost": total_cost,
            "avg_cost_per_query": avg_cost_per_query,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "health_score": health_score,
            "daily_avg_cost": daily_avg_cost,
            "monthly_projection": monthly_projection,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback
        }
    
    def get_agent_metrics(self, agent_id: str) -> Dict:
        """Calculate detailed metrics for a specific agent"""
        agent_df = self.df[self.df['agent_id'] == agent_id].copy()
        
        if agent_df.empty:
            return self._empty_agent_metrics(agent_id)
        
        # Add cost column
        agent_df['cost'] = agent_df.apply(self.calculate_cost, axis=1)
        
        agent_config = self.registry.get_agent(agent_id)
        
        total_queries = len(agent_df)
        total_sessions = agent_df['sessionId'].nunique()
        
        # Latency metrics (convert from ms to seconds)
        avg_latency_sec = agent_df['agentLatency'].mean() / 1000
        p50_latency_sec = agent_df['agentLatency'].quantile(0.50) / 1000
        p90_latency_sec = agent_df['agentLatency'].quantile(0.90) / 1000
        p95_latency_sec = agent_df['agentLatency'].quantile(0.95) / 1000
        p99_latency_sec = agent_df['agentLatency'].quantile(0.99) / 1000
        
        # Feedback metrics - ONLY count positive and negative (exclude empty)
        feedback_data = agent_df[
            (agent_df['feedbackStatus'] == 'positive') | 
            (agent_df['feedbackStatus'] == 'negative')
        ]
        total_feedback = len(feedback_data)
        positive_feedback = len(feedback_data[feedback_data['feedbackStatus'] == 'positive'])
        negative_feedback = len(feedback_data[feedback_data['feedbackStatus'] == 'negative'])
        positive_rate = (positive_feedback / total_feedback * 100) if total_feedback > 0 else 0
        
        # Error metrics
        total_errors = (agent_df['status'] != 'SUCCESS').sum()
        error_rate = (total_errors / total_queries * 100) if total_queries > 0 else 0
        success_rate = 100 - error_rate
        
        # Cost metrics
        total_cost = agent_df['cost'].sum()
        avg_cost_per_query = total_cost / total_queries if total_queries > 0 else 0
        
        # Token metrics
        total_input_tokens = agent_df['inputTokens'].sum()
        total_output_tokens = agent_df['outputTokens'].sum()
        avg_tokens_per_query = (total_input_tokens + total_output_tokens) / total_queries if total_queries > 0 else 0
        token_efficiency = total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0
        
        # Usage patterns
        agent_df['hour'] = agent_df['timestamp'].dt.hour
        agent_df['day_of_week'] = agent_df['timestamp'].dt.day_name()
        peak_hour = agent_df['hour'].mode().values[0] if not agent_df['hour'].mode().empty else 0
        
        # Session metrics
        avg_messages_per_session = total_queries / total_sessions if total_sessions > 0 else 0
        
        # SLA compliance (convert target from ms to seconds for comparison)
        target_latency_sec = agent_config.target_latency_ms / 1000 if agent_config else 5.0
        sla_latency_compliance = (agent_df['agentLatency'] / 1000 <= target_latency_sec).mean() * 100
        sla_success_compliance = success_rate >= (agent_config.target_success_rate if agent_config else 95)
        
        # Daily metrics for projections
        days_in_period = (agent_df['timestamp'].max() - agent_df['timestamp'].min()).days
        days_in_period = max(days_in_period, 1)
        daily_avg_cost = total_cost / days_in_period
        daily_avg_queries = total_queries / days_in_period
        monthly_cost_projection = daily_avg_cost * 30
        
        # Cost threshold checks
        cost_threshold_ok = True
        if agent_config and agent_config.daily_cost_threshold:
            cost_threshold_ok = daily_avg_cost <= agent_config.daily_cost_threshold
        
        return {
            "agent_id": agent_id,
            "agent_config": agent_config,
            "total_queries": total_queries,
            "total_sessions": total_sessions,
            "avg_latency_sec": avg_latency_sec,
            "p50_latency_sec": p50_latency_sec,
            "p90_latency_sec": p90_latency_sec,
            "p95_latency_sec": p95_latency_sec,
            "p99_latency_sec": p99_latency_sec,
            "positive_feedback_rate": positive_rate,
            "total_feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "success_rate": success_rate,
            "total_cost": total_cost,
            "avg_cost_per_query": avg_cost_per_query,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_tokens_per_query": avg_tokens_per_query,
            "token_efficiency": token_efficiency,
            "peak_hour": peak_hour,
            "avg_messages_per_session": avg_messages_per_session,
            "sla_latency_compliance": sla_latency_compliance,
            "sla_success_compliance": sla_success_compliance,
            "daily_avg_cost": daily_avg_cost,
            "daily_avg_queries": daily_avg_queries,
            "monthly_cost_projection": monthly_cost_projection,
            "cost_threshold_ok": cost_threshold_ok,
            "days_in_period": days_in_period,
            "target_latency_sec": target_latency_sec
        }
    
    def get_agent_comparison(self, agent_ids: List[str]) -> pd.DataFrame:
        """Compare metrics across multiple agents"""
        comparison_data = []
        
        for agent_id in agent_ids:
            metrics = self.get_agent_metrics(agent_id)
            agent_config = self.registry.get_agent(agent_id)
            
            comparison_data.append({
                "Agent": agent_config.display_name if agent_config else agent_id,
                "Queries": metrics['total_queries'],
                "Avg Latency (s)": round(metrics['avg_latency_sec'], 2),
                "Success Rate (%)": round(metrics['success_rate'], 1),
                "Positive Feedback (%)": round(metrics['positive_feedback_rate'], 1),
                "Total Cost ($)": round(metrics['total_cost'], 2),
                "Cost/Query ($)": round(metrics['avg_cost_per_query'], 4),
                "Monthly Projection ($)": round(metrics['monthly_cost_projection'], 2)
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_cost_breakdown(self) -> pd.DataFrame:
        """Get cost breakdown by agent"""
        if self.df.empty:
            return pd.DataFrame()
        
        self.df['cost'] = self.df.apply(self.calculate_cost, axis=1)
        
        cost_breakdown = self.df.groupby('agent_id').agg({
            'cost': 'sum',
            'interaction_id': 'count',
            'inputTokens': 'sum',
            'outputTokens': 'sum'
        }).reset_index()
        
        cost_breakdown.columns = ['agent_id', 'total_cost', 'queries', 'input_tokens', 'output_tokens']
        cost_breakdown['avg_cost_per_query'] = cost_breakdown['total_cost'] / cost_breakdown['queries']
        cost_breakdown['cost_percentage'] = (cost_breakdown['total_cost'] / cost_breakdown['total_cost'].sum() * 100)
        
        # Add agent names
        cost_breakdown['agent_name'] = cost_breakdown['agent_id'].apply(
            lambda x: self.registry.get_agent(x).display_name if self.registry.get_agent(x) else x
        )
        
        return cost_breakdown.sort_values('total_cost', ascending=False)
    
    def get_daily_metrics(self) -> pd.DataFrame:
        """Get daily aggregated metrics across all agents"""
        if self.df.empty:
            return pd.DataFrame()
        
        self.df['cost'] = self.df.apply(self.calculate_cost, axis=1)
        self.df['date'] = self.df['timestamp'].dt.date
        
        daily = self.df.groupby('date').agg({
            'interaction_id': 'count',
            'cost': 'sum',
            'agentLatency': lambda x: x.mean() / 1000,  # Convert to seconds
            'inputTokens': 'sum',
            'outputTokens': 'sum',
            'status': lambda x: (x != 'SUCCESS').sum()
        }).reset_index()
        
        daily.columns = ['date', 'queries', 'cost', 'avg_latency_sec', 'input_tokens', 'output_tokens', 'errors']
        daily['error_rate'] = (daily['errors'] / daily['queries'] * 100)
        
        return daily
    
    def get_agent_daily_metrics(self, agent_id: str) -> pd.DataFrame:
        """Get daily metrics for a specific agent"""
        agent_df = self.df[self.df['agent_id'] == agent_id].copy()
        
        if agent_df.empty:
            return pd.DataFrame()
        
        agent_df['cost'] = agent_df.apply(self.calculate_cost, axis=1)
        agent_df['date'] = agent_df['timestamp'].dt.date
        
        daily = agent_df.groupby('date').agg({
            'interaction_id': 'count',
            'cost': 'sum',
            'agentLatency': lambda x: x.mean() / 1000,  # Convert to seconds
            'inputTokens': 'sum',
            'outputTokens': 'sum',
            'status': lambda x: (x != 'SUCCESS').sum()
        }).reset_index()
        
        daily.columns = ['date', 'queries', 'cost', 'avg_latency_sec', 'input_tokens', 'output_tokens', 'errors']
        daily['error_rate'] = (daily['errors'] / daily['queries'] * 100)
        
        return daily
    
    def detect_anomalies(self, agent_id: Optional[str] = None) -> List[Dict]:
        """Detect anomalies in agent performance"""
        anomalies = []
        
        if agent_id:
            agent_df = self.df[self.df['agent_id'] == agent_id].copy()
        else:
            agent_df = self.df.copy()
        
        if agent_df.empty:
            return anomalies
        
        # Latency anomalies (>2 standard deviations from mean) - convert to seconds
        latency_sec = agent_df['agentLatency'] / 1000
        latency_mean = latency_sec.mean()
        latency_std = latency_sec.std()
        latency_threshold = latency_mean + (2 * latency_std)
        
        latency_anomalies = latency_sec[latency_sec > latency_threshold]
        if len(latency_anomalies) > 0:
            anomalies.append({
                "type": "latency_spike",
                "severity": "warning",
                "count": len(latency_anomalies),
                "message": f"{len(latency_anomalies)} queries exceeded latency threshold ({latency_threshold:.2f}s)"
            })
        
        # Error rate anomalies
        agent_df['date'] = agent_df['timestamp'].dt.date
        daily_errors = agent_df.groupby('date').apply(
            lambda x: (x['status'] != 'SUCCESS').sum() / len(x) * 100
        )
        
        if (daily_errors > 10).any():
            anomalies.append({
                "type": "high_error_rate",
                "severity": "critical",
                "count": (daily_errors > 10).sum(),
                "message": f"Error rate exceeded 10% on {(daily_errors > 10).sum()} days"
            })
        
        # Cost anomalies
        agent_df['cost'] = agent_df.apply(self.calculate_cost, axis=1)
        daily_costs = agent_df.groupby('date')['cost'].sum()
        
        if len(daily_costs) > 1:
            cost_mean = daily_costs.mean()
            cost_std = daily_costs.std()
            cost_threshold = cost_mean + (2 * cost_std)
            
            if (daily_costs > cost_threshold).any():
                anomalies.append({
                    "type": "cost_spike",
                    "severity": "warning",
                    "count": (daily_costs > cost_threshold).sum(),
                    "message": f"Daily cost exceeded ${cost_threshold:.2f} on {(daily_costs > cost_threshold).sum()} days"
                })
        
        return anomalies
    
    def _calculate_fleet_health(self, error_rate: float, avg_latency_sec: float, positive_rate: float) -> float:
        """Calculate overall fleet health score (0-100)"""
        # Error rate component (0-40 points) - lower is better
        error_score = max(0, 40 - (error_rate * 4))
        
        # Latency component (0-30 points) - lower is better
        # Target: under 5 seconds gets full points
        latency_score = max(0, 30 - (avg_latency_sec * 6))
        
        # Feedback component (0-30 points) - higher is better
        feedback_score = (positive_rate / 100) * 30
        
        return min(100, error_score + latency_score + feedback_score)
    
    def _empty_fleet_metrics(self) -> Dict:
        """Return empty fleet metrics"""
        return {
            "total_agents": 0,
            "total_queries": 0,
            "total_sessions": 0,
            "avg_latency_sec": 0,
            "p95_latency_sec": 0,
            "positive_feedback_rate": 0,
            "total_errors": 0,
            "error_rate": 0,
            "total_cost": 0,
            "avg_cost_per_query": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "health_score": 0,
            "daily_avg_cost": 0,
            "monthly_projection": 0,
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0
        }
    
    def _empty_agent_metrics(self, agent_id: str) -> Dict:
        """Return empty agent metrics"""
        agent_config = self.registry.get_agent(agent_id)
        target_latency_sec = agent_config.target_latency_ms / 1000 if agent_config else 5.0
        
        return {
            "agent_id": agent_id,
            "agent_config": agent_config,
            "total_queries": 0,
            "total_sessions": 0,
            "avg_latency_sec": 0,
            "p50_latency_sec": 0,
            "p90_latency_sec": 0,
            "p95_latency_sec": 0,
            "p99_latency_sec": 0,
            "positive_feedback_rate": 0,
            "total_feedback": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "total_errors": 0,
            "error_rate": 0,
            "success_rate": 0,
            "total_cost": 0,
            "avg_cost_per_query": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_tokens_per_query": 0,
            "token_efficiency": 0,
            "peak_hour": 0,
            "avg_messages_per_session": 0,
            "sla_latency_compliance": 0,
            "sla_success_compliance": False,
            "daily_avg_cost": 0,
            "daily_avg_queries": 0,
            "monthly_cost_projection": 0,
            "cost_threshold_ok": True,
            "days_in_period": 0,
            "target_latency_sec": target_latency_sec
        }
