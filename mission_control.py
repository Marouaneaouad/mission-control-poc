"""
Enterprise AI Observability Platform - Mission Control
Multi-agent observability and management platform for Ellucian
"""

import streamlit as st
import pandas as pd
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Attr
from decimal import Decimal
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from agent_registry import get_registry, AgentConfig
from metrics_engine import MetricsEngine
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Mission Control - AI Observability",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for enhanced UI ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .health-excellent { color: #10B981; font-weight: 700; }
    .health-good { color: #3B82F6; font-weight: 700; }
    .health-warning { color: #F59E0B; font-weight: 700; }
    .health-critical { color: #EF4444; font-weight: 700; }
    .status-active { 
        background-color: #10B981; 
        color: white; 
        padding: 0.25rem 0.75rem; 
        border-radius: 12px; 
        font-size: 0.875rem;
        font-weight: 600;
    }
    .rationale-thinking {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .rationale-tool {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }
    .rationale-answer {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    .feedback-negative {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Secrets Handling ---
try:
    AWS_ACCESS_KEY_ID = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_REGION = st.secrets["AWS_DEFAULT_REGION"]
    APP_PASSWORD = st.secrets.get("APP_PASSWORD", "admin")
    DYNAMODB_TABLE_NAME = st.secrets.get("DYNAMODB_TABLE_NAME", "PatrickUsageLogs")
except (FileNotFoundError, KeyError):
    load_dotenv()
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_DEFAULT_REGION")
    APP_PASSWORD = os.getenv("APP_PASSWORD", "admin")
    DYNAMODB_TABLE_NAME = os.getenv("DYNAMODB_TABLE_NAME", "PatrickUsageLogs")

# --- Helper function to parse agent rationale ---
def parse_agent_rationale(rationale_text):
    """
    Parse agent rationale and extract thinking, tool calls, and answers.
    Returns structured data for display.
    """
    if not rationale_text or pd.isna(rationale_text) or rationale_text == '':
        return []
    
    steps = []
    
    # Split by step markers
    step_pattern = r'(?:---\s*Step\s*\d+:|→|←)'
    parts = re.split(step_pattern, str(rationale_text))
    
    for part in parts:
        if not part.strip():
            continue
            
        # Extract thinking
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', part, re.DOTALL)
        if thinking_match:
            steps.append({
                'type': 'thinking',
                'content': thinking_match.group(1).strip()
            })
        
        # Extract tool calls
        tool_match = re.search(r'Calling Tool:\s*([^\(]+)\([^\)]*\)', part)
        if tool_match:
            steps.append({
                'type': 'tool_call',
                'content': f"Calling: {tool_match.group(1).strip()}"
            })
        
        # Extract tool results
        result_match = re.search(r'Tool Result:\s*(.+?)(?=→|←|---|\Z)', part, re.DOTALL)
        if result_match:
            steps.append({
                'type': 'tool_result',
                'content': result_match.group(1).strip()[:200]  # Limit length
            })
        
        # Extract answers
        answer_match = re.search(r'<answer[^>]*>(.*?)</answer>', part, re.DOTALL)
        if answer_match:
            steps.append({
                'type': 'answer',
                'content': answer_match.group(1).strip()
            })
    
    return steps

# --- Password Protection ---
def check_password():
    def password_entered():
        if st.session_state.get("password") == APP_PASSWORD:
            st.session_state["password_correct"] = True
            if "password" in st.session_state:
                del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.write("🔐 Please enter the password to access Mission Control.")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect. Please try again.")
        return False
    else:
        return True

# --- AWS Client Initializations ---
@st.cache_resource
def get_dynamodb_resource(access_key, secret_key, region):
    try:
        resource = boto3.resource("dynamodb", aws_access_key_id=access_key, 
                                 aws_secret_access_key=secret_key, region_name=region)
        return resource
    except Exception as e:
        st.error(f"Error initializing DynamoDB resource: {e}")
        return None

# --- Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_dynamodb_data(_dynamodb, table_name, days=7):
    """Fetch data from DynamoDB with caching"""
    try:
        table = _dynamodb.Table(table_name)
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
        cutoff_time_str = cutoff_time.isoformat()
        
        response = table.scan(
            FilterExpression=Attr('timestamp').gte(cutoff_time_str)
        )
        
        items = response.get('Items', [])
        
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression=Attr('timestamp').gte(cutoff_time_str),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            items.extend(response.get('Items', []))
        
        if not items:
            return pd.DataFrame()
        
        df = pd.DataFrame(items)
        
        # Convert Decimal to float
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill missing values
        df['feedbackStatus'] = df['feedbackStatus'].fillna('')
        df['feedbackReason'] = df['feedbackReason'].fillna('')
        df['agentLatency'] = df['agentLatency'].fillna(0)
        df['inputTokens'] = df['inputTokens'].fillna(0)
        df['outputTokens'] = df['outputTokens'].fillna(0)
        df['agentRationale'] = df['agentRationale'].fillna('')
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data from DynamoDB: {e}")
        return pd.DataFrame()

# --- Initialize Registry ---
registry = get_registry()

# --- Main Application ---
if check_password():
    
    # Initialize DynamoDB
    dynamodb = get_dynamodb_resource(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    
    # Sidebar Navigation
    st.sidebar.markdown("# 🎯 Mission Control")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Central Dashboard", "🤖 Agent Fleet", "💰 Cost Analytics", 
         "🔬 Session Explorer", "⚙️ Agent Management"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Stats")
    
    # Fetch data
    with st.spinner("Loading data..."):
        log_df = fetch_dynamodb_data(dynamodb, DYNAMODB_TABLE_NAME, days=7)
    
    if not log_df.empty:
        metrics_engine = MetricsEngine(log_df)
        fleet_metrics = metrics_engine.get_fleet_metrics()
        
        st.sidebar.metric("Active Agents", fleet_metrics['total_agents'])
        st.sidebar.metric("Total Queries (7d)", f"{fleet_metrics['total_queries']:,}")
        st.sidebar.metric("Fleet Health", f"{fleet_metrics['health_score']:.0f}%")
        st.sidebar.metric("Est. Monthly Cost", f"${fleet_metrics['monthly_projection']:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔄 Data refreshes every hour")
    st.sidebar.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # --- Page Routing ---
    if page == "🏠 Central Dashboard":
        st.markdown('<div class="main-header">🎯 Mission Control - Central Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Enterprise-wide AI agent observability and performance monitoring</div>', unsafe_allow_html=True)
        
        if log_df.empty:
            st.warning("⚠️ No data available for the selected time period.")
        else:
            metrics_engine = MetricsEngine(log_df)
            fleet_metrics = metrics_engine.get_fleet_metrics()
            
            # Fleet Health Score with explanation
            health_score = fleet_metrics['health_score']
            health_class = (
                "health-excellent" if health_score >= 90 else
                "health-good" if health_score >= 75 else
                "health-warning" if health_score >= 60 else
                "health-critical"
            )
            
            col_health, col_info = st.columns([2, 1])
            
            with col_health:
                st.markdown(f"""
                ### 🏥 Fleet Health Score: <span class="{health_class}">{health_score:.0f}%</span>
                """, unsafe_allow_html=True)
            
            with col_info:
                with st.expander("ℹ️ What is Fleet Health Score?"):
                    st.markdown("""
                    **Fleet Health Score** is a composite metric (0-100%) that measures overall system health:
                    
                    📊 **Components:**
                    - **Error Rate** (40 points): Lower errors = higher score
                    - **Response Time** (30 points): Faster responses = higher score  
                    - **User Satisfaction** (30 points): More positive feedback = higher score
                    
                    🎯 **Scoring:**
                    - **90-100%**: Excellent (Green) - System performing optimally
                    - **75-89%**: Good (Blue) - Minor issues, acceptable performance
                    - **60-74%**: Warning (Yellow) - Attention needed
                    - **<60%**: Critical (Red) - Immediate action required
                    """)
            
            st.markdown("---")
            
            # Key Metrics Row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Agents", fleet_metrics['total_agents'])
                st.metric("Total Sessions", f"{fleet_metrics['total_sessions']:,}")
            
            with col2:
                st.metric("Total Queries (7d)", f"{fleet_metrics['total_queries']:,}")
                daily_avg = fleet_metrics['total_queries'] / 7
                st.caption(f"~{daily_avg:.0f} queries/day")
            
            with col3:
                st.metric("Avg Response Time", f"{fleet_metrics['avg_latency_sec']:.2f}s")
                st.metric("P95 Latency", f"{fleet_metrics['p95_latency_sec']:.2f}s")
            
            with col4:
                st.metric("Success Rate", f"{100 - fleet_metrics['error_rate']:.1f}%")
                st.metric("Total Errors", f"{fleet_metrics['total_errors']:,}")
            
            with col5:
                st.metric("Positive Feedback", f"{fleet_metrics['positive_feedback_rate']:.1f}%")
                st.caption(f"{fleet_metrics['positive_feedback']} / {fleet_metrics['total_feedback']} rated")
            
            st.markdown("---")
            
            # Cost Overview
            st.markdown("### 💰 Cost Overview")
            
            col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
            
            with col_cost1:
                st.metric("Total Cost (7d)", f"${fleet_metrics['total_cost']:.2f}")
            
            with col_cost2:
                st.metric("Daily Average", f"${fleet_metrics['daily_avg_cost']:.2f}")
            
            with col_cost3:
                st.metric("Monthly Projection", f"${fleet_metrics['monthly_projection']:.2f}")
                st.caption("Based on current usage")
            
            with col_cost4:
                st.metric("Cost per Query", f"${fleet_metrics['avg_cost_per_query']:.4f}")
            
            st.markdown("---")
            
            # Agent Cost Breakdown
            col_breakdown, col_daily = st.columns([1, 1])
            
            with col_breakdown:
                st.markdown("#### 📊 Cost Distribution by Agent")
                cost_breakdown = metrics_engine.get_cost_breakdown()
                
                if not cost_breakdown.empty:
                    fig = px.pie(
                        cost_breakdown, 
                        values='total_cost', 
                        names='agent_name',
                        title='',
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_daily:
                st.markdown("#### 📈 Daily Query Volume")
                daily_metrics = metrics_engine.get_daily_metrics()
                
                if not daily_metrics.empty:
                    daily_metrics['date_str'] = pd.to_datetime(daily_metrics['date']).dt.strftime('%b %d')
                    
                    chart = alt.Chart(daily_metrics).mark_bar(color='#3B82F6').encode(
                        x=alt.X('date_str:N', sort=None, title='Date'),
                        y=alt.Y('queries:Q', title='Total Queries'),
                        tooltip=['date_str', alt.Tooltip('queries:Q', title='Queries')]
                    ).properties(height=300)
                    
                    st.altair_chart(chart, use_container_width=True)
            
            st.markdown("---")
            
            # Agent Performance Overview
            st.markdown("### 🤖 Agent Performance Overview")
            
            active_agents = registry.get_active_agents()
            
            for agent in active_agents:
                agent_metrics = metrics_engine.get_agent_metrics(agent.agent_id)
                
                with st.container():
                    col_info, col_metrics = st.columns([1, 3])
                    
                    with col_info:
                        st.markdown(f"## {agent.avatar_emoji}")
                        st.markdown(f"**{agent.display_name}**")
                        st.markdown(f'<span class="status-active">ACTIVE</span>', unsafe_allow_html=True)
                        st.caption(agent.description[:100] + "...")
                    
                    with col_metrics:
                        m1, m2, m3, m4, m5 = st.columns(5)
                        
                        with m1:
                            st.metric("Queries", f"{agent_metrics['total_queries']:,}")
                        
                        with m2:
                            st.metric("Avg Latency", f"{agent_metrics['avg_latency_sec']:.2f}s")
                        
                        with m3:
                            st.metric("Success Rate", f"{agent_metrics['success_rate']:.1f}%")
                        
                        with m4:
                            st.metric("Feedback", f"{agent_metrics['positive_feedback_rate']:.1f}%")
                        
                        with m5:
                            st.metric("Cost (7d)", f"${agent_metrics['total_cost']:.2f}")
                    
                    st.markdown("---")
            
            # Anomaly Alerts
            st.markdown("### 🚨 Anomaly Detection")
            anomalies = metrics_engine.detect_anomalies()
            
            if anomalies:
                for anomaly in anomalies:
                    severity_emoji = "⚠️" if anomaly['severity'] == "warning" else "🔴"
                    st.warning(f"{severity_emoji} **{anomaly['type'].replace('_', ' ').title()}**: {anomaly['message']}")
            else:
                st.success("✅ No anomalies detected. All systems operating normally.")
    
    elif page == "🤖 Agent Fleet":
        st.markdown('<div class="main-header">🤖 Agent Fleet Management</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Detailed performance analytics for each AI agent</div>', unsafe_allow_html=True)
        
        # Agent selector
        active_agents = registry.get_active_agents()
        agent_options = {agent.display_name: agent.agent_id for agent in active_agents}
        
        selected_agent_name = st.selectbox(
            "Select Agent for Deep Dive",
            options=list(agent_options.keys())
        )
        
        selected_agent_id = agent_options[selected_agent_name]
        agent_config = registry.get_agent(selected_agent_id)
        
        if not log_df.empty:
            metrics_engine = MetricsEngine(log_df)
            agent_metrics = metrics_engine.get_agent_metrics(selected_agent_id)
            
            # Agent Header
            col_header1, col_header2 = st.columns([1, 4])
            
            with col_header1:
                st.markdown(f"<div style='font-size: 5rem; text-align: center;'>{agent_config.avatar_emoji}</div>", unsafe_allow_html=True)
            
            with col_header2:
                st.markdown(f"## {agent_config.display_name}")
                st.markdown(f"**Type:** {agent_config.agent_type.title()} | **Department:** {agent_config.department}")
                st.markdown(f'<span class="status-active">ACTIVE</span>', unsafe_allow_html=True)
                st.caption(agent_config.description)
            
            st.markdown("---")
            
            # Key Metrics
            st.markdown("### 📊 Key Performance Indicators")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Queries", f"{agent_metrics['total_queries']:,}")
                st.metric("Sessions", f"{agent_metrics['total_sessions']:,}")
            
            with col2:
                st.metric("Avg Latency", f"{agent_metrics['avg_latency_sec']:.2f}s")
                sla_compliance = agent_metrics['sla_latency_compliance']
                st.caption(f"SLA Compliance: {sla_compliance:.1f}%")
            
            with col3:
                st.metric("Success Rate", f"{agent_metrics['success_rate']:.1f}%")
                st.caption(f"{agent_metrics['total_errors']} errors")
            
            with col4:
                st.metric("Positive Feedback", f"{agent_metrics['positive_feedback_rate']:.1f}%")
                st.caption(f"{agent_metrics['positive_feedback']}/{agent_metrics['total_feedback']} rated")
            
            with col5:
                st.metric("Total Cost (7d)", f"${agent_metrics['total_cost']:.2f}")
                st.caption(f"${agent_metrics['avg_cost_per_query']:.4f}/query")
            
            st.markdown("---")
            
            # Latency Deep Dive
            col_lat1, col_lat2 = st.columns([2, 1])
            
            with col_lat1:
                st.markdown("### ⏱️ Latency Distribution")
                
                agent_df = log_df[log_df['agent_id'] == selected_agent_id].copy()
                agent_df['agentLatency_sec'] = agent_df['agentLatency'] / 1000
                
                fig = px.histogram(
                    agent_df, 
                    x='agentLatency_sec',
                    nbins=50,
                    title='',
                    labels={'agentLatency_sec': 'Latency (seconds)', 'count': 'Frequency'},
                    color_discrete_sequence=['#3B82F6']
                )
                fig.add_vline(
                    x=agent_metrics['p95_latency_sec'], 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="P95"
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_lat2:
                st.markdown("### 📏 Percentiles")
                st.metric("P50 (Median)", f"{agent_metrics['p50_latency_sec']:.2f}s")
                st.metric("P90", f"{agent_metrics['p90_latency_sec']:.2f}s")
                st.metric("P95", f"{agent_metrics['p95_latency_sec']:.2f}s")
                st.metric("P99", f"{agent_metrics['p99_latency_sec']:.2f}s")
                
                st.markdown("---")
                st.caption(f"**SLA Target:** {agent_metrics['target_latency_sec']:.2f}s")
                st.caption(f"**Compliance:** {agent_metrics['sla_latency_compliance']:.1f}%")
            
            st.markdown("---")
            
            # Usage Patterns & Cost Trends
            col_usage, col_cost = st.columns(2)
            
            with col_usage:
                st.markdown("### 📅 Daily Query Volume")
                daily_metrics = metrics_engine.get_agent_daily_metrics(selected_agent_id)
                
                if not daily_metrics.empty:
                    daily_metrics['date_str'] = pd.to_datetime(daily_metrics['date']).dt.strftime('%b %d')
                    
                    fig = px.line(
                        daily_metrics,
                        x='date_str',
                        y='queries',
                        title='',
                        labels={'date_str': 'Date', 'queries': 'Queries'},
                        markers=True
                    )
                    fig.update_traces(line_color='#3B82F6', marker=dict(size=8))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_cost:
                st.markdown("### 💵 Daily Cost Trend")
                
                if not daily_metrics.empty:
                    fig = px.line(
                        daily_metrics,
                        x='date_str',
                        y='cost',
                        title='',
                        labels={'date_str': 'Date', 'cost': 'Cost ($)'},
                        markers=True
                    )
                    fig.update_traces(line_color='#10B981', marker=dict(size=8))
                    fig.update_layout(height=300)
                    
                    if agent_config.daily_cost_threshold:
                        fig.add_hline(
                            y=agent_config.daily_cost_threshold,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Daily Threshold"
                        )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Feedback Analysis
            st.markdown("### 💬 Feedback Analysis")
            
            agent_df = log_df[log_df['agent_id'] == selected_agent_id]
            negative_feedback_df = agent_df[
                (agent_df['feedbackStatus'] == 'negative') & 
                (agent_df['feedbackReason'] != '') &
                (agent_df['feedbackReason'].notna())
            ]
            
            if not negative_feedback_df.empty:
                reason_counts = negative_feedback_df['feedbackReason'].value_counts().reset_index()
                reason_counts.columns = ['Reason', 'Count']
                
                fig = px.bar(
                    reason_counts,
                    x='Count',
                    y='Reason',
                    orientation='h',
                    title='Top Negative Feedback Drivers',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("✅ No negative feedback recorded for this agent.")
            
            # Anomalies for this agent
            st.markdown("### 🚨 Anomaly Detection")
            agent_anomalies = metrics_engine.detect_anomalies(agent_id=selected_agent_id)
            
            if agent_anomalies:
                for anomaly in agent_anomalies:
                    severity_emoji = "⚠️" if anomaly['severity'] == "warning" else "🔴"
                    st.warning(f"{severity_emoji} **{anomaly['type'].replace('_', ' ').title()}**: {anomaly['message']}")
            else:
                st.success("✅ No anomalies detected for this agent.")
    
    elif page == "💰 Cost Analytics":
        st.markdown('<div class="main-header">💰 Cost Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive cost tracking with dynamic per-query model-based calculation</div>', unsafe_allow_html=True)
        
        if not log_df.empty:
            metrics_engine = MetricsEngine(log_df)
            fleet_metrics = metrics_engine.get_fleet_metrics()
            cost_breakdown = metrics_engine.get_cost_breakdown()
            
            # Top-level cost metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cost (7d)", f"${fleet_metrics['total_cost']:.2f}")
            
            with col2:
                st.metric("Daily Average", f"${fleet_metrics['daily_avg_cost']:.2f}")
                st.caption("Average per day")
            
            with col3:
                st.metric("Monthly Projection", f"${fleet_metrics['monthly_projection']:.2f}")
                st.caption("Based on current usage")
            
            with col4:
                st.metric("Cost per Query", f"${fleet_metrics['avg_cost_per_query']:.4f}")
                st.caption("Average across all agents")
            
            st.markdown("---")
            
            # Cost breakdown table
            st.markdown("### 📊 Cost Breakdown by Agent")
            st.caption("💡 **Note:** Costs calculated dynamically per query based on modelId - agents can use different models!")
            
            if not cost_breakdown.empty:
                display_df = cost_breakdown[['agent_name', 'queries', 'total_cost', 
                                            'avg_cost_per_query', 'cost_percentage', 
                                            'input_tokens', 'output_tokens']].copy()
                
                display_df.columns = ['Agent', 'Queries', 'Total Cost', 'Avg Cost/Query', 
                                     'Cost %', 'Input Tokens', 'Output Tokens']
                
                display_df['Total Cost'] = display_df['Total Cost'].apply(lambda x: f"${x:.2f}")
                display_df['Avg Cost/Query'] = display_df['Avg Cost/Query'].apply(lambda x: f"${x:.4f}")
                display_df['Cost %'] = display_df['Cost %'].apply(lambda x: f"{x:.1f}%")
                display_df['Input Tokens'] = display_df['Input Tokens'].apply(lambda x: f"{x:,.0f}")
                display_df['Output Tokens'] = display_df['Output Tokens'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # Cost trends and projections
            col_trend, col_proj = st.columns(2)
            
            with col_trend:
                st.markdown("### 📈 Daily Cost Trend")
                daily_metrics = metrics_engine.get_daily_metrics()
                
                if not daily_metrics.empty:
                    daily_metrics['date_str'] = pd.to_datetime(daily_metrics['date']).dt.strftime('%b %d')
                    
                    fig = px.area(
                        daily_metrics,
                        x='date_str',
                        y='cost',
                        title='',
                        labels={'date_str': 'Date', 'cost': 'Daily Cost ($)'},
                        color_discrete_sequence=['#10B981']
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_proj:
                st.markdown("### 🔮 Cost Projections")
                
                st.metric("7-Day Total", f"${fleet_metrics['total_cost']:.2f}")
                st.metric("30-Day Projection", f"${fleet_metrics['monthly_projection']:.2f}")
                st.metric("90-Day Projection", f"${fleet_metrics['monthly_projection'] * 3:.2f}")
                st.metric("Annual Projection", f"${fleet_metrics['monthly_projection'] * 12:.2f}")
                
                st.caption("Projections based on current usage patterns")
            
            st.markdown("---")
            
            # Token utilization
            st.markdown("### 🎯 Token Utilization")
            
            col_tok1, col_tok2 = st.columns(2)
            
            with col_tok1:
                st.metric("Total Input Tokens", f"{fleet_metrics['total_input_tokens']:,.0f}")
                st.metric("Total Output Tokens", f"{fleet_metrics['total_output_tokens']:,.0f}")
            
            with col_tok2:
                if not cost_breakdown.empty:
                    fig = go.Figure(data=[
                        go.Bar(name='Input Tokens', x=cost_breakdown['agent_name'], y=cost_breakdown['input_tokens']),
                        go.Bar(name='Output Tokens', x=cost_breakdown['agent_name'], y=cost_breakdown['output_tokens'])
                    ])
                    fig.update_layout(
                        barmode='group',
                        height=300,
                        title='Token Usage by Agent',
                        xaxis_title='Agent',
                        yaxis_title='Tokens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No cost data available.")
    
    elif page == "⚙️ Agent Management":
        st.markdown('<div class="main-header">⚙️ Agent Management</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Configure and manage AI agents in the enterprise</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📋 Agent Registry", "➕ Add New Agent"])
        
        with tab1:
            st.markdown("### Current Agents")
            
            for agent in registry.get_all_agents():
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.markdown(f"<div style='font-size: 3rem; text-align: center;'>{agent.avatar_emoji}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"### {agent.display_name}")
                        st.markdown(f"**ID:** `{agent.agent_id}`")
                        st.markdown(f"**Type:** {agent.agent_type.title()} | **Department:** {agent.department}")
                        st.markdown(f"**Model:** {agent.model_id}")
                        st.caption(agent.description)
                        st.caption(f"**Capabilities:** {', '.join(agent.capabilities)}")
                    
                    with col3:
                        status_class = "status-active" if agent.status == "active" else "status-maintenance"
                        st.markdown(f'<div style="text-align: center;"><span class="{status_class}">{agent.status.upper()}</span></div>', unsafe_allow_html=True)
                        
                        st.markdown("---")
                        
                        if agent.status == "active":
                            if st.button(f"🔧 Maintenance", key=f"maint_{agent.agent_id}"):
                                registry.update_agent_status(agent.agent_id, "maintenance")
                                st.success(f"Agent {agent.display_name} set to maintenance mode")
                                st.rerun()
                        else:
                            if st.button(f"✅ Activate", key=f"activate_{agent.agent_id}"):
                                registry.update_agent_status(agent.agent_id, "active")
                                st.success(f"Agent {agent.display_name} activated")
                                st.rerun()
                    
                    st.markdown("---")
        
        with tab2:
            st.markdown("### Add New Agent to Registry")
            st.info("🚧 Agent creation interface - Coming soon in production version")
