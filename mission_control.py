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
    .status-maintenance {
        background-color: #F59E0B;
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
        color: #000000;
    }
    .rationale-tool {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
        color: #000000;
    }
    .rationale-answer {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 4px;
        color: #000000;
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

# --- FIXED: Data Fetching with ALL TIME support ---
@st.cache_data(ttl=300)  # 5 minutes cache
def fetch_dynamodb_data(_dynamodb, table_name, days=None):
    """
    Fetch data from DynamoDB with optional time filtering
    days=None means fetch ALL TIME data
    """
    debug_info = {
        'table_name': table_name,
        'days_requested': 'ALL TIME' if days is None else days,
        'error': None,
        'items_found': 0,
        'scan_count': 0
    }
    
    try:
        if _dynamodb is None:
            debug_info['error'] = "DynamoDB resource is None"
            return pd.DataFrame(), debug_info
            
        table = _dynamodb.Table(table_name)
        
        # Test table connectivity
        try:
            table.load()
            debug_info['table_exists'] = True
        except Exception as e:
            debug_info['error'] = f"Table does not exist or cannot be accessed: {str(e)}"
            return pd.DataFrame(), debug_info
        
        # If days is None, get ALL data without time filter
        if days is None:
            debug_info['filter_type'] = 'NO TIME FILTER - ALL DATA'
            response = table.scan()
        else:
            # Apply time filter for specific days
            cutoff_time = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_time_str = cutoff_time.isoformat()
            debug_info['cutoff_time'] = cutoff_time_str
            debug_info['filter_type'] = f'FILTERED - Last {days} days'
            
            response = table.scan(
                FilterExpression=Attr('timestamp').gte(cutoff_time_str)
            )
        
        items = response.get('Items', [])
        debug_info['scan_count'] = 1
        debug_info['items_found'] = len(items)
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            if days is None:
                response = table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            else:
                response = table.scan(
                    FilterExpression=Attr('timestamp').gte(cutoff_time_str),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
            items.extend(response.get('Items', []))
            debug_info['scan_count'] += 1
            debug_info['items_found'] = len(items)
        
        if not items:
            debug_info['error'] = f"No items found in table"
            return pd.DataFrame(), debug_info
        
        df = pd.DataFrame(items)
        
        # Convert Decimal to float
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: float(x) if isinstance(x, Decimal) else x)
        
        # Ensure timestamp is datetime - let pandas infer format
        if 'timestamp' in df.columns:
            # Parse timestamps without strict format requirements
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Make timezone-aware (convert naive timestamps to UTC)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Fill missing values
        df['feedbackStatus'] = df['feedbackStatus'].fillna('') if 'feedbackStatus' in df.columns else ''
        df['feedbackReason'] = df['feedbackReason'].fillna('') if 'feedbackReason' in df.columns else ''
        df['agentLatency'] = df['agentLatency'].fillna(0) if 'agentLatency' in df.columns else 0
        df['inputTokens'] = df['inputTokens'].fillna(0) if 'inputTokens' in df.columns else 0
        df['outputTokens'] = df['outputTokens'].fillna(0) if 'outputTokens' in df.columns else 0
        df['agentRationale'] = df['agentRationale'].fillna('') if 'agentRationale' in df.columns else ''
        
        debug_info['columns_found'] = list(df.columns)
        debug_info['final_row_count'] = len(df)
        debug_info['date_range'] = f"{df['timestamp'].min()} to {df['timestamp'].max()}"
        
        return df, debug_info
        
    except Exception as e:
        debug_info['error'] = f"Exception during fetch: {str(e)}"
        import traceback
        debug_info['traceback'] = traceback.format_exc()
        return pd.DataFrame(), debug_info

# --- Initialize Registry ---
registry = get_registry()

# --- Main Application ---
if check_password():
    
    # Initialize DynamoDB
    dynamodb = get_dynamodb_resource(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)
    
    if dynamodb is None:
        st.error("❌ Failed to connect to DynamoDB. Please check your AWS credentials.")
        st.stop()
    
    # Sidebar Navigation
    st.sidebar.markdown("# 🎯 Mission Control")
    st.sidebar.markdown("---")
    
    # Time range selector
    st.sidebar.markdown("### 📅 Time Range")
    time_range = st.sidebar.selectbox(
        "Select data range",
        options=["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last 365 Days"],
        index=0  # Default to "All Time"
    )
    
    # Map selection to days parameter
    days_map = {
        "All Time": None,
        "Last 7 Days": 7,
        "Last 30 Days": 30,
        "Last 90 Days": 90,
        "Last 365 Days": 365
    }
    selected_days = days_map[time_range]
    
    st.sidebar.markdown("---")
    
    # Add debug toggle in sidebar
    show_debug = st.sidebar.checkbox("🐛 Show Debug Info", value=False)
    
    # Add manual refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Central Dashboard", "🤖 Agent Fleet", "💰 Cost Analytics", 
         "🔬 Session Explorer", "⚙️ Agent Management"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Stats")
    
    # Fetch data with selected time range
    loading_msg = f"Loading {time_range.lower()} data..." if selected_days else "Loading all-time data..."
    with st.spinner(loading_msg):
        log_df, debug_info = fetch_dynamodb_data(dynamodb, DYNAMODB_TABLE_NAME, days=selected_days)
    
    # Show debug information if enabled
    if show_debug:
        with st.sidebar.expander("📊 Debug Information"):
            st.json(debug_info)
    
    # Check if we have data
    if log_df.empty:
        st.sidebar.error("⚠️ No data loaded")
        if debug_info.get('error'):
            st.sidebar.caption(f"Error: {debug_info['error']}")
    else:
        metrics_engine = MetricsEngine(log_df)
        fleet_metrics = metrics_engine.get_fleet_metrics()
        
        st.sidebar.metric("Active Agents", fleet_metrics['total_agents'])
        st.sidebar.metric("Total Queries", f"{fleet_metrics['total_queries']:,}")
        st.sidebar.metric("Fleet Health", f"{fleet_metrics['health_score']:.0f}%")
        st.sidebar.metric("Est. Monthly Cost", f"${fleet_metrics['monthly_projection']:.2f}")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("🔄 Data refreshes every 5 minutes")
    st.sidebar.caption(f"📅 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # --- Show critical error message if no data ---
    if log_df.empty:
        st.error("### ❌ No Data Available")
        st.warning(f"""
        **Diagnosis:** {debug_info.get('error', 'Unknown error')}
        
        **Troubleshooting Steps:**
        1. **Check DynamoDB Table:** Verify table `{DYNAMODB_TABLE_NAME}` exists and has data
        2. **Check AWS Credentials:** Ensure your AWS credentials have DynamoDB read permissions
        3. **Data Scope:** {debug_info.get('filter_type', 'Unknown')}
        4. **Time Range Selected:** {time_range}
        
        **Quick Fixes:**
        - Check if data exists in DynamoDB console
        - Verify AWS credentials are correct
        - Try selecting "All Time" from the time range dropdown
        - Check table name is correct: `{DYNAMODB_TABLE_NAME}`
        """)
        
        if show_debug and debug_info.get('traceback'):
            with st.expander("🔍 Full Error Traceback"):
                st.code(debug_info['traceback'])
        
        st.stop()
    
    # --- Page Routing ---
    if page == "🏠 Central Dashboard":
        st.markdown('<div class="main-header">🎯 Mission Control - Central Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Enterprise-wide AI agent observability and performance monitoring</div>', unsafe_allow_html=True)
        
        # Show data freshness indicator
        col_fresh1, col_fresh2, col_fresh3 = st.columns([2, 1, 1])
        with col_fresh1:
            range_text = time_range.lower() if selected_days else "all-time"
            st.success(f"✅ Data loaded successfully - {len(log_df):,} total records ({range_text})")
        with col_fresh2:
            oldest_record = log_df['timestamp'].min()
            st.caption(f"Oldest: {oldest_record.strftime('%Y-%m-%d')}")
        with col_fresh3:
            newest_record = log_df['timestamp'].max()
            st.caption(f"Newest: {newest_record.strftime('%Y-%m-%d')}")
        
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
            st.metric("Total Queries", f"{fleet_metrics['total_queries']:,}")
            # Calculate average based on actual data timespan
            if len(log_df) > 0:
                days_span = (log_df['timestamp'].max() - log_df['timestamp'].min()).days + 1
                daily_avg = fleet_metrics['total_queries'] / max(days_span, 1)
                st.caption(f"~{daily_avg:.0f} queries/day avg")
        
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
        
        # Calculate actual timespan
        days_span = (log_df['timestamp'].max() - log_df['timestamp'].min()).days + 1
        
        col_cost1, col_cost2, col_cost3, col_cost4 = st.columns(4)
        
        with col_cost1:
            range_label = time_range if selected_days else "All Time"
            st.metric(f"Total Cost ({range_label})", f"${fleet_metrics['total_cost']:.2f}")
            st.caption(f"Over {days_span} days")
        
        with col_cost2:
            st.metric("Daily Average", f"${fleet_metrics['daily_avg_cost']:.2f}")
        
        with col_cost3:
            st.metric("Monthly Projection", f"${fleet_metrics['monthly_projection']:.2f}")
            st.caption("Based on daily average")
        
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
                        st.metric("Total Cost", f"${agent_metrics['total_cost']:.2f}")
                
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
                st.metric("Total Cost", f"${agent_metrics['total_cost']:.2f}")
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
    
    
    elif page == "🔬 Session Explorer":
        st.markdown('<div class="main-header">🔬 Session Explorer</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Deep dive into individual sessions and conversations</div>', unsafe_allow_html=True)
        
        if not log_df.empty:
            # Filter options
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            
            with col_filter1:
                agent_filter = st.multiselect(
                    "Filter by Agent",
                    options=log_df['agent_id'].unique(),
                    default=list(log_df['agent_id'].unique())
                )
            
            with col_filter2:
                feedback_filter = st.selectbox(
                    "Filter by Feedback",
                    options=["All", "Positive", "Negative", "No Feedback"],
                    index=0
                )
            
            with col_filter3:
                status_filter = st.selectbox(
                    "Filter by Status",
                    options=["All", "SUCCESS", "ERROR"],
                    index=0
                )
            
            # Apply filters
            filtered_df = log_df[log_df['agent_id'].isin(agent_filter)].copy()
            
            if feedback_filter == "Positive":
                filtered_df = filtered_df[filtered_df['feedbackStatus'] == 'positive']
            elif feedback_filter == "Negative":
                filtered_df = filtered_df[filtered_df['feedbackStatus'] == 'negative']
            elif feedback_filter == "No Feedback":
                filtered_df = filtered_df[filtered_df['feedbackStatus'] == '']
            
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            st.markdown(f"**Showing {len(filtered_df)} interactions across {filtered_df['sessionId'].nunique()} sessions**")
            
            st.markdown("---")
            
            # Session list
            if not filtered_df.empty:
                sessions = filtered_df.groupby('sessionId').agg(
                    agent_name=('agentName', 'first'),
                    latest_timestamp=('timestamp', 'max'),
                    message_count=('timestamp', 'count'),
                    errors=('status', lambda s: (s != 'SUCCESS').sum()),
                    avg_latency=('agentLatency', lambda x: x.mean() / 1000),
                    feedback=('feedbackStatus', lambda s: s.value_counts().to_dict())
                ).sort_values(by='latest_timestamp', ascending=False)
                
                for session_id, data in sessions.iterrows():
                    agent_emoji = "🤖"
                    for agent in registry.get_active_agents():
                        if agent.agent_name == data['agent_name']:
                            agent_emoji = agent.avatar_emoji
                            break
                    
                    summary = (
                        f"{agent_emoji} **{data['agent_name']}** | "
                        f"Session: `{session_id}` | "
                        f"Messages: {data['message_count']} | "
                        f"Errors: {data['errors']} | "
                        f"Avg Latency: {data['avg_latency']:.2f}s | "
                        f"Last Active: {data['latest_timestamp'].strftime('%Y-%m-%d %H:%M')}"
                    )
                    
                    with st.expander(summary):
                        session_df = filtered_df[filtered_df['sessionId'] == session_id].sort_values(by='timestamp', ascending=True)
                        
                        for idx, row in session_df.iterrows():
                            col_time, col_content = st.columns([1, 4])
                            
                            with col_time:
                                st.caption(row['timestamp'].strftime('%H:%M:%S'))
                                st.caption(f"{row['agentLatency']/1000:.2f}s")
                                if row['feedbackStatus']:
                                    emoji = "👍" if row['feedbackStatus'] == 'positive' else "👎"
                                    st.caption(f"{emoji} {row['feedbackStatus']}")
                            
                            with col_content:
                                st.markdown(f"**👤 User:** {row['userMessage']}")
                                st.markdown(f"**🤖 Agent:** {row['agentResponse']}")
                                
                                # CRITICAL: Show negative feedback reason
                                if row['feedbackStatus'] == 'negative' and row['feedbackReason'] and row['feedbackReason'] != '':
                                    st.markdown(f'<div class="feedback-negative">❌ Feedback Reason: {row["feedbackReason"]}</div>', unsafe_allow_html=True)
                                
                                # INNOVATIVE: Parse and display agent rationale
                                if row['agentRationale'] and row['agentRationale'] != '':
                                    with st.expander("🧠 View Agent Reasoning"):
                                        rationale_steps = parse_agent_rationale(row['agentRationale'])
                                        
                                        if rationale_steps:
                                            for step in rationale_steps:
                                                if step['type'] == 'thinking':
                                                    st.markdown(f'<div class="rationale-thinking">💭 **Thinking:** {step["content"]}</div>', unsafe_allow_html=True)
                                                elif step['type'] == 'tool_call':
                                                    st.markdown(f'<div class="rationale-tool">🔧 **{step["content"]}**</div>', unsafe_allow_html=True)
                                                elif step['type'] == 'tool_result':
                                                    st.markdown(f'<div class="rationale-tool">📊 **Result:** {step["content"]}</div>', unsafe_allow_html=True)
                                                elif step['type'] == 'answer':
                                                    st.markdown(f'<div class="rationale-answer">✅ **Answer:** {step["content"]}</div>', unsafe_allow_html=True)
                                        else:
                                            # Fallback: show raw rationale
                                            st.text(row['agentRationale'][:500])
                                
                                if row['status'] != 'SUCCESS':
                                    st.error(f"❌ Error: {row['status']}")
                            
                            st.markdown("---")
            else:
                st.info("No sessions match the selected filters.")
        else:
            st.warning("⚠️ No session data available.")
            
    elif page == "💰 Cost Analytics":
        st.markdown('<div class="main-header">💰 Cost Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Comprehensive cost tracking with dynamic per-query model-based calculation</div>', unsafe_allow_html=True)
        
        if not log_df.empty:
            metrics_engine = MetricsEngine(log_df)
            fleet_metrics = metrics_engine.get_fleet_metrics()
            cost_breakdown = metrics_engine.get_cost_breakdown()
            
            # Calculate actual timespan
            days_span = (log_df['timestamp'].max() - log_df['timestamp'].min()).days + 1
            
            # Top-level cost metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                range_label = time_range if selected_days else "All Time"
                st.metric(f"Total Cost ({range_label})", f"${fleet_metrics['total_cost']:.2f}")
                st.caption(f"Over {days_span} days")
            
            with col2:
                st.metric("Daily Average", f"${fleet_metrics['daily_avg_cost']:.2f}")
                st.caption("Average per day")
            
            with col3:
                st.metric("Monthly Projection", f"${fleet_metrics['monthly_projection']:.2f}")
                st.caption("Based on daily average")
            
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
                
                st.metric("Current Period", f"${fleet_metrics['total_cost']:.2f}")
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
