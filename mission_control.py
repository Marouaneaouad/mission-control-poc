"""
Enterprise AI Observability Platform - Mission Control (FIXED)
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
    .debug-info {
        background-color: #F3F4F6;
        border: 1px solid #D1D5DB;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        font-size: 0.85rem;
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

# --- FIXED: Data Fetching with better error handling and debugging ---
@st.cache_data(ttl=300)  # 5 minutes cache
def fetch_dynamodb_data(_dynamodb, table_name, days=None):
    """Fetch data from DynamoDB with improved error handling - ALL TIME if days=None"""
    debug_info = {
        'table_name': table_name,
        'days_requested': days if days else 'ALL TIME',
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
        
        # If days is None or very large, get ALL data without time filter
        if days is None or days >= 365:
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
            if days is None or days >= 365:
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
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill missing values
        df['feedbackStatus'] = df['feedbackStatus'].fillna('') if 'feedbackStatus' in df.columns else ''
        df['feedbackReason'] = df['feedbackReason'].fillna('') if 'feedbackReason' in df.columns else ''
        df['agentLatency'] = df['agentLatency'].fillna(0) if 'agentLatency' in df.columns else 0
        df['inputTokens'] = df['inputTokens'].fillna(0) if 'inputTokens' in df.columns else 0
        df['outputTokens'] = df['outputTokens'].fillna(0) if 'outputTokens' in df.columns else 0
        df['agentRationale'] = df['agentRationale'].fillna('') if 'agentRationale' in df.columns else ''
        
        debug_info['columns_found'] = list(df.columns)
        debug_info['final_row_count'] = len(df)
        
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
    
    # Fetch data with debug info - using selected time range
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
        3. **Data Scope:** Fetching ALL TIME data (no date filter applied)
        4. **Filter Type:** {debug_info.get('filter_type', 'Unknown')}
        
        **Quick Fixes:**
        - Check if data exists in DynamoDB console
        - Verify AWS credentials are correct
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
    
    # ... (rest of the pages remain the same)
    elif page == "🤖 Agent Fleet":
        st.info("Agent Fleet page - implementation continues as in original code...")
    
    elif page == "💰 Cost Analytics":
        st.info("Cost Analytics page - implementation continues as in original code...")
    
    elif page == "🔬 Session Explorer":
        st.info("Session Explorer page - implementation continues as in original code...")
    
    elif page == "⚙️ Agent Management":
        st.info("Agent Management page - implementation continues as in original code...")
