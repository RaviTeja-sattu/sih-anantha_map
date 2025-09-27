import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from geopy.distance import geodesic
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
import json

# Import custom modules
from database_manager import DatabaseManager
from ai_manager import AIManager

# Page configuration
st.set_page_config(
    page_title="Argo Float Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern UI styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a202c 100%);
        color: #e2e8f0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.4);
    }
    
    .float-panel {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #475569;
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    .float-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
        transition: all 0.3s ease;
    }
    
    .float-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
    }
    
    .selected-float {
        border-left-color: #10b981;
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
    }
    
    .chat-container {
        background: #1a202c;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #2d3748;
        min-height: 600px;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message {
        background: #2d3748;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #60a5fa;
    }
    
    .user-message {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border-left: 4px solid #93c5fd;
        margin-left: 2rem;
    }
    
    .ai-message {
        background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
        border-left: 4px solid #10b981;
        margin-right: 2rem;
    }
    
    .tab-container {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #334155;
    }
    
    .mode-button {
        background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .mode-button:hover {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        transform: translateY(-1px);
    }
    
    .active-mode {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    
    .viz-controls {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #374151;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stSelectbox > div > div {
        background-color: #374151;
        color: #e2e8f0;
        border: 1px solid #4b5563;
    }
    
    .stTextArea > div > div > textarea {
        background-color: #374151;
        color: #e2e8f0;
        border: 1px solid #4b5563;
    }
</style>
""", unsafe_allow_html=True)

# Language options
LANGUAGES = {
    'English': 'en', 'Spanish': 'es', 'French': 'fr', 'German': 'de', 
    'Italian': 'it', 'Portuguese': 'pt', 'Russian': 'ru', 'Japanese': 'ja',
    'Chinese': 'zh', 'Hindi': 'hi', 'Arabic': 'ar', 'Telugu': 'te'
}

# AI modes configuration
AI_MODES = {
    'Think Deeper': {'icon': '🧠', 'description': 'Comprehensive analysis with detailed reasoning'},
    'Quick Answer': {'icon': '⚡', 'description': 'Fast, concise responses'},
    'Research Mode': {'icon': '🔬', 'description': 'In-depth scientific research and citations'},
    'Creative': {'icon': '🎨', 'description': 'Creative and exploratory analysis'}
}

def initialize_session_state():
    """Initialize session state variables"""
    defaults = {
        'page': 'map',
        'selected_floats': [],
        'chat_history': [],
        'current_language': 'English',
        'current_ai_mode': 'Quick Answer',
        'active_tab': 'chat',
        'gemini_model': None,
        'last_query_data': None,
        'viz_data': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Initialize Gemini
    if st.session_state.gemini_model is None:
        st.session_state.gemini_model = initialize_gemini()

def initialize_gemini():
    """Initialize Gemini AI model"""
    try:
        api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCF1u-y3tCy2dKRqW0D4bWVKaDzOwdXmac')
        genai.configure(api_key=api_key)
        
        models = ["gemini-2.0-flash-exp", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        for model_name in models:
            try:
                model = genai.GenerativeModel(model_name)
                # Test the model
                response = model.generate_content("Test", generation_config=genai.types.GenerationConfig(max_output_tokens=5))
                if response and response.text:
                    return model
            except:
                continue
        return None
    except:
        return None

class MapManager:
    """Map management and float discovery"""
    
    @staticmethod
    def create_map(df: pd.DataFrame):
        """Create Indian Ocean centered map"""
        # Indian Ocean center
        m = folium.Map(
            location=[0, 75], # Indian Ocean coordinates
            zoom_start=4,
            tiles=None,
            prefer_canvas=True
        )
        
        # Add modern tile layers
        folium.TileLayer('CartoDB dark_matter', name='Dark Ocean').add_to(m)
        folium.TileLayer('CartoDB positron', name='Light').add_to(m)
        
        # Color scheme for regions
        region_colors = {
            'Indian Ocean': '#ef4444',
            'North Pacific': '#3b82f6',
            'South Pacific': '#10b981',
            'North Atlantic': '#f59e0b',
            'South Atlantic': '#8b5cf6',
            'Southern Ocean': '#06b6d4',
            'Arctic Ocean': '#ec4899',
            'Mediterranean Sea': '#f97316'
        }
        
        # Add float markers
        for _, row in df.iterrows():
            lat, lon = row.get('display_lat'), row.get('display_lon')
            
            if pd.notna(lat) and pd.notna(lon):
                region = str(row.get('dominant_region', 'Unknown'))
                color = region_colors.get(region, '#64748b')
                
                # Marker size based on data
                profiles = row.get('num_profiles', 0)
                radius = max(4, min(10, profiles / 50))
                
                popup_html = f"""
                <div style="font-family: 'Inter', sans-serif; width: 280px; 
                           background: #1e293b; color: white; padding: 12px; 
                           border-radius: 8px; border: 1px solid #334155;">
                    <h4 style="margin: 0 0 8px 0; color: #60a5fa;">Float {row['float_id']}</h4>
                    <p style="margin: 0; font-size: 13px;">
                        <strong>Region:</strong> {region}<br>
                        <strong>Profiles:</strong> {profiles:,}<br>
                        <strong>Status:</strong> {row.get('end_mission_status', 'Unknown')}<br>
                        <strong>Project:</strong> {str(row.get('project_name', 'N/A'))[:30]}...
                    </p>
                </div>
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"Float {row['float_id']} ({region})",
                    color='white',
                    fillColor=color,
                    fillOpacity=0.8,
                    weight=1.5
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
    
    @staticmethod
    def find_nearest_floats(click_lat: float, click_lon: float, df: pd.DataFrame, n: int = 10):
        """Find nearest floats to clicked location"""
        if df.empty:
            return pd.DataFrame()
        
        distances = []
        for idx, row in df.iterrows():
            lat, lon = row.get('display_lat'), row.get('display_lon')
            
            if pd.notna(lat) and pd.notna(lon):
                try:
                    distance = geodesic((click_lat, click_lon), (lat, lon)).kilometers
                    distances.append((distance, row))
                except:
                    continue
        
        if not distances:
            return pd.DataFrame()
        
        # Sort by distance and return top N
        distances.sort(key=lambda x: x[0])
        nearest_data = [row.to_dict() for dist, row in distances[:n]]
        for i, (dist, _) in enumerate(distances[:n]):
            nearest_data[i]['distance_km'] = dist
        
        return pd.DataFrame(nearest_data)

def render_map_page():
    """Render the main map interface"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Argo Float Explorer</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading global float network..."):
        df = DatabaseManager.load_float_metadata()
    
    if df.empty:
        st.error("No float data available. Check database connection.")
        return
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Floats", f"{len(df):,}")
    with col2:
        total_profiles = df['num_profiles'].sum()
        st.metric("Total Profiles", f"{total_profiles:,}")
    with col3:
        active = len(df[df['end_mission_status'].isin(['Active', 'Operational'])])
        st.metric("Active Floats", f"{active:,}")
    with col4:
        regions = df['dominant_region'].nunique()
        st.metric("Ocean Regions", regions)
    
    # Full screen map
    st.markdown("### 🗺️ Interactive Global Map")
    st.info("Click anywhere on the map to discover nearby floats")
    
    world_map = MapManager.create_map(df)
    map_data = st_folium(world_map, width=1200, height=700, returned_objects=["last_clicked"])
    
    # Handle map clicks
    if map_data.get('last_clicked'):
        click_lat = map_data['last_clicked']['lat']
        click_lon = map_data['last_clicked']['lng']
        
        st.success(f"📍 Selected location: {click_lat:.3f}°, {click_lon:.3f}°")
        
        # Find nearest floats
        with st.spinner("Finding nearest floats..."):
            nearest_floats = MapManager.find_nearest_floats(click_lat, click_lon, df, 10)
        
        if not nearest_floats.empty:
            st.markdown("""
            <div class="float-panel">
                <h3 style="color: #60a5fa; margin-bottom: 1rem;">🎯 Nearest Floats (Select for Analysis)</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Display floats in rows of 2
            for i in range(0, len(nearest_floats), 2):
                col1, col2 = st.columns(2)
                
                for j, col in enumerate([col1, col2]):
                    if i + j < len(nearest_floats):
                        float_data = nearest_floats.iloc[i + j]
                        float_id = int(float_data['float_id'])
                        
                        with col:
                            is_selected = float_id in st.session_state.selected_floats
                            card_class = "float-card selected-float" if is_selected else "float-card"
                            
                            st.markdown(f"""
                            <div class="{card_class}">
                                <h4>🌊 Float {float_id}</h4>
                                <p>📍 Distance: {float_data['distance_km']:.1f} km</p>
                                <p>🌍 Region: {float_data.get('dominant_region', 'Unknown')}</p>
                                <p>📊 Profiles: {float_data.get('num_profiles', 0):,}</p>
                                <p>📅 Status: {float_data.get('end_mission_status', 'Unknown')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Selection controls
                            button_col1, button_col2 = st.columns(2)
                            
                            with button_col1:
                                if is_selected:
                                    if st.button("❌ Remove", key=f"remove_{float_id}", type="secondary"):
                                        st.session_state.selected_floats.remove(float_id)
                                        st.rerun()
                                else:
                                    if st.button("✅ Select", key=f"select_{float_id}", type="primary"):
                                        st.session_state.selected_floats.append(float_id)
                                        st.rerun()
            
            # Analysis button
            if st.session_state.selected_floats:
                st.markdown("---")
                col_center = st.columns([1, 2, 1])[1]
                
                with col_center:
                    st.markdown(f"""
                    <div style="text-align: center; margin: 2rem 0;">
                        <p style="color: #10b981; font-size: 1.1rem; margin-bottom: 1rem;">
                            ✅ {len(st.session_state.selected_floats)} floats selected
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button("🚀 ANALYZE FLOATS", key="analyze_button", type="primary", use_container_width=True):
                        st.session_state.page = 'analysis'
                        st.rerun()
                    
                    if st.button("🗑️ Clear Selection", type="secondary", use_container_width=True):
                        st.session_state.selected_floats = []
                        st.rerun()
        else:
            st.warning("No floats found in this area. Try clicking in ocean regions.")

def render_analysis_page():
    """Render the analysis dashboard"""
    if not st.session_state.selected_floats:
        st.warning("No floats selected for analysis")
        if st.button("🗺️ Back to Map", type="primary"):
            st.session_state.page = 'map'
            st.rerun()
        return
    
    # Header with controls
    st.markdown("""
    <div class="main-header">
        <h1>📊 Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation and controls - simplified
    nav_col1, nav_col2 = st.columns([3, 1])
    
    with nav_col1:
        st.success(f"🎯 Analyzing {len(st.session_state.selected_floats)} floats: {', '.join(map(str, st.session_state.selected_floats[:3]))}{' ...' if len(st.session_state.selected_floats) > 3 else ''}")
    
    with nav_col2:
        if st.button("🗺️ Back to Map", type="secondary"):
            st.session_state.page = 'map'
            st.rerun()
    
    # Tab interface
    st.markdown("---")
    
    tab_col1, tab_col2, tab_col3, tab_col4 = st.columns(4)
    
    tabs = [
        ("💬 Chat", "chat"),
        ("📊 Visualization", "visualization"), 
        ("📋 Tables", "tables"),
        ("🌐 Language", "language")
    ]
    
    for i, (tab_name, tab_key) in enumerate(tabs):
        with [tab_col1, tab_col2, tab_col3, tab_col4][i]:
            button_type = "primary" if st.session_state.active_tab == tab_key else "secondary"
            if st.button(tab_name, key=f"tab_{tab_key}", type=button_type, use_container_width=True):
                st.session_state.active_tab = tab_key
                st.rerun()
    
    st.markdown("---")
    
    # Render active tab
    if st.session_state.active_tab == 'chat':
        render_chat_tab()
    elif st.session_state.active_tab == 'visualization':
        render_visualization_tab()
    elif st.session_state.active_tab == 'tables':
        render_tables_tab()
    elif st.session_state.active_tab == 'language':
        render_language_tab()

def render_chat_tab():
    """Compact ChatGPT-style interface"""
    st.markdown("### 💬 AI Oceanography Assistant")
    
    # Compact AI mode selection in header
    st.markdown("**Response Mode:**")
    mode_cols = st.columns(4)
    
    for i, (mode, config) in enumerate(AI_MODES.items()):
        with mode_cols[i]:
            button_type = "primary" if st.session_state.current_ai_mode == mode else "secondary"
            if st.button(f"{config['icon']}", key=f"mode_{i}", type=button_type, use_container_width=True, help=f"{mode}: {config['description']}"):
                st.session_state.current_ai_mode = mode
                st.rerun()
    
    st.caption(f"Active: {st.session_state.current_ai_mode}")
    st.markdown("---")
    
    # Chat history with compact styling
    if st.session_state.chat_history:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); 
                           color: white; padding: 12px; border-radius: 8px; margin: 8px 0; 
                           margin-left: 50px; border-left: 3px solid #60a5fa;">
                    <strong>You:</strong> {message["content"]}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #374151 0%, #4b5563 100%); 
                           color: #e2e8f0; padding: 12px; border-radius: 8px; margin: 8px 0; 
                           margin-right: 50px; border-left: 3px solid #10b981;">
                    <strong>AI ({message.get("mode", "AI")}):</strong><br>{message["content"]}
                </div>
                ''', unsafe_allow_html=True)
    
    # Compact suggested questions for first-time users
    if not st.session_state.chat_history:
        st.markdown("**💡 Quick Start:**")
        suggestions = [
            "Tell me about my selected floats",
            "How to interpret ocean temperature data?",
            "What do salinity measurements reveal?",
            "Explain depth profiles"
        ]
        
        suggestion_cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i % 2]:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    # Add user message
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': suggestion,
                        'timestamp': datetime.now()
                    })
                    
                    # Generate AI response
                    with st.spinner(f"{st.session_state.current_ai_mode} mode..."):
                        context = f"Selected floats: {st.session_state.selected_floats}"
                        response = AIManager.generate_response(
                            suggestion, context, st.session_state.current_ai_mode, st.session_state.current_language
                        )
                        
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response,
                            'mode': st.session_state.current_ai_mode,
                            'timestamp': datetime.now()
                        })
                    st.rerun()
    
    # Compact message input at bottom
    st.markdown("---")
    
    # Input box with send button inline
    input_col, button_col1, button_col2 = st.columns([6, 1, 1])
    
    with input_col:
        user_input = st.text_area(
            "Message:",
            placeholder="Ask about oceanography, floats, or data analysis...",
            height=80,
            key="chat_input",
            label_visibility="collapsed"
        )
    
    with button_col1:
        st.write("")  # spacing
        if st.button("📤", type="primary", use_container_width=True, help="Send Message") and user_input.strip():
            # Add user message
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Generate AI response
            with st.spinner("Thinking..."):
                context = f"Selected floats: {st.session_state.selected_floats}"
                response = AIManager.generate_response(
                    user_input, context, st.session_state.current_ai_mode, st.session_state.current_language
                )
                
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'mode': st.session_state.current_ai_mode,
                    'timestamp': datetime.now()
                })
            
            st.rerun()
    
    with button_col2:
        st.write("")  # spacing
        if st.button("🗑️", type="secondary", use_container_width=True, help="Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

def parse_query_fallback(query, available_columns):
    """Fallback parsing for visualization requests when AI is unavailable"""
    query_lower = query.lower()
    
    # Determine chart type from query
    if 'scatter' in query_lower:
        chart_type = "Scatter Plot"
    elif 'line' in query_lower or 'profile' in query_lower:
        chart_type = "Line Chart"
    elif 'histogram' in query_lower or 'distribution' in query_lower:
        chart_type = "Histogram"
    elif 'box' in query_lower:
        chart_type = "Box Plot"
    elif '3d' in query_lower:
        chart_type = "3D Scatter"
    else:
        chart_type = "Scatter Plot"
    
    # Smart variable detection based on available columns
    x_var, y_var = None, None
    
    # Check for specific column mentions
    for col in available_columns:
        if col.lower() in query_lower:
            if x_var is None:
                x_var = col
            elif y_var is None and col != x_var:
                y_var = col
    
    # Apply oceanographic logic for common patterns
    if 'latitude' in query_lower and 'longitude' in query_lower:
        x_var, y_var = "longitude", "latitude"
    elif 'temperature' in query_lower and 'salinity' in query_lower:
        if query_lower.find('temperature') < query_lower.find('salinity'):
            x_var, y_var = "temperature", "salinity"
        else:
            x_var, y_var = "salinity", "temperature"
    elif 'salinity' in query_lower and 'depth' in query_lower:
        x_var, y_var = "depth", "salinity"
    elif 'temperature' in query_lower and 'depth' in query_lower:
        x_var, y_var = "depth", "temperature"
    elif 'date' in query_lower or 'time' in query_lower:
        x_var = "date"
        if 'temperature' in query_lower:
            y_var = "temperature"
        elif 'salinity' in query_lower:
            y_var = "salinity"
        else:
            y_var = "temperature"
    
    # Defaults if nothing found
    if x_var is None:
        x_var = "depth" if "depth" in available_columns else available_columns[0]
    if y_var is None:
        y_var = "temperature" if "temperature" in available_columns else available_columns[1] if len(available_columns) > 1 else available_columns[0]
    
    # For histograms, both should be the same
    if chart_type == "Histogram":
        if 'temperature' in query_lower:
            x_var = y_var = "temperature"
        elif 'salinity' in query_lower:
            x_var = y_var = "salinity"
        elif 'depth' in query_lower:
            x_var = y_var = "depth"
    
    # Color variable
    color_var = "float_id" if chart_type != "Histogram" and "float_id" in available_columns else None
    
    return chart_type, x_var, y_var, color_var

def render_visualization_tab():
    """Clean visualization interface focused on charts"""
    st.markdown("### 📊 Data Visualization")
    
    # Load basic data without showing it
    with st.spinner("Loading profile data for selected floats..."):
        basic_query = f"""
        SELECT 
            ad.float_id,
            ad.profile,
            ad.date,
            ad.latitude,
            ad.longitude,
            ad.pres_adj_dbar as depth,
            ad.temp_adj_c as temperature,
            ad.psal_adj_psu as salinity
        FROM argo_data_clean ad
        WHERE ad.float_id IN ({', '.join(map(str, st.session_state.selected_floats))})
          AND ad.temp_adj_qc = '1' 
          AND ad.psal_adj_qc = '1'
          AND ad.pres_adj_dbar IS NOT NULL
          AND ad.temp_adj_c IS NOT NULL
          AND ad.psal_adj_psu IS NOT NULL
        ORDER BY ad.float_id, ad.profile, ad.pres_adj_dbar
        LIMIT 2000
        """
        
        default_data, error = DatabaseManager.execute_query(basic_query)
    
    if error:
        st.error(f"Error loading data: {error}")
        return
    
    if default_data is None or default_data.empty:
        st.warning("No profile data available for selected floats")
        return
    
    # Store data for visualization
    st.session_state.viz_data = default_data
    
    st.success(f"✅ Loaded {len(default_data)} profile measurements from {len(st.session_state.selected_floats)} floats")
    
    # Centered visualization controls
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown('<div class="viz-controls">', unsafe_allow_html=True)
        st.markdown("#### 🎛️ Chart Configuration")
        
        # Get all available columns
        numeric_columns = default_data.select_dtypes(include=[np.number]).columns.tolist()
        all_columns = default_data.columns.tolist()
        
        # Chart type buttons side by side
        st.markdown("**Chart Type:**")
        chart_types = ['Line', 'Scatter', 'Box', 'Histogram', '3D']
        chart_cols = st.columns(5)
        
        selected_chart = st.session_state.get('selected_chart_type', 'Line')
        
        for i, chart_name in enumerate(chart_types):
            with chart_cols[i]:
                button_type = "primary" if selected_chart == chart_name else "secondary"
                if st.button(chart_name, key=f"chart_{i}", type=button_type, use_container_width=True):
                    st.session_state.selected_chart_type = chart_name
                    st.rerun()
        
        # Map to full names for processing
        chart_type_map = {
            'Line': 'Line Chart',
            'Scatter': 'Scatter Plot', 
            'Box': 'Box Plot',
            'Histogram': 'Histogram',
            '3D': '3D Scatter'
        }
        chart_type = chart_type_map[selected_chart]
        
        col1, col2 = st.columns(2)
        with col1:
            x_column = st.selectbox("X-axis:", all_columns, index=all_columns.index('depth') if 'depth' in all_columns else 0)
        with col2:
            y_column = st.selectbox("Y-axis:", numeric_columns, index=numeric_columns.index('temperature') if 'temperature' in numeric_columns else 0)
        
        color_options = ['None'] + [col for col in all_columns if col not in [x_column, y_column]]
        color_by = st.selectbox("Color by:", color_options)
        color_by = None if color_by == 'None' else color_by
        
        if st.button("🎨 Generate Visualization", type="primary", use_container_width=True):
            viz_result = create_visualization(default_data, chart_type, x_column, y_column, color_by)
            
            # Only show download option if visualization was successful
            if viz_result:
                st.markdown("#### 💾 Export Data")
                csv_data = default_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Chart Data (CSV)",
                    csv_data,
                    f"argo_chart_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI-powered visualization suggestions  
    st.markdown("---")
    st.markdown("#### 🤖 Quick Visualizations")
    
    quick_viz = [
        ("Temperature vs Depth", "depth", "temperature", "float_id", "Line Chart"),
        ("Salinity vs Temperature", "temperature", "salinity", "float_id", "Scatter Plot"), 
        ("Temperature Distribution", "temperature", "temperature", None, "Histogram"),
        ("Salinity vs Depth", "depth", "salinity", "float_id", "Line Chart")
    ]
    
    viz_cols = st.columns(2)
    for i, (name, x_col, y_col, color_col, chart_type) in enumerate(quick_viz):
        with viz_cols[i % 2]:
            if st.button(f"📊 {name}", key=f"quick_viz_{i}", use_container_width=True):
                create_visualization(default_data, chart_type, x_col, y_col, color_col)
    
    # Custom query section
    st.markdown("---")
    custom_query = st.text_area(
        "🔍 Custom visualization request:",
        placeholder="e.g., 'Show temperature vs salinity scatter plot' or 'Temperature histogram for deep waters'",
        height=80
    )
    
    if st.button("🚀 Generate Custom Visualization", type="primary", use_container_width=True) and custom_query.strip():
        with st.spinner("Creating custom visualization..."):
            # Use Gemini AI for intelligent parsing
            if st.session_state.gemini_model:
                try:
                    available_columns = list(default_data.columns)
                    parsing_prompt = f"""
                    Parse this visualization request and return a JSON response:
                    
                    USER REQUEST: "{custom_query}"
                    AVAILABLE COLUMNS: {available_columns}
                    
                    Return JSON with:
                    {{
                        "chart_type": "Line Chart" | "Scatter Plot" | "Box Plot" | "Histogram" | "3D Scatter",
                        "x_column": "column_name",
                        "y_column": "column_name", 
                        "color_column": "column_name" or null
                    }}
                    
                    Guidelines:
                    - Match column names exactly from available columns
                    - For histograms, x_column and y_column should be the same
                    - Choose appropriate chart types based on the request
                    - Use "float_id" for color when showing multiple floats
                    - Consider latitude/longitude for geographic plots
                    - Default to meaningful oceanographic combinations
                    """
                    
                    response = st.session_state.gemini_model.generate_content(
                        parsing_prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=200,
                        )
                    )
                    
                    if response and response.text:
                        import json
                        # Extract JSON from response
                        json_text = response.text.strip()
                        if '```json' in json_text:
                            json_text = json_text.split('```json')[1].split('```')[0]
                        elif '```' in json_text:
                            json_text = json_text.split('```')[1].split('```')[0]
                        
                        try:
                            parsed = json.loads(json_text)
                            auto_chart_type = parsed.get("chart_type", "Scatter Plot")
                            x_var = parsed.get("x_column", "depth")
                            y_var = parsed.get("y_column", "temperature")
                            color_var = parsed.get("color_column")
                            
                            # Validate columns exist
                            if x_var not in available_columns:
                                x_var = "depth"
                            if y_var not in available_columns:
                                y_var = "temperature"
                            if color_var and color_var not in available_columns:
                                color_var = "float_id"
                                
                        except json.JSONDecodeError:
                            # Fallback to basic parsing
                            auto_chart_type, x_var, y_var, color_var = parse_query_fallback(custom_query, available_columns)
                    else:
                        # Fallback to basic parsing
                        auto_chart_type, x_var, y_var, color_var = parse_query_fallback(custom_query, available_columns)
                        
                except Exception:
                    # Fallback to basic parsing
                    auto_chart_type, x_var, y_var, color_var = parse_query_fallback(custom_query, available_columns)
            else:
                # Fallback to basic parsing if no AI
                auto_chart_type, x_var, y_var, color_var = parse_query_fallback(custom_query, available_columns)
            
            # Generate the visualization
            success = create_visualization(default_data, auto_chart_type, x_var, y_var, color_var)
            
            if success:
                st.success(f"✅ Generated {auto_chart_type}: {y_var.title()} vs {x_var.title()}")
                
                # Offer download for custom visualization
                csv_data = default_data.to_csv(index=False)
                st.download_button(
                    "📥 Download Visualization Data",
                    csv_data,
                    f"argo_custom_viz_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv"
                )

def create_visualization(df, chart_type, x_col, y_col, color_col):
    """Create and display visualization with improved error handling and hover effects"""
    try:
        # Dark theme colors
        colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4']
        
        # Common layout with hover effects
        layout = {
            'template': 'plotly_dark',
            'height': 600,
            'paper_bgcolor': '#1a202c',
            'plot_bgcolor': '#2d3748',
            'font': {'color': '#e2e8f0'},
            'title': {'font': {'color': '#60a5fa', 'size': 18}},
            'hoverlabel': {'bgcolor': '#374151', 'font_color': '#e2e8f0'}
        }
        
        if chart_type == 'Line Chart':
            fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                         title=f"{y_col.title()} vs {x_col.title()}")
            fig.update_traces(
                hovertemplate=f'<b>{x_col.title()}</b>: %{{x}}<br><b>{y_col.title()}</b>: %{{y}}<extra></extra>',
                line=dict(width=3)
            )
            
        elif chart_type == 'Scatter Plot':
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                           title=f"{y_col.title()} vs {x_col.title()}")
            fig.update_traces(
                hovertemplate=f'<b>{x_col.title()}</b>: %{{x}}<br><b>{y_col.title()}</b>: %{{y}}<extra></extra>',
                marker=dict(size=8, line=dict(width=1, color='white'))
            )
            
        elif chart_type == 'Box Plot':
            fig = px.box(df, x=color_col, y=y_col,
                        title=f"{y_col.title()} Distribution")
            fig.update_traces(
                hovertemplate=f'<b>{y_col.title()}</b>: %{{y}}<extra></extra>',
                marker=dict(size=6)
            )
            
        elif chart_type == 'Histogram':
            fig = px.histogram(df, x=y_col, nbins=30,
                             title=f"{y_col.title()} Distribution")
            fig.update_traces(
                hovertemplate=f'<b>{y_col.title()}</b>: %{{x}}<br><b>Count</b>: %{{y}}<extra></extra>',
                marker=dict(line=dict(width=1, color='white'))
            )
            
        elif chart_type == '3D Scatter':
            if len(df.select_dtypes(include=[np.number]).columns) >= 3:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                z_col = [col for col in numeric_cols if col not in [x_col, y_col]][0]
                
                fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                                  title=f"3D: {x_col.title()} vs {y_col.title()} vs {z_col.title()}")
                fig.update_traces(
                    marker=dict(size=5, line=dict(width=1, color='white'))
                )
            else:
                st.error("Need at least 3 numeric columns for 3D visualization")
                return False
        
        # Add hover effects and interactivity
        fig.update_layout(
            **layout,
            hovermode='closest',
            transition_duration=500
        )
        
        # Enhanced hover and selection effects
        fig.update_traces(
            hoverlabel=dict(
                bgcolor="#374151",
                bordercolor="#60a5fa",
                font_size=12,
                font_family="Inter"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        return True
        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return False

def render_tables_tab():
    """Data tables interface"""
    st.markdown("### 📋 Data Tables & Queries")
    
    # Quick query buttons
    quick_queries = [
        ("📊 Float Summary", "Show summary statistics for each selected float including region and profile count"),
        ("🌡️ Temperature Data", "Get temperature measurements with depth for all selected floats"),
        ("🧂 Salinity Profiles", "Show salinity data with depth and quality information"),
        ("📈 Recent Measurements", "Get the most recent measurements from each float")
    ]
    
    st.markdown("#### 🚀 Quick Queries")
    
    query_cols = st.columns(2)
    for i, (button_text, query_desc) in enumerate(quick_queries):
        with query_cols[i % 2]:
            if st.button(button_text, key=f"quick_query_{i}", use_container_width=True):
                with st.spinner("Executing query..."):
                    sql = AIManager.generate_sql_query(
                        query_desc, st.session_state.selected_floats, st.session_state.current_language
                    )
                    
                    if not sql.startswith("Error"):
                        result_df, error = DatabaseManager.execute_query(sql)
                        
                        if not error and result_df is not None and not result_df.empty:
                            st.success(f"✅ Query returned {len(result_df)} rows")
                            st.dataframe(result_df, use_container_width=True, height=400)
                            st.session_state.last_query_data = result_df
                        else:
                            st.error(f"Query failed: {error}")
                    else:
                        st.error(sql)
    
    # Custom query interface
    st.markdown("---")
    st.markdown("#### 💬 Custom Query")
    
    user_query = st.text_area(
        f"Describe the data you need in {st.session_state.current_language}:",
        placeholder="e.g., 'Show all measurements deeper than 500m' or 'Get temperature data from last 6 months'",
        height=100
    )
    
    if st.button("📊 Execute Query", type="primary", use_container_width=True) and user_query.strip():
        with st.spinner("Processing your query..."):
            sql = AIManager.generate_sql_query(
                user_query, st.session_state.selected_floats, st.session_state.current_language
            )
            
            if not sql.startswith("Error"):
                result_df, error = DatabaseManager.execute_query(sql)
                
                if not error and result_df is not None and not result_df.empty:
                    st.success(f"✅ Retrieved {len(result_df)} rows")
                    
                    # Display with pagination
                    if len(result_df) > 500:
                        st.info(f"Large dataset ({len(result_df)} rows). Showing first 500 rows.")
                        st.dataframe(result_df.head(500), use_container_width=True, height=400)
                    else:
                        st.dataframe(result_df, use_container_width=True, height=400)
                    
                    st.session_state.last_query_data = result_df
                    
                    # Export options with CSV prominent
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        csv_data = result_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv_data,
                            f"argo_query_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    with export_col2:
                        json_data = result_df.to_json(orient='records', indent=2)
                        st.download_button(
                            "📄 Download JSON",
                            json_data,
                            f"argo_query_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            "application/json",
                            use_container_width=True
                        )
                    
                    # Show generated SQL
                    with st.expander("🔍 View Generated SQL"):
                        st.code(sql, language="sql")
                else:
                    st.error(f"Query execution failed: {error}")
            else:
                st.error(sql)

def render_language_tab():
    """Clean language selection interface with fixed duplicate keys"""
    st.markdown("### 🌐 Language Settings")
    
    # Center the language selection
    col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
    
    with col_center:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); 
                   padding: 2rem; border-radius: 16px; text-align: center; 
                   border: 1px solid #475569;">
            <h3 style="color: #60a5fa; margin-bottom: 1rem;">Select Analysis Language</h3>
            <p style="color: #94a3b8; margin-bottom: 1.5rem;">
                Choose your preferred language for AI responses, explanations, and analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Language selection with flags/icons
        st.markdown("#### Available Languages")
        
        # Group languages by region with unique languages only
        language_groups = {
            "🌍 European": ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian"],
            "🌏 Asian": ["Japanese", "Chinese", "Arabic"],
            "🇮🇳 Indian": ["Hindi", "Telugu"]
        }
        
        current_language = st.session_state.current_language
        
        # Track button counter for unique keys
        button_counter = 0
        
        for group_name, languages in language_groups.items():
            st.markdown(f"**{group_name}**")
            lang_cols = st.columns(4)
            
            for i, language in enumerate(languages):
                if language in LANGUAGES:  # Make sure language exists in our dict
                    with lang_cols[i % 4]:
                        button_type = "primary" if current_language == language else "secondary"
                        if st.button(language, key=f"lang_btn_{button_counter}", type=button_type, use_container_width=True):
                            st.session_state.current_language = language
                            st.success(f"Language changed to {language}!")
                            st.rerun()
                        button_counter += 1
            
            st.markdown("---")
        
        # Current language status
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); 
                   padding: 1rem; border-radius: 8px; text-align: center; margin-top: 1rem;">
            <strong style="color: #d1fae5;">Current Language: {current_language}</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Custom language input with validation
        st.markdown("#### 🔧 Custom Language")
        custom_lang = st.text_input("Enter language name:", placeholder="e.g., Korean, Dutch, Swedish")
        
        if st.button("Set Custom Language", key="custom_lang_btn", use_container_width=True) and custom_lang.strip():
            # Basic format validation first
            if len(custom_lang.strip()) > 2 and custom_lang.strip().replace(' ', '').isalpha():
                # Use Gemini to validate if it's a real language
                with st.spinner("Validating language..."):
                    if st.session_state.gemini_model:
                        try:
                            validation_prompt = f"""
                            Is "{custom_lang.strip()}" a real language that exists in the world?
                            Respond with only "YES" if it's a real language, or "NO" if it's not a real language.
                            Consider major languages, regional languages, indigenous languages, and historical languages.
                            """
                            
                            response = st.session_state.gemini_model.generate_content(
                                validation_prompt,
                                generation_config=genai.types.GenerationConfig(
                                    temperature=0.1,
                                    max_output_tokens=10,
                                )
                            )
                            
                            if response and response.text and "YES" in response.text.upper():
                                st.session_state.current_language = custom_lang.strip().title()
                                st.success(f"Custom language set to: {custom_lang.strip().title()}")
                                st.rerun()
                            else:
                                st.error("Language does not exist. Please enter a real language name.")
                        except:
                            # Fallback if API fails
                            st.session_state.current_language = custom_lang.strip().title()
                            st.warning(f"Could not validate language, but set to: {custom_lang.strip().title()}")
                            st.rerun()
                    else:
                        # Fallback if no AI model
                        st.session_state.current_language = custom_lang.strip().title()
                        st.warning(f"AI validation unavailable, but set to: {custom_lang.strip().title()}")
                        st.rerun()
            else:
                st.error("Please choose a real language name (letters only, at least 3 characters)")
        
        st.caption("All AI responses, explanations, and analysis will be provided in the selected language.")

def main():
    """Main application"""
    initialize_session_state()
    
    # Page routing
    if st.session_state.page == 'map':
        render_map_page()
    elif st.session_state.page == 'analysis':
        render_analysis_page()
    else:
        st.session_state.page = 'map'
        st.rerun()

if __name__ == "__main__":
    main()