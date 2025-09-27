import streamlit as st
import google.generativeai as genai
import os
from typing import List, Optional

class AIManager:
    """AI query processing and response generation using Google Gemini"""
    
    @staticmethod
    def initialize_gemini():
        """Initialize Gemini AI model with fallback options"""
        try:
            # Get API key from Streamlit secrets or environment
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except (KeyError, AttributeError):
                api_key = os.getenv('GEMINI_API_KEY', 'AIzaSyCF1u-y3tCy2dKRqW0D4bWVKaDzOwdXmac')
            
            genai.configure(api_key=api_key)
            
            # Try different Gemini models in order of preference
            models = [
                "gemini-2.0-flash-exp",
                "gemini-2.0-flash", 
                "gemini-1.5-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash-001"
            ]
            
            for model_name in models:
                try:
                    model = genai.GenerativeModel(model_name)
                    
                    # Test the model with a simple query
                    test_response = model.generate_content(
                        "Test", 
                        generation_config=genai.types.GenerationConfig(
                            max_output_tokens=5,
                            temperature=0.1
                        )
                    )
                    
                    if test_response and test_response.text:
                        st.success(f"✅ AI Model loaded: {model_name}")
                        return model
                        
                except Exception as e:
                    st.warning(f"Failed to load {model_name}: {e}")
                    continue
            
            st.error("❌ No Gemini models available")
            return None
            
        except Exception as e:
            st.error(f"❌ Gemini initialization failed: {e}")
            return None
    
    @staticmethod
    def generate_sql_query(question: str, selected_floats: List[int], language: str = 'English') -> str:
        """Generate SQL query from natural language question"""
        if st.session_state.gemini_model is None:
            return "Error: AI model not available. Please check your Gemini API key."
        
        try:
            # Prepare float list for SQL IN clause
            float_list = ', '.join([str(f) for f in selected_floats]) if selected_floats else '0'
            
            # Enhanced prompt with correct schema information
            prompt = f"""
            You are an expert PostgreSQL and oceanographic data analyst for Argo floats.
            
            DATABASE SCHEMA:
            Table: argo_metadata (569 rows)
            - float_id (bigint): Unique identifier for each float
            - dominant_region (text): Ocean region (e.g., 'Indian Ocean', 'North Pacific')  
            - num_profiles (integer): Number of profiles collected
            - launch_date (text): When float was deployed
            - launch_latitude, launch_longitude (numeric): Deployment coordinates
            - centroid_lat, centroid_lon (numeric): Average position
            - end_mission_status (text): Current status
            - project_name (text): Research project
            - pi_name (text): Principal investigator
            
            Table: argo_data_clean (20.4 million rows)
            - float_id (bigint): Links to argo_metadata - THIS COLUMN EXISTS
            - profile (integer): Profile number
            - date (timestamp): Measurement date
            - latitude, longitude (numeric): Position coordinates  
            - pres_adj_dbar (numeric): Pressure/depth in decibars
            - temp_adj_c (numeric): Adjusted temperature in Celsius
            - psal_adj_psu (numeric): Adjusted salinity in PSU
            - unique_id (bigint): Unique identifier for each measurement
            
            DATA QUALITY NOTES:
            - Temperature range: -2°C to 57.6°C (some extreme values)
            - Salinity range: 0 to 171,521 PSU (some extreme outliers)  
            - Pressure range: -107 to 2,166 dbar (some negative values)
            - No quality control columns exist
            
            CRITICAL REQUIREMENTS:
            1. ALWAYS filter to selected floats: WHERE float_id IN ({float_list})
            2. Use realistic data quality filters:
               - temp_adj_c BETWEEN -5 AND 50 (remove extreme outliers)
               - psal_adj_psu BETWEEN 10 AND 45 (typical ocean salinity)
               - pres_adj_dbar BETWEEN 0 AND 2000 (positive pressures only)
            3. ALWAYS include float_id in SELECT clause for profile data
            4. Use LIMIT 2000 for performance (unless explicitly asked for more)
            5. Use clear, descriptive column aliases
            6. Handle NULL values appropriately  
            7. Only use SELECT statements - no modifications allowed
            
            OCEANOGRAPHIC CONTEXT:
            - Pressure (pres_adj_dbar) ≈ depth in meters
            - Temperature typically ranges -2°C to 40°C in oceans
            - Salinity typically ranges 30-40 PSU in open ocean
            - Profiles go from surface (low pressure) to deep (high pressure)
            
            USER QUESTION: "{question}"
            RESPONSE LANGUAGE: {language}
            
            Generate ONLY the PostgreSQL query. No explanations, comments, or markdown formatting.
            Make the query efficient and scientifically meaningful.
            """
            
            response = st.session_state.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                    top_p=0.8
                )
            )
            
            if response and response.text:
                sql = response.text.strip()
                
                # Clean up response - remove markdown formatting
                sql = sql.replace('```sql', '').replace('```', '').strip()
                
                # Basic SQL validation
                sql_lower = sql.lower()
                if not sql_lower.startswith('select'):
                    return "Error: Generated query is not a SELECT statement"
                
                # Check for dangerous operations
                dangerous = ['drop', 'delete', 'update', 'insert', 'truncate', 'alter']
                if any(word in sql_lower for word in dangerous):
                    return "Error: Generated query contains potentially dangerous operations"
                
                # Check for quality control columns that don't exist
                if 'temp_adj_qc' in sql_lower or 'psal_adj_qc' in sql_lower:
                    return "Error: Quality control columns (temp_adj_qc, psal_adj_qc) do not exist in this database"
                
                return sql
            else:
                return "Error: No SQL response generated from AI model"
                
        except Exception as e:
            return f"Error generating SQL: {str(e)}"
    
    @staticmethod
    def generate_response(question: str, context: str, mode: str, language: str) -> str:
        """Generate AI response based on selected mode and language"""
        if st.session_state.gemini_model is None:
            return "Error: AI model not available. Please check your Gemini API key."
        
        try:
            # Mode-specific configurations
            mode_configs = {
                'Think Deeper': {
                    'temperature': 0.3,
                    'max_tokens': 3000,
                    'style': 'Provide comprehensive, detailed analysis with step-by-step reasoning and scientific depth'
                },
                'Quick Answer': {
                    'temperature': 0.1,
                    'max_tokens': 800,
                    'style': 'Give concise, direct answers focused on key points'
                },
                'Research Mode': {
                    'temperature': 0.2,
                    'max_tokens': 4000,
                    'style': 'Provide detailed scientific research with explanations, context, and technical details'
                },
                'Creative': {
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'style': 'Think creatively and explore novel perspectives while maintaining scientific accuracy'
                }
            }
            
            config = mode_configs.get(mode, mode_configs['Quick Answer'])
            
            # Enhanced prompt for better responses
            prompt = f"""
            You are an expert oceanographer and Argo float specialist with deep knowledge of:
            - Physical oceanography and water mass properties
            - Argo float technology and data interpretation
            - Ocean circulation patterns and climate science
            - Data analysis and visualization techniques
            
            RESPONSE MODE: {mode}
            INSTRUCTIONS: {config['style']}
            LANGUAGE: {language} (respond entirely in this language)
            
            CONTEXT: {context}
            
            USER QUESTION: "{question}"
            
            Guidelines:
            - Be scientifically accurate and cite oceanographic principles
            - Explain technical concepts clearly
            - Use appropriate scientific terminology
            - Consider real-world oceanographic applications
            - If discussing data, mention quality control and limitations
            - Provide practical insights for ocean researchers
            - Note: This database does not have quality control flags, so data quality is ensured through range filters
            
            Respond entirely in {language} using the {mode} approach.
            """
            
            response = st.session_state.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=config['temperature'],
                    max_output_tokens=config['max_tokens'],
                    top_p=0.9,
                    top_k=40
                )
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return f"Error: No response generated in {mode} mode"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    @staticmethod
    def validate_language(language_name: str) -> bool:
        """Validate if a language name is real using AI"""
        if st.session_state.gemini_model is None:
            return True  # Default to accepting if no AI available
        
        try:
            validation_prompt = f"""
            Is "{language_name.strip()}" a real language that exists in the world?
            Consider major languages, regional languages, indigenous languages, historical languages, and sign languages.
            
            Respond with only "YES" if it's a real language, or "NO" if it's not a real language.
            """
            
            response = st.session_state.gemini_model.generate_content(
                validation_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                )
            )
            
            if response and response.text:
                return "YES" in response.text.upper()
            else:
                return True  # Default to accepting if no response
                
        except Exception:
            return True  # Default to accepting if validation fails
    
    @staticmethod
    def parse_visualization_request(query: str, available_columns: List[str]) -> dict:
        """Parse natural language visualization request using AI"""
        if st.session_state.gemini_model is None:
            # Fallback to simple parsing
            return AIManager._fallback_viz_parsing(query, available_columns)
        
        try:
            parsing_prompt = f"""
            Parse this data visualization request and return a JSON response.
            
            USER REQUEST: "{query}"
            AVAILABLE COLUMNS: {available_columns}
            
            Return valid JSON with these exact keys:
            {{
                "chart_type": "Line Chart" | "Scatter Plot" | "Box Plot" | "Histogram" | "3D Scatter",
                "x_column": "exact_column_name",
                "y_column": "exact_column_name", 
                "color_column": "exact_column_name" or null
            }}
            
            Guidelines:
            - Match column names EXACTLY from available columns
            - For histograms, x_column and y_column should be the same
            - Choose appropriate chart types based on the request
            - Use "float_id" for color when showing multiple floats
            - Consider oceanographic relationships (temp vs depth, salinity vs temp, etc.)
            - Default to meaningful combinations if request is unclear
            
            Return ONLY the JSON, no other text.
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
                    
                    # Validate and clean the response
                    chart_type = parsed.get("chart_type", "Scatter Plot")
                    x_col = parsed.get("x_column", "depth")
                    y_col = parsed.get("y_column", "temperature")
                    color_col = parsed.get("color_column")
                    
                    # Ensure columns exist
                    if x_col not in available_columns:
                        x_col = "depth" if "depth" in available_columns else available_columns[0]
                    if y_col not in available_columns:
                        y_col = "temperature" if "temperature" in available_columns else available_columns[0]
                    if color_col and color_col not in available_columns:
                        color_col = "float_id" if "float_id" in available_columns else None
                    
                    return {
                        "chart_type": chart_type,
                        "x_column": x_col,
                        "y_column": y_col,
                        "color_column": color_col
                    }
                    
                except json.JSONDecodeError:
                    return AIManager._fallback_viz_parsing(query, available_columns)
            else:
                return AIManager._fallback_viz_parsing(query, available_columns)
                
        except Exception:
            return AIManager._fallback_viz_parsing(query, available_columns)
    
    @staticmethod
    def _fallback_viz_parsing(query: str, available_columns: List[str]) -> dict:
        """Fallback visualization parsing when AI is unavailable"""
        query_lower = query.lower()
        
        # Determine chart type
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
        
        # Smart variable selection
        x_var = "depth"
        y_var = "temperature"
        color_var = "float_id" if "float_id" in available_columns else None
        
        # Check for specific mentions
        if 'temperature' in query_lower and 'salinity' in query_lower:
            x_var, y_var = "temperature", "salinity"
        elif 'salinity' in query_lower and 'depth' in query_lower:
            x_var, y_var = "depth", "salinity"
        elif 'latitude' in query_lower and 'longitude' in query_lower:
            x_var, y_var = "longitude", "latitude"
            color_var = None
        
        # Ensure variables exist in columns
        if x_var not in available_columns:
            x_var = available_columns[0]
        if y_var not in available_columns:
            y_var = available_columns[1] if len(available_columns) > 1 else available_columns[0]
        
        # For histograms, both variables should be the same
        if chart_type == "Histogram":
            if 'temperature' in query_lower:
                x_var = y_var = "temperature"
            elif 'salinity' in query_lower:
                x_var = y_var = "salinity"
            elif 'depth' in query_lower:
                x_var = y_var = "depth"
            color_var = None
        
        return {
            "chart_type": chart_type,
            "x_column": x_var,
            "y_column": y_var,
            "color_column": color_var
        }