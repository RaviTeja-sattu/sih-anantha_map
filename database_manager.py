import streamlit as st
import psycopg2
import pandas as pd
import os
from typing import Tuple, Optional

class DatabaseManager:
    """Database operations with Google Cloud PostgreSQL connection"""
    
    @staticmethod
    def get_db_config():
        """Get database configuration exactly like your working demo"""
        try:
            # Try Streamlit secrets first (for Streamlit Cloud deployment)
            return {
                'host': st.secrets["connections"]["postgresql"]["host"],
                'database': st.secrets["connections"]["postgresql"]["database"],
                'user': st.secrets["connections"]["postgresql"]["username"],
                'password': st.secrets["connections"]["postgresql"]["password"],
                'port': int(st.secrets["connections"]["postgresql"]["port"])
            }
        except:
            # Fallback to your exact working configuration
            return {
                'host': '136.114.8.204',
                'database': 'postgres',
                'user': 'postgres',
                'password': 'nOVEMBER@141530',
                'port': 5432
            }
    
    @staticmethod
    def get_connection():
        """Establish database connection with enhanced error handling"""
        try:
            config = DatabaseManager.get_db_config()
            
            conn = psycopg2.connect(
                host=config['host'],
                database=config['database'],
                user=config['user'],
                password=config['password'],
                port=config['port'],
                connect_timeout=30,
                sslmode='prefer',
                options='-c statement_timeout=30000'  # 30 second timeout for queries
            )
            
            # Test connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.fetchone()
            cursor.close()
            
            return conn
            
        except psycopg2.OperationalError as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                st.error("Database connection timeout. Please check your network connection.")
            elif "authentication failed" in error_msg.lower():
                st.error("Database authentication failed. Please check credentials.")
            elif "could not connect" in error_msg.lower():
                st.error("Could not connect to database server. Please check host and port.")
            else:
                st.error(f"Database connection failed: {error_msg}")
            return None
            
        except Exception as e:
            st.error(f"Unexpected database error: {e}")
            return None
    
    @staticmethod
    def get_table_schema():
        """Get the actual schema of argo_data_clean table"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return {}
            
            cursor = conn.cursor()
            
            # Get column information for argo_data_clean
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'public' 
                AND table_name = 'argo_data_clean'
                ORDER BY ordinal_position;
            """)
            
            columns = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            
            return columns
            
        except Exception as e:
            st.error(f"Error getting table schema: {e}")
            return {}
    
    @staticmethod
    @st.cache_data(ttl=1800, show_spinner=False)
    def load_float_metadata():
        """Load float metadata for mapping with caching"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return pd.DataFrame()
            
            query = """
            SELECT 
                float_id, 
                launch_latitude, 
                launch_longitude, 
                centroid_lat, 
                centroid_lon,
                dominant_region, 
                num_profiles, 
                end_mission_status, 
                project_name, 
                pi_name,
                start_date, 
                end_mission_date
            FROM argo_metadata 
            WHERE (launch_latitude IS NOT NULL AND launch_longitude IS NOT NULL)
               OR (centroid_lat IS NOT NULL AND centroid_lon IS NOT NULL)
            ORDER BY num_profiles DESC
            LIMIT 5000
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if not df.empty:
                # Use launch coordinates first, fall back to centroid
                df['display_lat'] = df['launch_latitude'].fillna(df['centroid_lat'])
                df['display_lon'] = df['launch_longitude'].fillna(df['centroid_lon'])
                
                # Remove rows with no valid coordinates
                df = df.dropna(subset=['display_lat', 'display_lon'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading float metadata: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def get_basic_profile_data(selected_floats):
        """Get basic profile data for visualization with proper float filtering"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return pd.DataFrame(), "Database connection failed"
            
            # Build query with selected floats - float_id column exists
            float_list = ', '.join(map(str, selected_floats))
            
            query = f"""
            SELECT 
                float_id,
                profile,
                date,
                latitude,
                longitude,
                pres_adj_dbar as depth,
                temp_adj_c as temperature,
                psal_adj_psu as salinity
            FROM argo_data_clean 
            WHERE float_id IN ({float_list})
              AND pres_adj_dbar IS NOT NULL
              AND temp_adj_c IS NOT NULL
              AND psal_adj_psu IS NOT NULL
              AND temp_adj_c BETWEEN -5 AND 50
              AND psal_adj_psu BETWEEN 10 AND 45
              AND pres_adj_dbar BETWEEN 0 AND 2000
            ORDER BY float_id, profile, pres_adj_dbar
            LIMIT 2000
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            return df, None
            
        except Exception as e:
            return pd.DataFrame(), f"Query execution error: {e}"
    
    @staticmethod
    def execute_query(sql: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Execute SQL query with enhanced error handling and schema adaptation"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return None, "Database connection failed"
            
            # Add query timeout and memory limits
            cursor = conn.cursor()
            cursor.execute("SET statement_timeout = '60s';")
            cursor.execute("SET work_mem = '256MB';")
            
            # Check if query contains quality control columns that might not exist
            sql_lower = sql.lower()
            if 'temp_adj_qc' in sql_lower or 'psal_adj_qc' in sql_lower:
                # Remove quality control conditions since these columns don't exist
                import re
                sql = re.sub(r"AND\s+\w*\.?temp_adj_qc\s*=\s*'1'\s*", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"AND\s+\w*\.?psal_adj_qc\s*=\s*'1'\s*", "", sql, flags=re.IGNORECASE)
                sql = re.sub(r"WHERE\s+\w*\.?temp_adj_qc\s*=\s*'1'\s+AND\s+", "WHERE ", sql, flags=re.IGNORECASE)
                sql = re.sub(r"WHERE\s+\w*\.?psal_adj_qc\s*=\s*'1'\s+AND\s+", "WHERE ", sql, flags=re.IGNORECASE)
            
            # Execute the main query
            df = pd.read_sql(sql, conn)
            conn.close()
            
            return df, None
            
        except psycopg2.errors.QueryCanceled:
            return None, "Query timeout (60 seconds exceeded). Try a more specific query."
            
        except psycopg2.errors.OutOfMemory:
            return None, "Query requires too much memory. Try limiting results with LIMIT clause."
            
        except psycopg2.errors.SyntaxError as e:
            return None, f"SQL syntax error: {e}"
            
        except psycopg2.errors.UndefinedTable as e:
            return None, f"Table not found: {e}"
            
        except psycopg2.errors.UndefinedColumn as e:
            return None, f"Column not found: {e}"
            
        except Exception as e:
            return None, f"Query execution error: {e}"
    
    @staticmethod
    def test_connection():
        """Test database connection and return status"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return False, "Connection failed"
            
            cursor = conn.cursor()
            
            # Test basic connection
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            # Check for required tables
            cursor.execute("""
                SELECT tablename, 
                       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public' 
                AND tablename IN ('argo_data_clean', 'argo_metadata')
                ORDER BY tablename;
            """)
            tables = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if len(tables) < 2:
                return False, f"Required tables not found. Found {len(tables)} of 2 required tables."
            
            return True, f"Connection successful. PostgreSQL version: {version[:50]}..."
            
        except Exception as e:
            return False, f"Connection test failed: {e}"
    
    @staticmethod
    def get_table_info():
        """Get information about database tables"""
        try:
            conn = DatabaseManager.get_connection()
            if not conn:
                return {}
            
            cursor = conn.cursor()
            
            # Get table information
            cursor.execute("""
                SELECT 
                    t.tablename,
                    pg_size_pretty(pg_total_relation_size('public.'||t.tablename)) as size,
                    (SELECT COUNT(*) FROM information_schema.columns 
                     WHERE table_schema = 'public' AND table_name = t.tablename) as column_count
                FROM pg_tables t
                WHERE t.schemaname = 'public' 
                AND t.tablename IN ('argo_data_clean', 'argo_metadata')
                ORDER BY t.tablename;
            """)
            
            table_info = {}
            for table_name, size, col_count in cursor.fetchall():
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                row_count = cursor.fetchone()[0]
                
                table_info[table_name] = {
                    'size': size,
                    'columns': col_count,
                    'rows': row_count
                }
            
            cursor.close()
            conn.close()
            
            return table_info
            
        except Exception as e:
            st.error(f"Error getting table info: {e}")
            return {}
    
    @staticmethod
    def validate_query(sql: str) -> Tuple[bool, str]:
        """Validate SQL query before execution"""
        sql_lower = sql.lower().strip()
        
        # Check for dangerous operations
        dangerous_keywords = [
            'drop', 'delete', 'truncate', 'alter', 'create', 
            'update', 'insert', 'grant', 'revoke'
        ]
        
        for keyword in dangerous_keywords:
            if f' {keyword} ' in f' {sql_lower} ':
                return False, f"Query contains potentially dangerous keyword: {keyword}"
        
        # Must be a SELECT query
        if not sql_lower.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        # Check for required tables
        required_tables = ['argo_data_clean', 'argo_metadata']
        has_required_table = any(table in sql_lower for table in required_tables)
        
        if not has_required_table:
            return False, "Query must reference argo_data_clean or argo_metadata tables"
        

        return True, "Query validation passed"
