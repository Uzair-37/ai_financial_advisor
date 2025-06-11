#!/usr/bin/env python3
"""
Full AI Financial Advisor - Uses complete project structure
Alternative entry point with all original features
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

# Configure page
st.set_page_config(
    page_title="AI Financial Advisor - Full Version",
    page_icon="🦙💰", 
    layout="wide"
)

try:
    # Import the comprehensive web app
    from web_app.app import main
    
    # Run the full application
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"""
    🚧 **Full Version Not Available**
    
    Missing dependencies: {str(e)}
    
    **Options:**
    1. Use `streamlit_app.py` for the simplified version
    2. Install missing packages: `pip install -r requirements.txt`
    3. Check that all files were uploaded correctly
    """)
    
    st.info("""
    💡 **Quick Fix:** Use the main `streamlit_app.py` file instead - it has all the core features and works independently!
    """)