import streamlit as st
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pages
from pages.home import show_home
from pages.analytics import show_analytics
from pages.settings import show_settings
from pages.about import show_about

# Configure the Streamlit app
st.set_page_config(
    page_title="Traffic Counter",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Analytics", "Settings", "About"]
    )
    
    # Display the selected page
    if page == "Home":
        show_home()
    elif page == "Analytics":
        show_analytics()
    elif page == "Settings":
        show_settings()
    elif page == "About":
        show_about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Traffic Counter")
    st.sidebar.markdown("Version 1.0")
    st.sidebar.markdown("© 2024 All Rights Reserved")

if __name__ == "__main__":
    main() 