import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from pathlib import Path

def load_traffic_data():
    """Load and combine all traffic statistics files"""
    stats_dir = Path("traffic_stats")
    if not stats_dir.exists():
        return None
    
    dfs = []
    for file in stats_dir.glob("*.csv"):
        df = pd.read_csv(file)
        dfs.append(df)
    
    if not dfs:
        return None
    
    return pd.concat(dfs, ignore_index=True)

def show_analytics():
    st.title("Traffic Analytics")
    data_file = Path("traffic_stats.csv")
    if data_file.exists():
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.subheader("Traffic Patterns Over Time")
        # Plot each vehicle type over time
        vehicle_types = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
        fig_time = go.Figure()
        for vtype in vehicle_types:
            if vtype in df.columns:
                fig_time.add_trace(go.Scatter(x=df['timestamp'], y=df[vtype], mode='lines', name=vtype.title()))
        fig_time.update_layout(title='Vehicle Counts Over Time', xaxis_title='Time', yaxis_title='Vehicle Count')
        st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("Average Speed Over Time")
        fig_speed = px.line(df, x='timestamp', y='avg_speed',
                          title='Average Speed Over Time',
                          labels={'avg_speed': 'Speed (km/h)', 'timestamp': 'Time'})
        st.plotly_chart(fig_speed, use_container_width=True)

        st.subheader("Lane Distribution (Latest)")
        if not df.empty:
            latest = df.iloc[-1]
            direction_data = pd.DataFrame({
                'Lane': ['Lane 1', 'Lane 2'],
                'Count': [latest['north_count'], latest['south_count']]
            })
            fig_direction = px.bar(direction_data, x='Lane', y='Count',
                                  title='Traffic Distribution by Lane')
            st.plotly_chart(fig_direction, use_container_width=True)
        st.subheader("Raw Data")
        st.dataframe(df)
        if st.button("Export Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="traffic_analysis.csv",
                mime="text/csv"
            )
    else:
        st.info("No traffic data available yet. Start counting vehicles on the Home page to generate analytics.")

def show_analytics_old():
    st.title("Traffic Analytics 📊")
    
    # Load data
    df = load_traffic_data()
    
    if df is None:
        st.info("No traffic data available yet. Process some videos first!")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time range selector
    st.subheader("Time Range")
    cols = st.columns(2)
    with cols[0]:
        start_date = st.date_input("Start Date", df['timestamp'].min().date())
    with cols[1]:
        end_date = st.date_input("End Date", df['timestamp'].max().date())
    
    # Filter data by date range
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    filtered_df = df[mask]
    
    # Traffic Overview
    st.header("Traffic Overview")
    
    # Vehicle type distribution
    st.subheader("Vehicle Distribution")
    vehicle_counts = filtered_df['vehicle_type'].value_counts()
    fig = px.pie(values=vehicle_counts.values, names=vehicle_counts.index, title="Vehicle Type Distribution")
    st.plotly_chart(fig)
    
    # Hourly patterns
    st.subheader("Hourly Traffic Patterns")
    hourly_counts = filtered_df.groupby([filtered_df['timestamp'].dt.hour, 'vehicle_type']).size().unstack(fill_value=0)
    fig = px.line(hourly_counts, title="Hourly Traffic Volume")
    st.plotly_chart(fig)
    
    # Direction Analysis
    st.header("Direction Analysis")
    cols = st.columns(2)
    
    # North/South split
    with cols[0]:
        direction_counts = filtered_df['direction'].value_counts()
        fig = px.pie(values=direction_counts.values, names=direction_counts.index, title="Traffic Direction Split")
        st.plotly_chart(fig)
    
    # Direction by vehicle type
    with cols[1]:
        direction_vehicle = pd.crosstab(filtered_df['direction'], filtered_df['vehicle_type'])
        fig = px.bar(direction_vehicle, title="Vehicle Types by Direction")
        st.plotly_chart(fig)
    
    # Speed Analysis
    st.header("Speed Analysis")
    
    # Speed distribution
    fig = px.histogram(filtered_df, x='speed', nbins=30, title="Speed Distribution")
    fig.add_vline(x=filtered_df['speed'].mean(), line_dash="dash", line_color="red", annotation_text="Mean Speed")
    st.plotly_chart(fig)
    
    # Speed by vehicle type
    fig = px.box(filtered_df, x='vehicle_type', y='speed', title="Speed by Vehicle Type")
    st.plotly_chart(fig)
    
    # Time-based Analysis
    st.header("Time-based Analysis")
    
    # Daily patterns
    daily_counts = filtered_df.groupby([filtered_df['timestamp'].dt.date]).size()
    fig = px.line(daily_counts, title="Daily Traffic Volume")
    st.plotly_chart(fig)
    
    # Day of week patterns
    dow_counts = filtered_df.groupby([filtered_df['timestamp'].dt.day_name()]).size()
    # Reorder days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_counts = dow_counts.reindex(days_order)
    fig = px.bar(dow_counts, title="Traffic by Day of Week")
    st.plotly_chart(fig)
    
    # Export options
    st.header("Export Analysis")
    if st.button("Export Analysis to CSV"):
        # Create a more detailed analysis DataFrame
        analysis = pd.DataFrame({
            'Total_Vehicles': [len(filtered_df)],
            'Average_Speed': [filtered_df['speed'].mean()],
            'Peak_Hour': [hourly_counts.sum(axis=1).idxmax()],
            'Most_Common_Vehicle': [vehicle_counts.index[0]],
            'Date_Range': [f"{start_date} to {end_date}"]
        })
        
        # Save analysis
        analysis_path = Path("traffic_stats") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        analysis.to_csv(analysis_path, index=False)
        
        # Provide download button
        with open(analysis_path, 'rb') as file:
            st.download_button(
                label="Download Analysis CSV",
                data=file,
                file_name=f"traffic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            ) 