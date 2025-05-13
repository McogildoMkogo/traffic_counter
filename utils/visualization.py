import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from pathlib import Path
from plotly.subplots import make_subplots

def plot_vehicle_distribution(stats):
    """Create a pie chart showing the distribution of vehicle types."""
    if not stats:
        return None
    
    df = pd.DataFrame(list(stats.items()), columns=['Vehicle Type', 'Count'])
    fig = px.pie(df, values='Count', names='Vehicle Type',
                 title='Vehicle Type Distribution',
                 color_discrete_sequence=px.colors.qualitative.Set3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_direction_distribution(direction_stats):
    """Create a stacked bar chart showing vehicle counts by direction."""
    if not direction_stats:
        return None
    
    data = []
    for direction, counts in direction_stats.items():
        for vehicle_type, count in counts.items():
            data.append({
                'Direction': direction.capitalize(),
                'Vehicle Type': vehicle_type,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    if not df.empty:
        fig = px.bar(df, x='Direction', y='Count', color='Vehicle Type',
                    title='Traffic Direction Analysis',
                    color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(barmode='stack')
        return fig
    return None

def plot_hourly_pattern(time_stats):
    """Create a line chart showing hourly traffic patterns."""
    if not time_stats:
        return None
    
    data = []
    for timestamp, counts in time_stats.items():
        hour = pd.to_datetime(timestamp)
        for vehicle_type, count in counts.items():
            data.append({
                'Hour': hour,
                'Vehicle Type': vehicle_type,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    if not df.empty:
        fig = px.line(df, x='Hour', y='Count', color='Vehicle Type',
                     title='Hourly Traffic Pattern',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Vehicle Count')
        return fig
    return None

def plot_speed_distribution(stats_df):
    """Create a histogram showing the distribution of vehicle speeds."""
    if stats_df is None or stats_df.empty or 'speed' not in stats_df.columns:
        return None
    
    fig = px.histogram(stats_df, x='speed', nbins=20,
                      title='Speed Distribution',
                      labels={'speed': 'Speed (km/h)', 'count': 'Frequency'},
                      color_discrete_sequence=['rgb(102, 197, 204)'])
    fig.update_traces(opacity=0.75)
    return fig

def create_metrics_grid(stats, direction_stats):
    """Create a grid of key metrics for the dashboard."""
    total_vehicles = sum(stats.values())
    
    # Calculate direction totals
    north_total = sum(direction_stats['north'].values())
    south_total = sum(direction_stats['south'].values())
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Vehicles", total_vehicles)
    with col2:
        st.metric("Northbound", north_total)
    with col3:
        st.metric("Southbound", south_total)

def plot_daily_pattern(stats_df):
    """Create a bar chart showing daily traffic patterns."""
    if stats_df is None or stats_df.empty or 'timestamp' not in stats_df.columns:
        return None
    
    # Convert timestamp to datetime if it's not already
    stats_df['timestamp'] = pd.to_datetime(stats_df['timestamp'])
    
    # Group by date and vehicle type
    daily_counts = stats_df.groupby([
        stats_df['timestamp'].dt.date,
        'vehicle_type'
    ]).size().reset_index(name='count')
    
    fig = px.bar(daily_counts, x='timestamp', y='count',
                 color='vehicle_type', title='Daily Traffic Pattern',
                 labels={'timestamp': 'Date', 'count': 'Vehicle Count',
                        'vehicle_type': 'Vehicle Type'},
                 color_discrete_sequence=px.colors.qualitative.Set3)
    
    fig.update_layout(barmode='stack')
    return fig

def plot_peak_hours(stats_df):
    """Create a heatmap showing peak traffic hours."""
    if stats_df is None or stats_df.empty or 'timestamp' not in stats_df.columns:
        return None
    
    # Convert timestamp to datetime if it's not already
    stats_df['timestamp'] = pd.to_datetime(stats_df['timestamp'])
    
    # Group by hour and weekday
    hourly_counts = stats_df.groupby([
        stats_df['timestamp'].dt.hour,
        stats_df['timestamp'].dt.day_name()
    ]).size().reset_index(name='count')
    
    # Pivot the data for the heatmap
    pivot_table = hourly_counts.pivot(
        index='hour',
        columns='timestamp',
        values='count'
    ).fillna(0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        hoverongaps=False))
    
    fig.update_layout(
        title='Peak Traffic Hours',
        xaxis_title='Day of Week',
        yaxis_title='Hour of Day'
    )
    return fig

def export_analysis_report(stats_df):
    """Generate a detailed analysis report from the traffic statistics."""
    if stats_df is None or stats_df.empty:
        return None
    
    # Convert timestamp to datetime if it's not already
    stats_df['timestamp'] = pd.to_datetime(stats_df['timestamp'])
    
    # Calculate summary statistics
    analysis = {
        'Total Vehicles': len(stats_df),
        'Average Speed': stats_df['speed'].mean(),
        'Peak Hour': stats_df.groupby(stats_df['timestamp'].dt.hour)['vehicle_type'].count().idxmax(),
        'Most Common Vehicle': stats_df['vehicle_type'].mode().iloc[0],
        'Date Range': f"{stats_df['timestamp'].min()} to {stats_df['timestamp'].max()}"
    }
    
    # Create a DataFrame for export
    analysis_df = pd.DataFrame([analysis])
    return analysis_df

def load_traffic_data():
    try:
        return pd.read_csv("traffic_stats.csv")
    except:
        return None

def create_time_series_plot(df):
    """Create a time series plot of vehicle counts."""
    fig = px.line(df, x='timestamp', y=['total_count', 'north_count', 'south_count'],
                  title='Traffic Patterns Over Time',
                  labels={'value': 'Vehicle Count', 'timestamp': 'Time'},
                  template='plotly_white')
    return fig

def create_direction_pie_chart(df):
    """Create a pie chart showing traffic distribution by direction."""
    direction_data = {
        'Direction': ['Northbound', 'Southbound'],
        'Count': [df['north_count'].iloc[-1], df['south_count'].iloc[-1]]
    }
    fig = px.pie(direction_data, values='Count', names='Direction',
                 title='Traffic Distribution by Direction',
                 template='plotly_white')
    return fig

def create_speed_histogram(df):
    """Create a histogram of vehicle speeds."""
    fig = px.histogram(df, x='avg_speed',
                      title='Vehicle Speed Distribution',
                      labels={'avg_speed': 'Speed (km/h)', 'count': 'Number of Vehicles'},
                      template='plotly_white')
    return fig

def create_hourly_pattern(df):
    """Create an hourly traffic pattern visualization."""
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_avg = df.groupby('hour')['total_count'].mean().reset_index()
    
    fig = px.bar(hourly_avg, x='hour', y='total_count',
                 title='Average Hourly Traffic Pattern',
                 labels={'hour': 'Hour of Day', 'total_count': 'Average Vehicle Count'},
                 template='plotly_white')
    return fig

def create_dashboard(df):
    if df.empty:
        return None

    vehicle_types = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    type_values = [df.get(col, pd.Series([0])).iloc[-1] if col in df else 0 for col in vehicle_types]
    type_labels = [col.title() for col in vehicle_types]

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # Pie chart in (1,2)
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ],
        subplot_titles=(
            'Vehicle Count Over Time',
            'Vehicle Type Distribution',
            'Direction Distribution',
            'Average Speed Over Time',
            'Vehicle Types Over Time',
            'Hourly Traffic Pattern'
        )
    )

    # 1. Total Vehicle Count Over Time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['total_count'],
                  name="Total Count", line=dict(color='blue')),
        row=1, col=1
    )

    # 2. Vehicle Type Distribution (Pie Chart)
    fig.add_trace(
        go.Pie(labels=type_labels, values=type_values,
               name="Vehicle Types"),
        row=1, col=2
    )

    # 3. Direction Distribution
    fig.add_trace(
        go.Bar(x=['Northbound', 'Southbound'],
               y=[df['north_count'].iloc[-1], df['south_count'].iloc[-1]],
               name="Direction"),
        row=2, col=1
    )

    # 4. Average Speed Over Time
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['avg_speed'],
                  name="Average Speed", line=dict(color='red')),
        row=2, col=2
    )

    # 5. Vehicle Types Over Time
    for vehicle_type in vehicle_types:
        if vehicle_type in df:
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df[vehicle_type],
                          name=vehicle_type.title()),
                row=3, col=1
            )

    # 6. Hourly Traffic Pattern
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly_counts = df.groupby('hour')['total_count'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=hourly_counts['hour'],
               y=hourly_counts['total_count'],
               name="Hourly Pattern"),
        row=3, col=2
    )

    fig.update_layout(height=1000, showlegend=True,
                     title_text="Traffic Analysis Dashboard")
    
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=3, col=2)
    
    fig.update_yaxes(title_text="Vehicle Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Speed (km/h)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_yaxes(title_text="Average Count", row=3, col=2)

    return fig 