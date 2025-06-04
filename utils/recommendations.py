import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_traffic_patterns(df):
    """Analyze traffic patterns and generate recommendations."""
    if df is None or df.empty:
        return []
    
    recommendations = []
    
    # Convert timestamp to datetime if it's not already
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 1. Peak Hour Analysis
    hourly_counts = df.groupby(df['timestamp'].dt.hour)['total_count'].mean()
    peak_hour = hourly_counts.idxmax()
    peak_count = hourly_counts.max()
    
    if peak_count > hourly_counts.mean() * 1.5:  # If peak is 50% higher than average
        recommendations.append({
            'type': 'peak_hour',
            'severity': 'high',
            'message': f'Heavy traffic detected during hour {peak_hour}:00. Consider implementing traffic management measures during this period.',
            'suggestion': 'Consider adjusting traffic light timings or implementing temporary lane changes during peak hours.'
        })
    
    # 2. Speed Analysis
    if 'avg_speed' in df.columns:
        avg_speed = df['avg_speed'].mean()
        if avg_speed < 30:  # If average speed is below 30 km/h
            recommendations.append({
                'type': 'speed',
                'severity': 'medium',
                'message': f'Average speed ({avg_speed:.1f} km/h) is below optimal levels.',
                'suggestion': 'Review speed limits and traffic flow patterns. Consider implementing traffic calming measures if necessary.'
            })
    
    # 3. Vehicle Type Distribution
    vehicle_types = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    vehicle_counts = {vtype: df[vtype].sum() for vtype in vehicle_types if vtype in df.columns}
    total_vehicles = sum(vehicle_counts.values())
    
    if total_vehicles > 0:
        for vtype, count in vehicle_counts.items():
            percentage = (count / total_vehicles) * 100
            if percentage > 70:  # If any vehicle type dominates
                recommendations.append({
                    'type': 'vehicle_distribution',
                    'severity': 'low',
                    'message': f'{vtype.title()}s make up {percentage:.1f}% of total traffic.',
                    'suggestion': f'Consider implementing {vtype}-specific lanes or traffic management strategies.'
                })
    
    # 4. Direction Analysis
    if 'north_count' in df.columns and 'south_count' in df.columns:
        north_total = df['north_count'].sum()
        south_total = df['south_count'].sum()
        total_direction = north_total + south_total
        
        if total_direction > 0:
            north_percentage = (north_total / total_direction) * 100
            if abs(north_percentage - 50) > 20:  # If traffic is significantly unbalanced
                direction = 'northbound' if north_percentage > 50 else 'southbound'
                recommendations.append({
                    'type': 'direction',
                    'severity': 'medium',
                    'message': f'Traffic is heavily skewed towards {direction} ({abs(north_percentage - 50):.1f}% difference).',
                    'suggestion': 'Consider implementing reversible lanes or adjusting traffic light timings to balance flow.'
                })
    
    # 5. Time-based Recommendations
    current_hour = datetime.now().hour
    if current_hour in hourly_counts.index:
        current_traffic = hourly_counts[current_hour]
        if current_traffic > hourly_counts.mean() * 1.3:  # If current traffic is 30% above average
            recommendations.append({
                'type': 'current_traffic',
                'severity': 'high',
                'message': 'Current traffic levels are above average.',
                'suggestion': 'Consider activating additional traffic management measures.'
            })
    
    return recommendations

def get_traffic_insights(df):
    """Generate traffic insights and statistics."""
    if df is None or df.empty:
        return None
    
    insights = {
        'total_vehicles': len(df),
        'average_speed': df['avg_speed'].mean() if 'avg_speed' in df.columns else None,
        'peak_hour': df.groupby(df['timestamp'].dt.hour)['total_count'].mean().idxmax(),
        'busiest_day': df.groupby(df['timestamp'].dt.day_name())['total_count'].mean().idxmax(),
        'vehicle_distribution': {},
        'direction_balance': None
    }
    
    # Vehicle distribution
    vehicle_types = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    for vtype in vehicle_types:
        if vtype in df.columns:
            insights['vehicle_distribution'][vtype] = df[vtype].sum()
    
    # Direction balance
    if 'north_count' in df.columns and 'south_count' in df.columns:
        north_total = df['north_count'].sum()
        south_total = df['south_count'].sum()
        total = north_total + south_total
        if total > 0:
            insights['direction_balance'] = {
                'north_percentage': (north_total / total) * 100,
                'south_percentage': (south_total / total) * 100
            }
    
    return insights

def generate_traffic_report(df):
    """Generate a comprehensive traffic analysis report."""
    if df is None or df.empty:
        return None
    
    recommendations = analyze_traffic_patterns(df)
    insights = get_traffic_insights(df)
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'insights': insights,
        'recommendations': recommendations,
        'summary': {
            'total_recommendations': len(recommendations),
            'high_priority': len([r for r in recommendations if r['severity'] == 'high']),
            'medium_priority': len([r for r in recommendations if r['severity'] == 'medium']),
            'low_priority': len([r for r in recommendations if r['severity'] == 'low'])
        }
    }
    
    return report 