"""
Visualization utilities for the Smart Agriculture System.
Contains functions for generating charts and plots.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_crop_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of crop distribution in dataset."""
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["Crop", "Count"]
    fig = px.bar(counts, x="Crop", y="Count", color="Count",
                 color_continuous_scale="Greens", title="Crop Distribution in Dataset")
    fig.update_layout(xaxis_tickangle=-45, height=400)
    return fig


def plot_feature_importance(importances: list, feature_names: list) -> go.Figure:
    """Horizontal bar chart of feature importances."""
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df = df.sort_values("Importance", ascending=True)
    fig = px.bar(df, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Viridis",
                 title="Feature Importance for Crop Prediction")
    fig.update_layout(height=350)
    return fig


def plot_growth_trend(df: pd.DataFrame) -> go.Figure:
    """Line chart of crop growth over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["day"], y=df["height_cm"],
        mode="lines+markers",
        name="Plant Height",
        line=dict(color="#2E7D32", width=3),
        marker=dict(size=8, color="#43A047"),
        fill="tozeroy",
        fillcolor="rgba(76, 175, 80, 0.15)"
    ))
    fig.update_layout(
        title="Crop Growth Monitoring",
        xaxis_title="Days After Planting",
        yaxis_title="Plant Height (cm)",
        height=400,
        template="plotly_white"
    )
    return fig


def plot_market_prices(df: pd.DataFrame, selected_crops: list) -> go.Figure:
    """Line chart of market price trends."""
    fig = go.Figure()
    colors = ["#2E7D32", "#F57F17", "#E65100", "#1565C0", "#AD1457",
              "#6A1B9A", "#00838F", "#4E342E", "#D84315", "#33691E"]
    for i, crop in enumerate(selected_crops):
        if crop in df.columns:
            fig.add_trace(go.Scatter(
                x=df["month"], y=df[crop],
                mode="lines+markers",
                name=crop.title(),
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
    fig.update_layout(
        title="Market Price Trends (INR per unit)",
        xaxis_title="Month",
        yaxis_title="Price (INR)",
        height=450,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3)
    )
    return fig


def plot_market_comparison_bar(df: pd.DataFrame, selected_crops: list) -> go.Figure:
    """Grouped bar chart comparing average prices."""
    avg_prices = {}
    for crop in selected_crops:
        if crop in df.columns:
            avg_prices[crop.title()] = df[crop].mean()
    
    fig = go.Figure(data=[
        go.Bar(x=list(avg_prices.keys()), y=list(avg_prices.values()),
               marker_color=["#2E7D32", "#F57F17", "#E65100", "#1565C0", "#AD1457"][:len(avg_prices)])
    ])
    fig.update_layout(
        title="Average Market Price Comparison (INR)",
        xaxis_title="Crop",
        yaxis_title="Average Price (INR)",
        height=400,
        template="plotly_white"
    )
    return fig


def plot_confidence_pie(class_names: list, confidences: list) -> go.Figure:
    """Pie chart showing prediction confidence distribution."""
    fig = go.Figure(data=[go.Pie(
        labels=class_names[:5],
        values=confidences[:5],
        hole=0.4,
        marker_colors=px.colors.sequential.Greens_r
    )])
    fig.update_layout(
        title="Top-5 Prediction Confidence",
        height=400
    )
    return fig


def plot_seasonal_heatmap(df: pd.DataFrame, crop: str) -> go.Figure:
    """Heatmap of seasonal price patterns."""
    if crop not in df.columns:
        return go.Figure()
    
    months = df["month"].tolist()
    prices = df[crop].tolist()
    
    # Reshape for heatmap (4 quarters x 3 months)
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    data = []
    for q_idx, q in enumerate(quarters):
        row = prices[q_idx * 3: (q_idx + 1) * 3]
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=["Month 1", "Month 2", "Month 3"],
        y=quarters,
        colorscale="Greens",
        text=[[f"INR {v}" for v in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 14}
    ))
    fig.update_layout(
        title=f"Quarterly Price Heatmap - {crop.title()}",
        height=350
    )
    return fig
