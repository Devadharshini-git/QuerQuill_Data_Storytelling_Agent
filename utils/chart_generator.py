import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

# ── Detect if user wants a chart ─────────────────────────
def is_chart_request(user_input: str) -> bool:
    """Check if the user is asking for a visualization."""
    chart_keywords = [
        "plot", "chart", "graph", "visualize", "visualization",
        "show me", "display", "draw", "histogram", "bar",
        "pie", "scatter", "line", "distribution", "trend"
    ]
    user_lower = user_input.lower()
    return any(keyword in user_lower for keyword in chart_keywords)


# ── Ask LLM what chart to make ────────────────────────────
def get_chart_config(user_input: str, df: pd.DataFrame) -> dict:
    """Ask the LLM to decide what chart to generate."""
    llm = get_llm()

    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()

    prompt = f"""
You are a data visualization expert. Based on the user request and dataset, 
decide what chart to create.

User request: "{user_input}"

Available columns: {columns}
Column types: {dtypes}

Return ONLY a valid JSON object with these exact keys:
{{
    "chart_type": "bar" or "line" or "pie" or "scatter" or "histogram" or "box",
    "x_column": "column name or null",
    "y_column": "column name or null",
    "title": "chart title",
    "color_column": "column name or null"
}}

Rules:
- For categorical vs numeric: use bar chart
- For distribution of one numeric column: use histogram
- For time series: use line chart
- For part of whole: use pie chart
- For two numeric columns: use scatter chart
- Only use column names that exist in the dataset
- Return ONLY the JSON, no explanation
"""

    response = llm.invoke(prompt)
    
    try:
        clean = response.content.strip()
        clean = clean.replace("```json", "").replace("```", "").strip()
        config = json.loads(clean)
        return config
    except Exception:
        # Default fallback config
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        return {
            "chart_type": "bar",
            "x_column": cat_cols[0] if cat_cols else df.columns[0],
            "y_column": numeric_cols[0] if numeric_cols else df.columns[1],
            "title": "Data Chart",
            "color_column": None
        }


# ── Generate the actual chart ─────────────────────────────
def generate_chart(user_input: str, df: pd.DataFrame):
    """
    Generate a Plotly chart based on user request.
    Returns a plotly figure or None if generation fails.
    """
    try:
        config = get_chart_config(user_input, df)

        chart_type = config.get("chart_type", "bar")
        x_col = config.get("x_column")
        y_col = config.get("y_column")
        title = config.get("title", "Chart")
        color_col = config.get("color_column")

        # Validate columns exist
        if x_col and x_col not in df.columns:
            x_col = df.columns[0]
        if y_col and y_col not in df.columns:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            y_col = numeric_cols[0] if numeric_cols else None
        if color_col and color_col not in df.columns:
            color_col = None

        # ── Smart aggregation for large datasets ──────────
        MAX_POINTS = 50
        plot_df = df.copy()

        if chart_type in ["bar", "line"] and x_col and y_col:
            # Aggregate: group by x and mean of y
            if df[x_col].nunique() > MAX_POINTS:
                plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                plot_df = plot_df.sort_values(y_col, ascending=False).head(MAX_POINTS)
                title += f" (Top {MAX_POINTS} by avg)"
            else:
                plot_df = df.groupby(x_col)[y_col].mean().reset_index()

        elif chart_type == "pie" and x_col:
            # Limit pie to top 10 categories
            if df[x_col].nunique() > 10:
                top = df[x_col].value_counts().head(10).index
                plot_df = df[df[x_col].isin(top)]
                title += " (Top 10)"

        elif chart_type == "scatter" and x_col and y_col:
            # Sample for large datasets
            if len(df) > 500:
                plot_df = df.sample(500, random_state=42)
                title += " (500 sample)"

        elif chart_type == "histogram" and x_col:
            plot_df = df  # histogram handles large data fine

        # ── Generate chart ────────────────────────────────
        if chart_type == "bar":
            fig = px.bar(
                plot_df, x=x_col, y=y_col,
                title=title, color=color_col,
                template="plotly_dark"
            )

        elif chart_type == "line":
            fig = px.line(
                plot_df, x=x_col, y=y_col,
                title=title, color=color_col,
                template="plotly_dark"
            )

        elif chart_type == "pie":
            if y_col:
                fig = px.pie(
                    plot_df, names=x_col, values=y_col,
                    title=title, template="plotly_dark"
                )
            else:
                counts = plot_df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                fig = px.pie(
                    counts, names=x_col, values="count",
                    title=title, template="plotly_dark"
                )

        elif chart_type == "scatter":
            fig = px.scatter(
                plot_df, x=x_col, y=y_col,
                title=title, color=color_col,
                template="plotly_dark"
            )

        elif chart_type == "histogram":
            fig = px.histogram(
                plot_df, x=x_col,
                title=title, template="plotly_dark",
                nbins=30
            )

        elif chart_type == "box":
            fig = px.box(
                plot_df, x=x_col, y=y_col,
                title=title, template="plotly_dark"
            )

        else:
            fig = px.bar(
                plot_df, x=x_col, y=y_col,
                title=title, template="plotly_dark"
            )

        # ── Style ─────────────────────────────────────────
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            title_font_size=18,
            margin=dict(t=50, l=20, r=20, b=20)
        )

        return fig

    except Exception as e:
        print(f"Chart generation error: {e}")
        return None