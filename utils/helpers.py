import pandas as pd
from datetime import datetime

def load_csv(file) -> pd.DataFrame:
    """Load uploaded CSV file into a DataFrame."""
    df = pd.read_csv(file)
    return df

def get_dataframe_summary(df: pd.DataFrame) -> dict:
    """Return a summary of the dataframe."""
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include='number').empty else {}
    }
    return summary

def dataframe_to_string(df: pd.DataFrame, max_rows: int = 5) -> str:
    """Convert dataframe to a readable string for the AI agent."""
    summary = get_dataframe_summary(df)
    text = f"""
Dataset Overview:
- Total Rows: {summary['rows']}
- Total Columns: {summary['columns']}
- Column Names: {', '.join(summary['column_names'])}
- Data Types: {summary['dtypes']}
- Missing Values: {summary['missing_values']}

First {max_rows} rows:
{df.head(max_rows).to_string()}
    """
    return text.strip()

def export_chat_to_text(messages: list) -> str:
    """Export chat history to a text string."""
    lines = []
    lines.append("=" * 50)
    lines.append("🪶 QueryQuill - Chat Export")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 50)
    lines.append("")

    for msg in messages:
        if msg["role"] == "user":
            lines.append(f"👤 You:")
            lines.append(f"   {msg['content']}")
            lines.append("")
        elif msg["role"] == "assistant":
            lines.append(f"🪶 QueryQuill:")
            lines.append(f"   {msg['content']}")
            lines.append("")

    return "\n".join(lines)

def get_quick_insights(df: pd.DataFrame) -> list:
    """Generate quick automatic insights from the dataframe."""
    insights = []

    # Total records
    insights.append(f"📋 Dataset has **{df.shape[0]:,} rows** and **{df.shape[1]} columns**")

    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"⚠️ Found **{missing:,} missing values** across the dataset")
    else:
        insights.append("✅ No missing values found in the dataset")

    # Numeric insights
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    for col in numeric_cols[:3]:
        insights.append(
            f"📊 **{col}**: min={df[col].min():,.2f}, "
            f"max={df[col].max():,.2f}, "
            f"avg={df[col].mean():,.2f}"
        )

    # Categorical insights
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols[:2]:
        top_val = df[col].value_counts().index[0]
        top_count = df[col].value_counts().iloc[0]
        insights.append(
            f"🏷️ **{col}**: {df[col].nunique()} unique values, "
            f"most common: '{top_val}' ({top_count:,} times)"
        )

    return insights