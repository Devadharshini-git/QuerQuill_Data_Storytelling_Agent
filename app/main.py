import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.helpers import load_csv, get_dataframe_summary, dataframe_to_string
from utils.chart_generator import generate_chart, is_chart_request
from agents.data_agent import ask_agent, ChatHistoryManager, generate_suggested_questions
from rag.rag_pipeline import build_vectorstore, rag_answer

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="QueryQuill 🪶",
    page_icon="🪶",
    layout="wide"
)

st.title("🪶 QueryQuill")
st.subheader("Agentic AI-Powered Data Storytelling Assistant")
st.markdown("---")

# ── Session State Init ────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "summary" not in st.session_state:
    st.session_state.summary = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history_manager" not in st.session_state:
    st.session_state.chat_history_manager = ChatHistoryManager()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []
if "charts" not in st.session_state:
    st.session_state.charts = {}
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Your Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_new = load_csv(uploaded_file)
        st.session_state.df = df_new
        st.session_state.summary = get_dataframe_summary(df_new)
        st.success(f"✅ {uploaded_file.name}")
        st.markdown(f"**Rows:** {st.session_state.summary['rows']}")
        st.markdown(f"**Columns:** {st.session_state.summary['columns']}")
        if st.session_state.vectorstore is None:
            with st.spinner("🔍 Building RAG index..."):
                st.session_state.vectorstore = build_vectorstore(df_new)

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.charts = {}
        st.session_state.vectorstore = None
        st.session_state.chat_history_manager = ChatHistoryManager()
        st.session_state.suggested_questions = []
        st.session_state.pending_input = None

# ── Main Area ─────────────────────────────────────────────
if st.session_state.df is not None:
    df = st.session_state.df
    summary = st.session_state.summary

    tab1, tab2 = st.tabs(["🤖 Chat with Data", "📊 Data Preview"])

    # ════════════════════════════════════════════════════
    # TAB 1: CHAT
    # ════════════════════════════════════════════════════
    with tab1:

        # Generate suggested questions once
        if not st.session_state.suggested_questions:
            with st.spinner("🤔 Generating smart questions..."):
                dataset_str = dataframe_to_string(df)
                st.session_state.suggested_questions = generate_suggested_questions(dataset_str)

        # Suggested question buttons
        st.markdown("### 💡 Suggested Questions")
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.suggested_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sq_{i}"):
                    st.session_state.pending_input = question

        st.markdown("---")
        st.markdown("### 💬 Chat")

        # Display all chat messages including charts
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(msg["content"])
            elif msg["role"] == "chart":
                chart_key = msg["content"]
                if chart_key in st.session_state.charts:
                    st.plotly_chart(st.session_state.charts[chart_key])

        # ── Process pending input (from suggested questions) ──
        if st.session_state.pending_input:
            query = st.session_state.pending_input
            st.session_state.pending_input = None

            with st.spinner("🪶 Thinking..."):
                if st.session_state.vectorstore:
                    response = rag_answer(query, st.session_state.vectorstore)
                else:
                    response = ask_agent(
                        user_input=query,
                        dataset_summary=dataframe_to_string(df),
                        chat_history_manager=st.session_state.chat_history_manager
                    )

            st.session_state.messages.append({"role": "user", "content": query})
            st.session_state.messages.append({"role": "assistant", "content": response})

            if is_chart_request(query):
                with st.spinner("📊 Generating chart..."):
                    fig = generate_chart(query, df)
                if fig:
                    chart_key = f"chart_{len(st.session_state.charts)}"
                    st.session_state.charts[chart_key] = fig
                    st.session_state.messages.append({"role": "chart", "content": chart_key})

            st.rerun()

        # ── Chat input box ────────────────────────────────────
        user_input = st.chat_input("Ask anything about your data...")
        if user_input:
            with st.spinner("🪶 Thinking..."):
                if st.session_state.vectorstore:
                    response = rag_answer(user_input, st.session_state.vectorstore)
                else:
                    response = ask_agent(
                        user_input=user_input,
                        dataset_summary=dataframe_to_string(df),
                        chat_history_manager=st.session_state.chat_history_manager
                    )

            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": response})

            if is_chart_request(user_input):
                with st.spinner("📊 Generating chart..."):
                    fig = generate_chart(user_input, df)
                if fig:
                    chart_key = f"chart_{len(st.session_state.charts)}"
                    st.session_state.charts[chart_key] = fig
                    st.session_state.messages.append({"role": "chart", "content": chart_key})

            st.rerun()

    # ════════════════════════════════════════════════════
    # TAB 2: DATA PREVIEW
    # ════════════════════════════════════════════════════
    with tab2:
        col1, col2, col3 = st.columns(3)
        col1.metric("📋 Rows", summary["rows"])
        col2.metric("📊 Columns", summary["columns"])
        col3.metric("❓ Missing Values", sum(summary["missing_values"].values()))

        st.markdown("---")
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10))

        st.subheader("🔍 Column Information")
        col_info = pd.DataFrame({
            "Column": summary["column_names"],
            "Data Type": [summary["dtypes"][c] for c in summary["column_names"]],
            "Missing Values": [summary["missing_values"][c] for c in summary["column_names"]]
        })
        st.dataframe(col_info)

        if summary["numeric_summary"]:
            st.subheader("📈 Numeric Statistics")
            st.dataframe(df.describe())

else:
    st.info("👈 Upload a CSV file from the sidebar to get started!")
    st.markdown("""
    ### 🪶 What can QueryQuill do?
    - 📂 **Upload** any CSV dataset
    - 🤖 **Ask questions** in plain English
    - 📊 **Auto-generate** charts and visualizations
    - 📝 **Get insights** and data stories
    - 🔍 **Query** your data intelligently using AI
    """)