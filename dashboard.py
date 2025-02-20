import streamlit as st
import pandas as pd
import json

def load_logs():
    """Load chat logs from the JSON file."""
    try:
        with open("chat_logs.json", "r") as file:
            data = [json.loads(line) for line in file]
        return pd.DataFrame(data)
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp", "user_id", "query", "intent", "response"])

def run_dashboard():
    """Run the chatbot analytics dashboard using Streamlit."""
    st.title("Chatbot Analytics Dashboard")

    df = load_logs()

    if df.empty:
        st.warning("No chatbot interactions logged yet.")
        return

    # Display metrics
    st.subheader("📊 Chatbot Usage Summary")
    st.metric("Total Queries", len(df))

    # Show most common topics (if intent data exists)
    if "intent" in df.columns:
        st.subheader("🔥 Most Common Topics")
        st.bar_chart(df["intent"].value_counts())

    # Show recent chat history
    st.subheader("📝 Recent Interactions")
    st.dataframe(df.tail(10))

if __name__ == "__main__":
    run_dashboard()
