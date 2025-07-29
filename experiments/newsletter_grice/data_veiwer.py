import streamlit as st
import pandas as pd
import json


# Example: replace with your real data source
@st.cache_data
def load_data():
    # Replace with your actual data file
    df = pd.read_csv("display_df.csv")
    return df[df['run'] == 0]

df = load_data()

# Ensure required columns exist
required_columns = {'rubric_id', 'response', 'parsed_answer'}
if not required_columns.issubset(df.columns):
    st.error(f"Missing required columns: {required_columns - set(df.columns)}")
    st.stop()

# Sidebar: rubric_id selector
rubric_ids = df['rubric_id'].dropna().unique()
selected_id = st.sidebar.selectbox("Select rubric_id", sorted(rubric_ids))

# Filter DataFrame
matching_rows = df[(df['rubric_id'] == selected_id)]

# Display responses
st.markdown(f"### Responses for rubric_id: `{selected_id}` Truth: {matching_rows['gt'].iloc[0]}")

if matching_rows.empty:
    st.warning("No matching rows found.")
else:
    for idx, row in matching_rows.iterrows():
        st.markdown(f"---{row['task/model']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Response**")
            st.text(row['response'])
        with col2:
            st.markdown("**Parsed Answer**")
            st.text(row['parsed_answer'])
        with col3:
            st.markdown("**Prompt**")
            st.json(json.loads(row['prompt']), expanded=False)
