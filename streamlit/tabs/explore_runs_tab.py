import streamlit as st
import cells.cell_util

ss = st.session_state

page = "Explore Runs"
print(f"Rendering tab explore_runs_tab.py, menu item: {page}")
cells.cell_util.render(ss.config, page)
