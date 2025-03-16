import streamlit as st
import cells.cell_util

ss = st.session_state

page = "Home"
print(f"Rendering tab home_tab.py, menu item: {page}")
cells.cell_util.render(ss.config, page)
