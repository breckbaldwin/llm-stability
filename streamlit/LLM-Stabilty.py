import streamlit as st
import cells.cell_util


import sys

ss = st.session_state

menu_dict = {
    "Get help": None,
    "Report a Bug": None,
}
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=menu_dict,
)

# setup globals
if 'odd_even' not in ss:
        ss.odd_even = 0
ss.odd_even = ss.odd_even + 1 % 2
if "args_d" not in ss:
    ss.args_d = cells.cell_util.parse_args(sys.argv)
    print(f"Read invocation arguments, may have defaults: {ss.args_d}")
if "default_config" not in ss:
    if ss.args_d["config"] is not None:
        ss.default_config = ss.args_d["config"]
    else:
        ss.default_config = 'stability_ui.json' #'one_token.json'
if "sidebar_state" not in ss:
    ss.sidebar_state = "collapsed"
if "config" not in ss:
    ss.config = None
if "config_file" not in ss:
    ss.config_file = None
if "nothing_to_save" not in ss:
    ss.nothing_to_save = True
if "auth_d" not in ss:
    ss.auth_d = None
if "username" not in ss:
    ss.username = None
if "refresh_counter" not in ss:
    ss.refresh_counter = 0
if "exp_loaded" not in ss:
    ss.exp_loaded = False



print(f"-----------LLM-Stability REFRESH {ss.refresh_counter}-----------------")

if "loader_config" not in ss:
    print(f"setting ss.loader_config")
    ss.loader_config = {
        "kernel": [
            {"fn": "title", "import": "title"},
            {"fn": "get_config_file", "import": "configure_from_url"},
        ]
    }
page = "kernel"
print(f"Rendering with key: {page}")
cells.cell_util.render(ss.loader_config, page)

if ss.refresh_counter == 0:
    cells.cell_util.setup_globals(ss.config)

tabs = []

for tab in ss.config["tabs"]:
    print(f'adding tab: {tab["file_path"]}, {tab["tab_title"]}')
    tabs.append(st.Page(tab["file_path"], title=tab["tab_title"]))

pg = st.navigation(tabs)
pg.run()
ss.refresh_counter += 1

