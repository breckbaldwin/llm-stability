import streamlit as st
import json
import sys
import os

ss = st.session_state

def get_config_file() -> None:
    """Reads in query params from streamlit app url (ex: https://<app url>/?file=multipage.json) the config file name
    is extracted if the query params contains the key `file`. otherwise the config used will be the designated
    default config file. The valid file path for the config file is stored in ss.config_file
    """
    indent = "        "
    if ss.config is not None:
        print(f"{indent}config already loaded, not loading")
        return
    print(f"{indent}Getting config file")
    params = st.query_params.to_dict()
    print(f"{indent}Query Parameters From URL:{json.dumps(params)}")
    if "file" not in params:
        params["file"] = ss.default_config
        print(f"{indent}No URL specified config, going with file: {params}")
    ss.config_file = params["file"]
    print(f"{indent}Loading config from {ss.config_file}")
    try:
        with open(os.path.join("configs", ss.config_file), "r") as file:
            ss.config = json.loads(file.read())
    except FileNotFoundError as e:
        st.warning(f"Configuration file {ss.config_file} not found!")
        st.stop()
    print(f"{indent}Loaded config file from disk")


if __name__ == "__main__":
    # run as streamlit run cells/configure_from_url.py 
    ss = st.session_state
    if 'config' not in ss:
        ss.config = None
    if 'default_config' not in ss:
        ss.default_config = 'generalized.json'
    get_config_file()
    st.markdown(f"Config file is: {ss.config_file}")
    st.json(ss.config)
    st.text_input("URL to test:", 
                    value="http://localhost:8501?file=test_config.json",   
                    key="test_url_ti")
    if st.button("Test config file url load"):
        ss.config = None
        st.markdown(f"[{ss.test_url_ti}]({ss.test_url_ti})")