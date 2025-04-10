import json

import streamlit as st
from googlesearch import search
from streamlit import session_state as sst

from utils import make_request_structured
from website_parser import get_url_text_and_images


def init_session_state(resources):
    if "g_init" not in sst:
        sst.g_init = True
        sst.g_query = ""
        sst.g_response = "{}"
        sst.g_generated_artifacts = {}
        sst.g_confirmed_artifacts = {}
        sst.resources = resources


def add_artifact(toggle_key, artifact_id, artifact):
    widget_state = sst[toggle_key]
    if widget_state:
        sst.g_confirmed_artifacts[artifact_id] = artifact
    else:
        sst.g_confirmed_artifacts.pop(artifact_id, None)


def display_generated_artifacts():
    st.subheader("Generierte Artefakte")
    if len(sst.g_generated_artifacts) == 0:
        st.write("Keine vorhanden")
        return
    for key, values in sst.g_generated_artifacts.items():
        artifact = values["content"]
        importance = values["importance"]
        reference = values["reference"]
        with st.container():
            columns = st.columns([1, 3, 1, 1], vertical_alignment="center")
            with columns[1]:
                st.markdown(artifact)
                st.markdown(f"*Sources:* {reference}")
            with columns[3]:
                # st.button(":heavy_check_mark:", key=f"button_{artifact}_check", on_click=add_artifact,
                #          kwargs={"artifact_id": key, "artifact": artifact})
                st.toggle("Übernehmen", key=f"button_{artifact}_check",
                          on_change=add_artifact,
                          kwargs={"toggle_key": f"button_{artifact}_check", "artifact_id": key, "artifact": artifact})
            # with columns[4]:
            #    st.button(":x:", key=f"button_{artifact}_discard", on_click=remove_artifact,
            #              kwargs={"artifact_id": key})
            # on_click=delete_artifact, kwargs={"category": category_selected, "artifact": artifact}
            st.divider()


def scrape_texts(query, num_results):
    sst.g_query = query
    texts = {}
    for url in search(query, num_results=num_results):
        st.write(url)
        try:
            text, _ = get_url_text_and_images(url)
            # st.write(text)
            texts[url] = text[:10000]
        except:
            print("Continuing")
    return texts


def format_func(option):
    return f":material/{option}:"


def resource_selection_view(selected_option):
    home_url = None
    if "home" in selected_option:
        with st.container(border=True):
            st.subheader("Website")
            home_url = st.text_input(label="URL der Website").strip()

    query = None
    number_entries_used = 0
    if "globe" in selected_option:
        with st.container(border=True):
            st.subheader("Google Suche")
            query = st.text_input(label="Search Query").strip()
            number_entries_used = st.number_input(label="Number of Entries Used", min_value=1, max_value=10, value=2)
    uploaded_files = None
    if "Description" in selected_option:
        with st.container(border=True):
            st.subheader("Dokumente")
            uploaded_files = st.file_uploader(label="Dokumente hochladen", type="pdf")
    return home_url, query, number_entries_used, uploaded_files


def artifact_generation_view():
    selected_resources = {}
    resources_keys = list(sst.resources.keys())
    if len(resources_keys) > 0:
        selected_keys = st.multiselect(label="Verwendete Elemente", options=resources_keys, default=resources_keys[0])
        for selected_key in selected_keys:
            selected_resources[selected_key] = sst.resources[selected_key]

    options = ["home", "globe", "Description"]
    selected_option = st.segmented_control(label="Zusätzliche Ressourcen", options=options, selection_mode='multi',
                                           format_func=format_func)
    texts = {}
    home_url, query, number_entries_used, uploaded_files = resource_selection_view(selected_option)
    home_url_given = home_url is not None and str(home_url).strip() != ""
    query_given = query is not None and str(query).strip() != ""
    disabled = not home_url_given and not query_given
    if st.button("Generate", disabled=disabled):
        if home_url_given:
            home_url_text, _ = get_url_text_and_images(home_url)
            texts[home_url] = home_url_text[:10000]
            st.write("URL retrieved")
        if query_given:
            texts_scrape = scrape_texts(query, number_entries_used)
            texts.update(texts_scrape)
        with st.spinner("Generating"):
            # st.json(texts)
            sst.g_response = make_request_structured(query, texts)
            if sst.g_response is not None and str(sst.g_response).strip() != "":
                try:
                    response_dict = json.loads(sst.g_response)
                    if len(response_dict) > 0:
                        for i, values in enumerate(response_dict["points"]):
                            sst.g_generated_artifacts[i] = values
                except Exception as e:
                    st.error("Result received but could not be processed")
                    print(e)
            else:
                st.warning("No results found")
    display_generated_artifacts()
    if len(sst.g_generated_artifacts) > 0:
        if st.button("Bestätigen"):
            return sst.g_confirmed_artifacts
    return {}
    # st.write(sst.g_generated_artifacts)
    # st.write(sst.g_confirmed_artifacts)


def get_confirmed_artifacts():
    return sst.g_confirmed_artifacts


if __name__ == "__main__":
    test_resources = {"InnovationIssue": "Stop small children from drowning", "TargetGroup": "careful parents"}
    init_session_state(test_resources)
    st.set_page_config(page_title="Align Guide", layout="wide",
                       initial_sidebar_state="collapsed")
    artifact_generation_view()
