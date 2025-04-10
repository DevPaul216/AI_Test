import streamlit as st
from streamlit import session_state as sst

from shared_streamlit_views import prompting_view, additional_pdf
from utils import make_request
from website_parser import get_url_text_and_images, filter_invalid_image_paths, get_valid_image_urls

st.set_page_config(page_title="Align Prototype", layout="wide",
                   initial_sidebar_state="collapsed")


def init_session_state():
    if "init" not in sst:
        sst.init = True
        sst.editMode = False
        sst.messageContent = ""
        sst.text = ""
        sst.imagePaths = []
        sst.websiteURL = ""


def additional_information_view():
    st.subheader("Verwendete Informationen")
    website_url = st.text_input(label="Website URL", value=sst.websiteURL)
    with st.spinner(text="Processing..."):
        load_data_from_url(website_url)
        st.write("Laden:", sst.text[0:1000])
        st.image(sst.imagePaths, width=100)
    return website_url


def load_data_from_url(website_url):
    if str(website_url).strip() != "":
        sst.text, sst.imagePaths = get_url_text_and_images(website_url)
        sst.imagePaths = filter_invalid_image_paths(sst.imagePaths)
        sst.imagePaths = get_valid_image_urls(sst.imagePaths)

def assemble_view():
    if sst.editMode:
        st.text_area(label="State", value=sst.messageContent, height=800)
    else:
        st.markdown("State")
        st.markdown(sst.messageContent)
    mode_before = sst.editMode
    sst.editMode = st.toggle(label="Edit")
    if mode_before != sst.editMode:
        st.rerun()


st.header("Vision Creator")
init_session_state()
columns = st.columns([4, 1, 4], vertical_alignment="top")

with columns[0]:
    website_url = additional_information_view()
    st.divider()
    text_pdf = additional_pdf("Zusätzliche Dokumente")

with columns[2]:
    st.subheader("Prompting")
    prompt = prompting_view("./src/canned_prompts/")
    if st.button("Ok", disabled=str(prompt).strip() == ""):
        if website_url != sst.websiteURL:
            load_data_from_url(website_url)
            sst.websiteURL = website_url
        with st.spinner():
            sst.messageContent = make_request(prompt, sst.text, text_pdf)
        with st.container(border=True):
            st.subheader("Resultat")
            st.markdown(sst.messageContent)
