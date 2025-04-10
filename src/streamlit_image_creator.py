import os
import json
import glob
import streamlit as st
from openai import OpenAI
from streamlit import session_state as sst

from shared_streamlit_views import prompting_view

st.set_page_config(page_title="Image Generator", layout="wide",
                   initial_sidebar_state="collapsed")

def init_session_state():
    if "init" not in sst:
        sst.init = True
        sst.prompt = ""
        sst.responses = []

st.header("Image Creator")
init_session_state()
sst.prompt = prompting_view("./src/canned_prompts/")
number_images = st.number_input(label="Number of images", value=1, step=1, max_value=4)
if st.button("Bestätigen", disabled=sst.prompt is None or str(sst.prompt).strip() == ""):
    with st.spinner("Processing"):
        with open("./src/config/keys.json") as f:
            config = json.load(f)
        openai_api_key = config["openai_api_key"]
        client = OpenAI(api_key=openai_api_key)
        sst.responses = []
        for i in range(number_images):
            response = client.images.generate(
                model="dall-e-3",
                prompt=sst.prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            sst.responses.append(response)
max_item = 2
position = 0
cols = columns = st.columns(2)
for response in sst.responses:
    if position >= max_item:
        cols = columns = st.columns(2)
        position = 0
    with cols[position]:
        st.image(response.data[0].url, caption=response.data[0].revised_prompt)
    position += 1
