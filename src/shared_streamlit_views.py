import glob
import os
import PyPDF2
import streamlit as st


def prompting_view(base_path):
    file_paths = glob.glob(f"{base_path}*.txt")
    file_names = [os.path.basename(file_path) for file_path in file_paths]streamlit run .\src\streamlit_modular.py
    all_options = ["Kein"] + file_names
    selected_file_name = st.selectbox(label="Preset Prompts", options=all_options, index=0)
    prompt_text = ""
    if selected_file_name is not None and selected_file_name != "Kein":
        with open(f'{base_path}{selected_file_name}', 'r', encoding='utf8') as file:
            # Read the entire content of the file
            prompt_text = file.read()
    estimated_space_needed = int(len(prompt_text) * 0.75)
    text_area_height = max(200, estimated_space_needed)
    return st.text_area(label="Prompt", height=text_area_height, value=prompt_text)


def additional_pdf(subheader):
    st.subheader(subheader)
    uploaded_pdf_files = st.file_uploader(label="PDF Dokumente hochladen", type="pdf", accept_multiple_files=True)
    text = ''
    if uploaded_pdf_files is not None and len(uploaded_pdf_files) > 0:
        reader = PyPDF2.PdfReader(uploaded_pdf_files[0])
        for page in reader.pages:
            text += page.extract_text()
    with st.expander("Auszug"):
        st.write(text[0:5000])
    return text
