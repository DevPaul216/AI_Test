import streamlit as st
from streamlit import session_state as sst
from streamlit_extras.stylable_container import stylable_container

from streamlit_artifact_generation import artifact_generation_view, get_confirmed_artifacts


def init_session_state(artifact_dict):
    if "a_init" not in sst:
        sst.a_init = True
        sst.artifact_dict = artifact_dict

def set_input_dict(input_dict):
    sst.artifact_dict = input_dict

def get_input_dict():
    return sst.artifact_dict

def delete_artifact(category, artifact):
    available_values = sst.artifact_dict[category]
    sst.artifact_dict[category] = [value for value in available_values if value != artifact]
    print(category, artifact)



def display_artifacts_view():
    available_artifacts = sst.artifact_dict[sst.a_category_selected]
    st.markdown("**Existierende Artefakte**")
    if len(available_artifacts) == 0:
        st.write("Noch nichts vorhanden")
    for i, artifact in enumerate(available_artifacts):
        with st.container():
            columns = st.columns([1, 3, 1, 1], vertical_alignment="center")
            with columns[1]:
                st.markdown(artifact)
            with columns[2]:
                st.button(":x:", key=f"button{artifact}", on_click=delete_artifact,
                          kwargs={"category": sst.a_category_selected, "artifact": artifact})
        if i != len(available_artifacts) - 1:
            st.divider()


def display_template_view():
    # row1 = st.columns([3, 2, 1])
    # row2 = st.columns(3)
    target_elements_per_row = 2
    columns = st.columns(2)
    position = 0

    keys = list(sst.artifact_dict.keys())
    artifact_texts = {}
    max_characters = 0

    for key in keys:
        element_artifacts = sst.artifact_dict[key]
        artifact_text = ""
        for artifact in element_artifacts:
            artifact_text += "- " + artifact + "  \n"
        if len(artifact_text) > max_characters:
            max_characters = len(artifact_text)
        artifact_texts[key] = artifact_text

    estimated_height = int(max_characters * 0.8)
    estimated_height = max(200, estimated_height)
    for i in range(0, len(keys)):
        key = keys[i]
        column = columns[position]
        tile = column.container(height=estimated_height)
        # css_styles = """
        #        {
        #            backgorund-color: rgba(49, 51, 63, 0.2);
        #            border: 1px solid rgba(49, 51, 63, 0.2);
        #            border-radius: 0.5rem;
        #            padding: calc(1em - 1px)
        #        }
        #        """
        tile.subheader(key)
        sub_columns = tile.columns([1, 5, 1])
        artifact_text = artifact_texts[key]
        if len(artifact_text) > 0:
            text_to_show = artifact_text
        else:
            text_to_show = "Erstelle Objekte werden hier angezeigt"
        sub_columns[1].markdown(text_to_show)
        position += 1
        if position >= target_elements_per_row:
            position = 0
            columns = st.columns(target_elements_per_row)


def artifact_creation_view():
    selection = st.segmented_control(label="Creation type", options=["Manuell", "Generierung"])
    artifacts_to_add = []
    if selection == "Manuell":
        artifact_input = st.text_area(label="Artifact")
        artifact_input_clean = str(artifact_input).strip()
        if st.button("Hinzufügen"):
            artifacts_to_add = [artifact_input_clean]
    elif selection == "Generierung":
        artifact_generation_view()
        confirmed_artifacts = get_confirmed_artifacts()
        artifacts_to_add = list(confirmed_artifacts.values())
        for artifact_to_add in artifacts_to_add:
            if artifact_to_add not in sst.artifact_dict:
                sst.artifact_dict[sst.a_category_selected].append(artifact_to_add)
    else:
        st.write("Auswählen auf welche Weise Objekte hinzugefügt werden sollen")
    for artifact in artifacts_to_add:
        existing_artifacts = sst.artifact_dict[sst.a_category_selected]
        if artifact not in existing_artifacts:
            sst.artifact_dict[sst.a_category_selected].append(artifact)
        else:
            st.warning("Artefakt existiert bereits")
    return artifacts_to_add


def artifact_overview_view():
    # Select element
    with st.container(border=True):
        sst.a_category_selected = st.selectbox(label="Element", options=sst.artifact_dict.keys())
    with st.container(border=True):
        artifact_creation_view()
    with st.container(border=True):
        display_artifacts_view()
    st.write(sst.artifact_dict[sst.a_category_selected])

if __name__ == '__main__':
    test_artifact_dict = {"A": [], "B": [], "C": []}
    init_session_state(test_artifact_dict)
    st.set_page_config(page_title="Align Guide", layout="wide",
                       initial_sidebar_state="collapsed")
    artifact_overview_view()
    #display_template_view()
