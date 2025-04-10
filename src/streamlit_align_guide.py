import glob
import streamlit as st
from streamlit import session_state as sst
from enum import Enum

from shared_streamlit_views import additional_pdf
from utils import make_request, load_prompt

st.set_page_config(page_title="Align Guide", layout="wide",
                   initial_sidebar_state="collapsed")

class ViewState(Enum):
    Busy = 0
    Ready = 1
    Skip = 2

def init_session_state():
    if "init" not in sst:
        sst.init = True
        sst.position = 0
        sst.shared_data = {}
        sst.editMode = False
        sst.messageContent = ""
        sst.text = ""
        sst.imagePaths = []
        sst.websiteURL = ""
        sst.response_message = None


def vertical_space(count):
    for i in range(0, count):
        st.write("")


def response_subview(enable_editing=True):
    st.divider()
    response_sub_view()
    if enable_editing:
        st.divider()
        edit_sub_view()


def welcome_screen_view():
    vertical_space(5)
    columns = st.columns([1, 3, 1])
    with columns[1]:
        st.header("Willkommen")
        welcome_text = "Welcome to this research prototype. Do you accept the terms and want to continue…"
        st.markdown(f'<span style="font-size: 24px;">{welcome_text}</span>', unsafe_allow_html=True)
    return ViewState.Ready


def innovation_issue_view():
    for i in range(0, 4):
        st.write("")
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("What is the innovation issue you are trying to solve today?")
        innovation_issue = columns[1].text_area(label="Innovation Issue", height=200)
        sst.shared_data["innovation_issue"] = innovation_issue
        welcome_text = """Try to be as general as possible yet as specific as necessary  
You can try to describe:  
- Problem a person has (e.g. Kitchen tools take up to much space) 
- Business related issue (My existing products need to generate more monthly cash flow) 
- Larger scale problem (a lot of pets are bored when home alone)
"""
        st.markdown(f'<span style="font-size: 20px;">{welcome_text}</span>', unsafe_allow_html=True)
    if innovation_issue is not None and str(innovation_issue).strip() != "":
        return ViewState.Ready
    else:
        return ViewState.Busy


def target_demographic_view():
    for i in range(0, 4):
        st.write("")
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("Who is affected by the issue you are trying to solve today?")
        existing_value = ""
        if 'target_demographic' in sst.shared_data:
            existing_value = sst.shared_data['target_demographic']
        target_demographic = st.text_area(label="What is your Target Demographic?", value=existing_value, height=200)
        sst.shared_data['target_demographic'] = target_demographic
        st.write("Or generate suggestions if you are unsure")
        if st.button("Generate suggestions", type="secondary"):
            prompt_text = load_prompt("function_01")
            #st.write(prompt_text)
            if prompt_text is not None:
                with st.spinner():
                    innovation_issue = sst.shared_data['innovation_issue']
                    sst.response_message = make_request(prompt_text, [innovation_issue])
                    #sst.shared_data['target_demographic'] = sst.response_message
        if sst.response_message is not None:
            response_subview(enable_editing=False)
        if str(target_demographic).strip() != "":
            return ViewState.Ready
    return ViewState.Busy


def resource_type_view():
    vertical_space(3)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header(
            "Are you trying to build a new solution for your issue from scratch or do you have some existing solutions or resources you want to utilize?")
        vertical_space(2)
        button_name_first = "I want to start from scratch"
        button_name_second = "I have a short document describing my companies resources"
        button_name_three = "I want to enter the resources on my own"
        button_names = [button_name_first, button_name_second, button_name_three]
        selection_names = ["scratch", "document", "own"]
        button_id = None
        if st.button(button_names[0], type="secondary"):
            button_id = 0
        if st.button(button_names[1], type="secondary"):
            button_id = 1
        if st.button(button_names[2], type="secondary"):
            button_id = 2
        if button_id is not None:
            button_name = button_names[button_id]
            selection = selection_names[button_id]
            st.caption(f"You selected **{button_name}**")
            sst.shared_data['resource_type'] = selection
            return ViewState.Ready
    return ViewState.Busy


def resource_gathering_view():
    vertical_space(4)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        resource_type = sst.shared_data['resource_type']
        if resource_type == "scratch":
            return ViewState.Skip
        if resource_type == "document":
            st.header("Please check whether you agree with this or edit it accordingly")
            text_pdf = additional_pdf("Upload your business documents")
            if st.button("Generieren", type="secondary", disabled=text_pdf == ''):
                with st.spinner():
                    prompt_text = load_prompt("function_02")
                    if prompt_text is not None:
                        sst.response_message = make_request(prompt_text, [text_pdf])
                    else:
                        st.error("Could not load prompt")
            if sst.response_message is not None:
                response_subview()
                return ViewState.Ready
        elif resource_type == "own":
            st.header("Please enter your organizations resources below")
            resource_names = ["Financial Resources",
                              "Human Resources",
                              "Intellectual Property",
                              "Physical Resources",
                              "Specific Knowledge",
                              "Market Position"]
            resources_dict = {}
            for resource_name in resource_names:
                value = st.text_area(resource_name, height=200, key=resource_name)
                resources_dict[resource_name] = value
            sst.shared_data['resources_dict'] = resources_dict
            return ViewState.Ready
    return ViewState.Busy


def design_challenge_view():
    vertical_space(4)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("Design Challenge")
        info_text = "Based on your innovation issue and the target demographic several “How-Might-We” questions are formulated"
        columns[1].markdown(f'<span style="font-size: 24px;">{info_text}</span>', unsafe_allow_html=True)
        vertical_space(2)
        if st.button("Generate 'How-Might-We' Questions", type="secondary"):
            with st.spinner():
                prompt_text = load_prompt("function_03")
                if prompt_text is not None:
                    innovation_issue = sst.shared_data['innovation_issue']
                    target_demographic = sst.shared_data['target_demographic']
                    sst.response_message = make_request(prompt_text, [innovation_issue, target_demographic])
                else:
                    st.error("Could not load prompt")
        if sst.response_message is not None:
            response_subview()
            sst.shared_data["HowMightWeQuestions"] = sst.response_message
            return ViewState.Ready
    return ViewState.Busy


def additional_resources_subview():
    sub_colums = st.columns([1, 1], vertical_alignment="center")
    with sub_colums[0]:
        website_url = st.text_input(label="Website URL", value=sst.websiteURL)
    with sub_colums[1]:
        uploaded_files = st.file_uploader(label="Upload pdf files", type=["pdf"])
    return website_url, uploaded_files


def mi_vi_va_view():
    vertical_space(4)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("Mission / Vision / Values")
        info_text = """We need to know more about your task in order to help us understand you problem from the perspective of your personal or business perspective. 
        Please tell us about what you are trying to achieve, how you plan to do it, and what principles you strive by."""
        columns[1].markdown(f'<span style="font-size: 24px;">{info_text}</span>', unsafe_allow_html=True)
        vertical_space(1)
        info_text = """If you already have documents or a website describing that, we can find out on our own. 
        If you don’t, you need to enter the information manually."""
        columns[1].markdown(f'<span style="font-size: 24px;">{info_text}</span>', unsafe_allow_html=True)
        vertical_space(2)
        resource_names = ["Mission", "Vision", "Values"]
        message_text = None
        selection = st.pills(label="Select", options=["Yes, please generate it", "I want to add them manually"],
                             selection_mode="single")
        if selection == "Yes, please generate it":
            st.write("Before generating you can add additional information to use")
            website_url, uploaded_files = additional_resources_subview()
            if st.button("Generate"):
                with st.spinner():
                    function_filenames = ["function_04_mission", "function_04_vision", "function_04_goals"]
                    for resource_name, filename in zip(resource_names, function_filenames):
                        prompt_text = load_prompt(filename)
                        if prompt_text is not None:
                            innovation_issue = sst.shared_data['innovation_issue']
                            target_demographic = sst.shared_data['target_demographic']
                            sst.response_message = make_request(prompt_text, [innovation_issue, target_demographic])
                            sst.shared_data[resource_name] = sst.response_message
                        else:
                            st.error("Could not load prompt")
                message_text = "Mission / Vision / Values were generated. Move forward to view and possible edit them"
                return ViewState.Skip
        if selection == "I want to add them manually":
            for resource_name in resource_names:
                sst.shared_data[resource_name] = None
            message_text = "You selected **I want to add them manually**. Move forward to enter the values"
            return ViewState.Skip
        if message_text is not None:
            st.caption(message_text)
        if "Mission" in sst.shared_data:
            return ViewState.Ready
    return ViewState.Busy


def mi_vi_va_edit_view():
    vertical_space(4)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("Mission / Vision / Values")
        info_text = """Please enter your Mission / Vision / Values below or edit them if they were generated"""
        columns[1].markdown(f'<span style="font-size: 24px;">{info_text}</span>', unsafe_allow_html=True)
        vertical_space(1)
        resources_dict = {}
        resource_names = ["Mission", "Vision", "Values"]
        for resource_name in resource_names:
            value = ""
            if resource_name in sst.shared_data:
                value = sst.shared_data[resource_name]
            value = st.text_area(resource_name, height=200, key=resource_name, value=value)
            resources_dict[resource_name] = value
    return True


def align_overview_view():
    vertical_space(4)
    columns = st.columns([1, 5, 1])
    with columns[1]:
        st.header("Overview")
        info_text = """These are the info’s we have about you and your endeavor. 
        We will continue with those for discovering your issue in depth!"""
        columns[1].markdown(f'<span style="font-size: 24px;">{info_text}</span>', unsafe_allow_html=True)
        with st.container(border=True):
            for key, value in sst.shared_data.items():
                summary = f"""**{key}**  \n{value}"""
                st.write(summary)
    return ViewState.Busy


def edit_sub_view():
    edit_enabled = st.toggle(label="Editieren")
    if edit_enabled:
        estimated_vertical_space = int(len(sst.response_message) * 0.4)
        estimated_vertical_space = max(70, estimated_vertical_space)
        sst.response_message = st.text_area(label="Edited Response", value=sst.response_message,
                                            height=estimated_vertical_space)
    else:
        st.caption("Edited Response")
        st.markdown(sst.response_message, unsafe_allow_html=True)


def response_sub_view():
    st.subheader("Response")
    st.markdown(sst.response_message, unsafe_allow_html=True)
    st.caption("How satisfied are you with the response?")
    feedback = st.feedback("stars")
    if feedback is not None:
        st.write(feedback + 1)


def navigation_view(view_state):
    vertical_space(2)
    columns = st.columns([3, 1, 3])
    #with columns[1]:
    #    if st.button("Back", type="primary", disabled=sst.position == 0):
    #        sst.position -= 1
    #        st.rerun()
    with columns[1]:
        if st.button("Forward", type="primary", disabled=sst.position == len(views) - 1 or view_state==ViewState.Busy) or (view_state == ViewState.Skip):
            sst.position += 1
            sst.response_message = None
            st.rerun()


if __name__ == "__main__":
    init_session_state()
    views = [welcome_screen_view, innovation_issue_view, target_demographic_view, resource_type_view,
             resource_gathering_view, design_challenge_view, mi_vi_va_view, mi_vi_va_edit_view, align_overview_view]
    current_view = views[sst.position]
    view_state = current_view()
    navigation_view(view_state)
