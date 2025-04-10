import streamlit as st
from streamlit_extras.stylable_container import stylable_container

container_css = """
        stMainBlockContainer { 
            background-color: yellow;
            border: True;
            color: yellow;
        }
        """
#st.markdown(container_css, unsafe_allow_html=True)

rows = [([1, 2], 400), ([2], 200)]
position = 0
vertical_gap = 7
for row in rows:
    number_cols = len(row[0])
    cols = st.columns(number_cols, vertical_alignment='center')
    sub_rows = row[0]
    for col, sub_row in zip(cols, sub_rows):
            with col:
                height_single = int(row[1] / sub_row) - (sub_row - 1) * vertical_gap
                for number_subrows in range(0, sub_row):
                    with st.container(border=True, height=height_single):
                        with stylable_container(key="sc_" + str(position), css_styles=container_css):
                            container = st.container(border=False)
                            sub_columns = container.columns([2, 2], vertical_alignment='center')
                                #if st.button("Submit"):
                                #    st.write("Hi", key=str(position))
                                #st.text_area(key="ta" + str(position), label=str(position))
                            with sub_columns[1]:
                                st.title(":balloon:")
                                position += 1

