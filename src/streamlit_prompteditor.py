import streamlit as st
import os


def prompt_editor_view(folder_path):
    st.title("Prompt Editor")

    # Get list of .txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    selected_file = st.session_state.get("selected_file", None)

    col1, col2 = st.columns([2, 3])

    

    st.markdown("""
        <style>
        div[role="radiogroup"] {
            overflow-y: auto;
            max-height: 400px;
        }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        st.subheader("Available Prompts")

        if txt_files:
            selected_file = st.radio("Select a file to edit", txt_files, index=txt_files.index(selected_file) if selected_file in txt_files else 0)
            st.session_state.selected_file = selected_file
        else:
            st.info("No prompt files found.")

        # --- Create new file ---
        new_name = st.text_input("New prompt name (.txt)", key="new_file")
        if st.button("Create"):
            if not new_name.endswith(".txt"):
                st.error("File name must end with .txt")
            elif not new_name.strip():
                st.error("Please enter a file name.")
            else:
                path = os.path.join(folder_path, new_name)
                if os.path.exists(path):
                    st.error("File already exists.")
                else:
                    with open(path, "w", encoding="utf-8"):
                        pass
                    st.success(f"Created {new_name}")
                    st.session_state.selected_file = new_name
                    st.rerun()

        # --- Rename file ---
        if selected_file:
            new_name = st.text_input("Rename to", value=selected_file, key="rename")
            if st.button("Rename"):
                if new_name == selected_file:
                    st.warning("New name is the same.")
                elif not new_name.endswith(".txt"):
                    st.error("File name must end with .txt")
                elif os.path.exists(os.path.join(folder_path, new_name)):
                    st.error("A file with that name already exists.")
                else:
                    os.rename(os.path.join(folder_path, selected_file), os.path.join(folder_path, new_name))
                    st.success(f"Renamed to {new_name}")
                    st.session_state.selected_file = new_name
                    st.rerun()

            # --- Delete file ---
            if st.button("Delete"):
                os.remove(os.path.join(folder_path, selected_file))
                st.success(f"Deleted {selected_file}")
                st.session_state.selected_file = None
                st.rerun()

    with col2:
        # Show text editor for selected file
        if selected_file:
            full_path = os.path.join(folder_path, selected_file)
            if os.path.exists(full_path):
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()

                edited = st.text_area("Edit Prompt", value=content, height=800)
                if st.button("Save"):
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(edited)
                    st.success(f"Saved changes to {selected_file}")
            else:
                st.warning(f"{selected_file} no longer exists.")
