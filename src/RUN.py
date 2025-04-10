import os
import subprocess

def run_prompteditor():
    # 1) Your environment's python.exe
    python_path = r"C:\Users\pauls\anaconda3\envs\myenv\python.exe"
    # Dynamically determine the working directory
    working_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(working_directory, "src")

    try:
        print("Launching streamlit_modular.py...")
        subprocess.Popen(
            [
                python_path,
                "-m", "streamlit", "run",
                os.path.join(folder_path, "streamlit_modular.py")
            ],
            cwd=working_directory
        )
        print("streamlit_modular.py launched successfully.")
    except Exception as e:
        print(f"Error: Failed to launch streamlit_modular.py\n\n{e}")

if __name__ == "__main__":
    run_prompteditor()