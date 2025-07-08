# csv_chat_app.py

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output
from langchain.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import io
from types import SimpleNamespace

def launch_csv_chat_app():
    # Initialize shared state using SimpleNamespace
    state = SimpleNamespace()
    state.llm = ChatOpenAI(temperature=0, model="gpt-4")  # or gpt-3.5-turbo
    state.agent = None
    state.df = None

    # UI Widgets
    upload_widget = widgets.FileUpload(
        accept='.csv', multiple=False, description='Upload CSV'
    )
    chat_box = widgets.Textarea(
        value='',
        placeholder='Ask a question about your dataset...',
        description='Query:',
        layout=widgets.Layout(width='100%', height='100px')
    )
    submit_button = widgets.Button(
        description='Submit', button_style='success'
    )
    output_area = widgets.Output()

    # Upload handler
    def handle_upload(change):
        output_area.clear_output()
        if upload_widget.value:
            uploaded_filename = list(upload_widget.value.keys())[0]
            uploaded_bytes = upload_widget.value[uploaded_filename]['content']
            state.df = pd.read_csv(io.BytesIO(uploaded_bytes))

            with output_area:
                print(f"✅ Successfully loaded: {uploaded_filename}")
                display(state.df.head())

            # Create the agent
            state.agent = create_pandas_dataframe_agent(state.llm, state.df, verbose=False)

    upload_widget.observe(handle_upload, names='value')

    # Submit handler
    def on_submit_click(b):
        output_area.clear_output()

        if state.agent is None:
            with output_area:
                print("⚠️ Please upload a CSV file first.")
            return
        
        query = chat_box.value.strip()
        if not query:
            with output_area:
                print("⚠️ Please enter a query about the dataset.")
            return

        try:
            with output_area:
                print("🤖 Querying the dataset...\n")
                response = state.agent.run(query)
                print(response)
        except Exception as e:
            with output_area:
                print("❌ Error while processing:", str(e))

    submit_button.on_click(on_submit_click)

    # Display app
    display(widgets.VBox([
        upload_widget,
        chat_box,
        submit_button,
        output_area
    ]))
