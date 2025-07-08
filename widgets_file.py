# csv_chat_app.py

import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from langchain.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import io
from types import SimpleNamespace
import base64
import os

def launch_csv_chat_app(background_image_file: str = None, header_image_file: str = None):
    # Shared state
    state = SimpleNamespace()
    state.llm = ChatOpenAI(temperature=0, model="gpt-4")
    state.agent = None
    state.df = None

    # Background image (if provided)
    if background_image_file and os.path.exists(background_image_file):
        with open(background_image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if background_image_file.endswith(".png") else "image/jpeg"
        data_url = f"data:{mime};base64,{encoded}"

        display(HTML(f"""
            <style>
                .widget-app-container {{
                    background-image: url('{data_url}');
                    background-repeat: repeat;
                    background-size: auto;
                    padding: 20px;
                    border-radius: 10px;
                }}
                .widget-app-container textarea {{
                    font-family: monospace;
                }}
            </style>
        """))

    # Header image (if provided)
    header_image_widget = None
    if header_image_file and os.path.exists(header_image_file):
        with open(header_image_file, "rb") as f:
            header_encoded = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if header_image_file.endswith(".png") else "image/jpeg"
        header_image_widget = widgets.Image(
            value=base64.b64decode(header_encoded),
            format=mime.split("/")[-1],
            layout=widgets.Layout(width='90%', height='80px')
        )

    # Title box (styled)
    title_box = widgets.Textarea(
        value='Survival - Cox-Mixed Model',
        disabled=True,
        layout=widgets.Layout(width='90%', height='80px'),
        style={'description_width': 'initial'}
    )
    title_box.add_class("title-box-style")

    # Title styling
    display(HTML("""
        <style>
            .title-box-style textarea {
                background-color: gray !important;
                color: white !important;
                font-weight: bold;
                font-size: 16px;
                border: none;
                padding: 10px;
                resize: none;
            }
        </style>
    """))

    # Widgets
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
    output_area = widgets.Output(layout=widgets.Layout(border='1px solid gray', padding='10px'))

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

            state.agent = create_pandas_dataframe_agent(
                state.llm,
                state.df,
                verbose=False,
                handle_parsing_errors=True
            )

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

    # Build UI
    elements = []
    if header_image_widget:
        elements.append(header_image_widget)
    elements += [
        title_box,
        upload_widget,
        chat_box,
        submit_button,
        output_area
    ]
    container = widgets.VBox(elements)

    if background_image_file:
        container.add_class("widget-app-container")

    display(container)
