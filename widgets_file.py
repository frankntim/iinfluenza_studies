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

    # Tailwind CSS injection
    display(HTML("""
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
      .tailwind-container {
        @apply flex flex-col items-center gap-4 p-4 bg-white bg-opacity-70 rounded-xl shadow-md w-full max-w-4xl mx-auto mt-6;
      }
      .no-cursor { cursor: default !important; }
    </style>
    """))

    # Background image
    if background_image_file and os.path.exists(background_image_file):
        with open(background_image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        mime = "image/png" if background_image_file.endswith(".png") else "image/jpeg"
        data_url = f"data:{mime};base64,{encoded}"

        display(HTML(f"""
        <style>
            body {{
                background-image: url('{data_url}');
                background-size: auto;
                background-repeat: repeat;
            }}
        </style>
        """))

    # Optional header image
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

    # Styled widgets
    top_button = widgets.Button(
        description="Survival Analysis",
        disabled=True,
        layout=widgets.Layout(width='90%', height='80px')
    )
    top_button.add_class("no-cursor")
    top_button.style.button_color = "#ADD8E6"

    title_box = widgets.Textarea(
        value='Survival - Cox-Mixed Model',
        disabled=True,
        layout=widgets.Layout(width='90%', height='80px')
    )
    title_box.add_class("no-cursor")

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

    output_area = widgets.Output(layout=widgets.Layout(border='1px solid #ccc', padding='10px', width='100%'))

    # Upload logic
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

    # Submit logic
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

    # Layout
    ui_elements = []
    if header_image_widget:
        ui_elements.append(header_image_widget)

    ui_elements += [
        top_button,
        title_box,
        upload_widget,
        chat_box,
        submit_button,
        output_area
    ]

    container = widgets.VBox(ui_elements)
    container.add_class("tailwind-container")

    display(container)
