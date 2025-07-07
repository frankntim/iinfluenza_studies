# Cell 1: Imports
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

from langchain.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import os


# Cell 2: Environment setup
# Set your OpenAI API key (you can set this securely in your environment instead)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # replace with your actual key


# Cell 3: File upload + agent integration
upload = widgets.FileUpload(
    accept='.csv',
    multiple=False,
    description="Upload CSV"
)

query_box = widgets.Textarea(
    placeholder='Ask a question about the dataset...',
    description='Query:',
    layout=widgets.Layout(width='100%', height='100px')
)

submit_button = widgets.Button(description="Submit")
output = widgets.Output()

# Global variables
df = None
agent = None

def handle_upload(change):
    global df, agent
    clear_output(wait=True)
    uploaded_file = list(upload.value.values())[0]
    content = uploaded_file['content']
    df = pd.read_csv(pd.io.common.BytesIO(content))
    print("✅ CSV loaded successfully. Shape:", df.shape)

    # Display the first few rows
    display(df.head())

    # Create the LLM agent
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    agent = create_pandas_dataframe_agent(llm, df, verbose=False)

    # Show widgets again
    display(upload, query_box, submit_button, output)

upload.observe(handle_upload, names='value')
display(upload)


# Cell 4: LLM chat with dataset
def on_submit_clicked(b):
    with output:
        clear_output()
        user_query = query_box.value.strip()
        if df is None or agent is None:
            print("⚠️ Please upload a dataset first.")
            return
        if not user_query:
            print("⚠️ Please enter a query.")
            return
        try:
            print(f"📩 You asked: {user_query}\n")
            response = agent.run(user_query)
            print(f"🤖 Response:\n{response}")
        except Exception as e:
            print("❌ Error:", str(e))

submit_button.on_click(on_submit_clicked)
display(query_box, submit_button, output)
