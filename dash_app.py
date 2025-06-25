import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.llms import OpenAI  # Or your chosen LLM
import os
import json
import plotly.io as pio

# Load data (replace with your actual data loading)
data = {'Category': ['A', 'B', 'C', 'D'],
        'Value': [25, 40, 30, 55]}
df = pd.DataFrame(data)

# Initialize the Dash app
app = dash.Dash(__name__)

# Custom prefix for the LangChain agent
# This prefix guides the agent to produce bar chart instructions
custom_prefix = """
You are a data analysis assistant working with a pandas DataFrame named `df`.
Focus on generating bar charts when requested.
If a user asks for a bar chart, respond in a JSON format like this:
{{
  "bar_chart": {{
    "x_col": "column_for_x_axis",
    "y_col": "column_for_y_axis",
    "title": "Title of the Bar Chart"
  }}
}}
If the request is not for a bar chart, provide a concise text answer in a JSON format like this:
{{
  "answer": "Your text response here."
}}
"""

# Initialize the LangChain agent
# You'll need to provide your API key for the chosen LLM
llm = OpenAI(api_key="YOUR_OPENAI_API_KEY") # Replace with your LLM setup
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    prefix=custom_prefix,
    allow_dangerous_code=True # Be cautious with this setting; use in a secure environment
)

# Define the layout
app.layout = html.Div([
    html.H1("Data Analysis with AI Agent"),
    html.Div([
        html.Label("Enter your query:"),
        dcc.Input(id='query-input', type='text', value='Show a bar chart of Category vs Value'),
        html.Button('Ask Agent', id='ask-button', n_clicks=0),
    ]),
    html.Div(id='agent-response', children=''),
    dcc.Graph(id='bar-chart-graph', figure={})  # Area for the bar chart
])

# Define the callback to handle the agent interaction and graph update
@app.callback(
    [Output('agent-response', 'children'),
     Output('bar-chart-graph', 'figure')],
    Input('ask-button', 'n_clicks'),
    [State('query-input', 'value')]
)
def update_response_and_graph(n_clicks, query):
    if n_clicks > 0 and query:
        try:
            # Run the agent with the query
            response = agent.run(query)

            # Attempt to parse the agent's response as JSON
            response_json = json.loads(response)

            if "bar_chart" in response_json:
                # If the agent responded with bar chart instructions
                chart_info = response_json["bar_chart"]
                x_col = chart_info.get("x_col")
                y_col = chart_info.get("y_col")
                title = chart_info.get("title", "Bar Chart")

                if x_col and y_col and x_col in df.columns and y_col in df.columns:
                    # Create the Plotly bar chart
                    fig = px.bar(df, x=x_col, y=y_col, title=title)

                    # Return the response message and the figure
                    return f"Agent responded with a bar chart request. Title: {title}", fig
                else:
                    return "Agent requested a bar chart, but missing or invalid columns.", {}

            elif "answer" in response_json:
                # If the agent responded with a text answer
                return response_json["answer"], {}

            else:
                return "Agent response not in expected format.", {}

        except Exception as e:
            return f"Error processing agent response: {e}", {}

    return '', {} # Return empty response and figure if no query or button not clicked

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
