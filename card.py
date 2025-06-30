# === Linear Mixed Model tool ===
from statsmodels.formula.api import mixedlm
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, ctx, callback
import dash_bootstrap_components as dbc
import dash
import pandas as pd

class LMMArgs(BaseModel):
    response: str
    time: str
    group: str
    subject_id: str

parser_lmm = PydanticOutputParser(pydantic_object=LMMArgs)
prompt_lmm = PromptTemplate(
    template="""Extract the response, time variable, group variable, and subject_id for a linear mixed effects model.

Query: {query}
{format_instructions}
""",
    input_variables=["query"],
    partial_variables={"format_instructions": parser_lmm.get_format_instructions()}
)
llm_chain_lmm = LLMChain(llm=ChatOpenAI(model="gpt-4", temperature=0), prompt=prompt_lmm)

def lmm_fit(query: str):
    try:
        args = parser_lmm.parse(llm_chain_lmm.run(query))
        y, t, g, subj = args.response, args.time, args.group, args.subject_id
        for col in [y, t, g, subj]:
            if col not in df.columns:
                return f"⚠️ Column `{col}` not found."
        dff = df[[y, t, g, subj]].dropna()
        dff["interaction"] = dff[t].astype(float) * dff[g].astype("category").cat.codes
        model = mixedlm(f"{y} ~ {t} + {g} + interaction", dff, groups=dff[subj])
        result = model.fit()
        summary_df = result.summary().tables[1]
        summary_df.columns = summary_df.columns.str.strip()
        streamed_table["html"] = summary_df.round(3).to_dict("records")
        streamed_table["columns"] = [{"name": col, "id": col} for col in summary_df.columns]

        # Plot only smoothed trend lines (fixed effects) by group with confidence intervals
        dff_sorted = dff.sort_values(by=[g, t])
        group_stats = dff_sorted.groupby([g, t])[y].agg(["mean", "sem"]).reset_index()
        group_stats["lower"] = group_stats["mean"] - 1.96 * group_stats["sem"]
        group_stats["upper"] = group_stats["mean"] + 1.96 * group_stats["sem"]

        fig = go.Figure()
        for grp in group_stats[g].unique():
            group_df = group_stats[group_stats[g] == grp]
            fig.add_trace(go.Scatter(x=group_df[t], y=group_df["mean"], mode='lines',
                                     name=f"Mean Trend - {grp}", line=dict(width=3)))
            fig.add_trace(go.Scatter(x=list(group_df[t]) + list(group_df[t])[::-1],
                                     y=list(group_df["upper"]) + list(group_df["lower"])[::-1],
                                     fill='toself', fillcolor='rgba(0,100,80,0.2)',
                                     line=dict(color='rgba(255,255,255,0)'),
                                     hoverinfo="skip", showlegend=False))

        fig.update_layout(title="Fixed Effect Trends with Confidence Intervals", xaxis_title=t, yaxis_title=y)
        streamed_plot["fig"] = fig

        return "Linear mixed model fitted. Summary and fixed effect trends shown below."
    except Exception as e:
        return f"❌ LMM Error: {e}"

# === Tools ===
tools = [
    Tool(name="ChartGenerator", func=plot_chart, description="Basic charts like age distribution, fare vs age, etc."),
    Tool(name="KaplanMeierPlot", func=km_survival_plot, description="Kaplan-Meier plots with time/event/group."),
    Tool(name="CoxPHFitter", func=coxph_plot, description="Cox regression with time/event/covariates."),
    Tool(name="LogRankTest", func=logrank_comparison, description="Compare two survival groups using the log-rank test."),
    Tool(name="LinearMixedModel", func=lmm_fit, description="Fit a linear mixed effects model with time, group, and subject ID.")
]

# === Dash Layout ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dcc.Store(id="selected-tool"),

    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardImg(src="/assets/general_analysis.png", top=True, style={"height": "180px", "objectFit": "cover"}),
                dbc.CardBody(html.H5("General Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", id="run-general", color="primary"))
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardImg(src="/assets/survival_analysis.png", top=True, style={"height": "180px", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Survival Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", id="run-survival", color="primary"))
            ]), width=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardImg(src="/assets/cox_fitting.png", top=True, style={"height": "180px", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Cox Fitting Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", id="run-cox", color="primary"))
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardImg(src="/assets/linear_mixed.png", top=True, style={"height": "180px", "objectFit": "cover"}),
                dbc.CardBody(html.H5("Linear Mixed Analysis", className="card-title")),
                dbc.CardFooter(dbc.Button("Run Analysis", id="run-lmm", color="primary"))
            ]), width=6),
        ])
    ]),

    dbc.Modal([
        dbc.ModalHeader("Analysis Agent"),
        dbc.ModalBody([
            dcc.Input(id="user-query", type="text", placeholder="Type your question here...", style={"width": "100%"}),
            html.Br(), html.Br(),
            dbc.Button("Submit", id="submit-analysis", color="success"),
            html.Hr(),
            html.Div(id="chat-log", style={"whiteSpace": "pre-wrap", "overflowY": "auto", "maxHeight": "300px"}),
            dcc.Graph(id="chart-output"),
            html.Div(id="table-output")
        ]),
        dbc.ModalFooter(dbc.Button("Close", id="close-modal", className="ml-auto"))
    ], id="chat-modal", is_open=False)
])

# === Callbacks ===
@app.callback(
    Output("selected-tool", "data"),
    Output("chat-modal", "is_open"),
    Input("run-general", "n_clicks"),
    Input("run-survival", "n_clicks"),
    Input("run-cox", "n_clicks"),
    Input("run-lmm", "n_clicks"),
    prevent_initial_call=True
)
def open_modal(general, survival, cox, lmm):
    if not ctx.triggered:
        return dash.no_update, False
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    tool_map = {
        "run-general": "ChartGenerator",
        "run-survival": "KaplanMeierPlot",
        "run-cox": "CoxPHFitter",
        "run-lmm": "LinearMixedModel"
    }
    return tool_map.get(button_id), True

@app.callback(
    Output("chat-modal", "is_open", allow_duplicate=True),
    Input("close-modal", "n_clicks"),
    prevent_initial_call=True
)
def close_modal(n):
    return False

@app.callback(
    Output("chat-log", "children"),
    Output("chart-output", "figure"),
    Output("table-output", "children"),
    Input("submit-analysis", "n_clicks"),
    State("selected-tool", "data"),
    State("user-query", "value"),
    prevent_initial_call=True
)
def run_tool(n, selected_tool, query):
    tool = next((t for t in tools if t.name == selected_tool), None)
    if tool is None:
        return "❌ Tool not found", {}, ""
    result = tool.func(query)
    fig = streamed_plot.get("fig", {})
    table_data = streamed_table.get("html", [])
    table_cols = streamed_table.get("columns", [])
    if table_data and table_cols:
        table = dbc.Table.from_dataframe(pd.DataFrame(table_data), striped=True, bordered=True, hover=True)
    else:
        table = ""
    return result, fig, table
