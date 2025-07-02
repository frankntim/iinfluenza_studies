from lifelines.statistics import logrank_test

class LogRankArgs(BaseModel):
    time_column: str
    event_column: str
    group_column: str

def logrank_comparison(query: str):
    try:
        from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model="gpt-4", temperature=0)

extract_prompt_logrank = PromptTemplate(
    template="""From the query below, extract and return a JSON with the keys: time_column, event_column, group_column.
Only use column names from this dataset: {columns}
Query: {query}
""",
    input_variables=["columns", "query"]
)
llm_chain_logrank = LLMChain(llm=llm, prompt=extract_prompt_logrank)

columns = list(df.columns)
llm_response = llm_chain_logrank.invoke({"columns": ", ".join(columns), "query": query})
import json
json_response = llm_response if isinstance(llm_response, dict) else json.loads(llm_response)
        instruction = f"""
        From this dataset, extract the time_column, event_column, and group_column for log-rank test.
        User Query: {query}
        Respond with a JSON object like this:
        {{"time_column": ..., "event_column": ..., "group_column": ...}}
        """
        import json
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            args = LogRankArgs(**json.loads(json_response))
        else:
            args = LogRankArgs(**json_response)

        T_col = args.time_column
        E_col = args.event_column
        G_col = args.group_column

        if G_col not in df.columns:
            return f"⚠️ Group column `{G_col}` not found."

        unique_groups = df[G_col].dropna().unique()
        if len(unique_groups) != 2:
            return f"⚠️ Log-rank test requires exactly 2 groups, found {len(unique_groups)}."

        df1 = df[df[G_col] == unique_groups[0]]
        df2 = df[df[G_col] == unique_groups[1]]

        result = logrank_test(df1[T_col], df2[T_col], event_observed_A=df1[E_col], event_observed_B=df2[E_col])
        p_val = result.p_value
        return f"Log-rank test between {unique_groups[0]} and {unique_groups[1]}: p = {p_val:.4f}"
    except Exception as e:
        return f"❌ Log-rank Error: {e}"

# === Linear Mixed Model Tool using LLM via DataFrame Agent ===
import statsmodels.formula.api as smf

class LMMArgs(BaseModel):
    response: str
    time_column: str
    group_column: str
    subject_column: str

def lmm_plot(query: str):
    try:
        extract_prompt_lmm = PromptTemplate(
    template="""From the query below, extract and return a JSON with the keys: response, time_column, group_column, subject_column.
Only use column names from this dataset: {columns}
Query: {query}
""",
    input_variables=["columns", "query"]
)
llm_chain_lmm = LLMChain(llm=llm, prompt=extract_prompt_lmm)

columns = list(df.columns)
llm_response = llm_chain_lmm.invoke({"columns": ", ".join(columns), "query": query})
import json
json_response = llm_response if isinstance(llm_response, dict) else json.loads(llm_response)
        instruction = f"""
        From this dataset, extract the response, time_column, group_column, and subject_column for linear mixed model analysis.
        User Query: {query}
        Respond with a JSON object like this:
        {{"response": ..., "time_column": ..., "group_column": ..., "subject_column": ...}}
        """
        import json
        json_response = llm_agent.invoke(instruction)
        if isinstance(json_response, str):
            args = LMMArgs(**json.loads(json_response))
        else:
            args = LMMArgs(**json_response)

        y, time, group, subject = args.response, args.time_column, args.group_column, args.subject_column

        required_cols = [y, time, group, subject]
        if not all(col in df.columns for col in required_cols):
            return f"⚠️ Missing one or more columns: {required_cols}"

        df_clean = df[[y, time, group, subject]].dropna()
        df_clean[group] = df_clean[group].astype(str)

        formula = f"{y} ~ {time} * {group}"
        model = smf.mixedlm(formula, df_clean, groups=df_clean[subject])
        result = model.fit()

        pred_df = df_clean.copy()
        pred_df["predicted"] = result.predict()

        fig = px.line(pred_df, x=time, y="predicted", color=group, title="Smoothed Group-Level Trends with LMM")
        streamed_plot["fig"] = fig

        html_table = html.Div([
            html.H6("📊 LMM Fixed Effects Table"),
            dbc.Table.from_dataframe(result.summary().tables[1], striped=True, bordered=True, hover=True, className="table-sm")
        ])
        streamed_table["html"] = html_table
        return f"Linear mixed model fitted with {y} as response."
    except Exception as e:
        return f"❌ LMM Error: {e}"