"""
Databricks Catalog Browser (ipywidgets + LangChain‑OpenAI)
========================================================
An interactive Unity Catalog explorer for Databricks notebooks **enhanced with an LLM**: type a natural‑language question and the app converts it to Spark SQL on the fly using `langchain_openai`.

Quick start
-----------
```python
%pip install --quiet langchain_openai langchain_core --upgrade   # once per cluster, if needed
%run ./databricks_catalog_browser.py                             # path may vary

import databricks_catalog_browser as dcb

dcb.launch_catalog_browser(default_limit=50)
```
Features
--------
* Browse **catalog → schema → table** and multi‑select tables.
* **Ask questions in plain English** or paste raw SQL—LLM decides.
* Adjustable **row‑limit** (applied to generated SQL too).
* Results rendered inline.
* Soft‑blue app background to stand out in the notebook.

Requirements
------------
* Databricks Runtime 14.x +
* `ipywidgets` (bundled) and `langchain_openai` (install via `%pip`).
"""
from __future__ import annotations

import json
from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output
from pyspark.sql import SparkSession

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

__all__ = ["launch_catalog_browser"]

# -----------------------------------------------------------------------------
# Spark session helper
# -----------------------------------------------------------------------------

def _spark() -> SparkSession:
    return SparkSession.builder.getOrCreate()


# -----------------------------------------------------------------------------
# Metadata helpers
# -----------------------------------------------------------------------------

def _list_catalogs() -> List[str]:
    return [r.catalog_name for r in _spark().sql("SHOW CATALOGS").collect()]


def _list_schemas(catalog: str) -> List[str]:
    return [r.databaseName for r in _spark().sql(f"SHOW SCHEMAS IN {catalog}").collect()]


def _list_tables(catalog: str, schema: str) -> List[str]:
    rows = _spark().sql(f"SHOW TABLES IN {catalog}.{schema}").collect()
    return [r.tableName for r in rows]


def _table_schema_json(catalog: str, schema: str, table: str) -> dict:
    """Return minimal JSON schema for a table (name & columns).
    We include only column name + type to keep the prompt short.
    """
    df = _spark().sql(f"DESCRIBE TABLE {catalog}.{schema}.{table}")
    cols = [row.col_name + " " + row.data_type for row in df.collect() if row.col_name]
    return {"table": table, "columns": cols}


# -----------------------------------------------------------------------------
# LLM helper
# -----------------------------------------------------------------------------

def _nl_to_sql(question: str, catalog: str, schema: str, tables: List[str], row_limit: int) -> str:
    """Use ChatOpenAI (via LangChain) to translate NL → Spark SQL."""
    # Build schema manifest for context (truncated to keep prompt size moderate)
    tables_meta = [_table_schema_json(catalog, schema, t) for t in tables]
    manifest = json.dumps(tables_meta, indent=2)[:6000]  # safety cutoff

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are an expert Spark SQL assistant on Databricks. "
                "Given the user's question and the available tables (with schemas) in catalog '{catalog}.{schema}', "
                "write an executable Spark SQL query that answers the question. "
                "Return ONLY the SQL statement without Markdown fences. "
                "Ensure the query LIMITs the result to {row_limit} rows if no limit is specified."
            ),
        ),
        (
            "human",
            (
                "User question: {question}\n\n"
                "Available tables JSON:\n{manifest}"
            ),
        ),
    ])

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    sql: str = llm.invoke(prompt.format(question=question, manifest=manifest, catalog=catalog, schema=schema, row_limit=row_limit)).content.strip()
    # Add fallback limit if LLM forgets
    if " limit " not in sql.lower():
        sql += f" LIMIT {row_limit}"
    return sql


# -----------------------------------------------------------------------------
# UI builder
# -----------------------------------------------------------------------------

def _build_ui(default_limit: int = 20) -> widgets.Box:
    catalog_dd = widgets.Dropdown(options=_list_catalogs(), description="Catalog:", layout=widgets.Layout(width="300px"))
    schema_dd = widgets.Dropdown(options=[], description="Schema:", layout=widgets.Layout(width="300px"))
    table_sel = widgets.SelectMultiple(options=[], description="Tables:", rows=8, layout=widgets.Layout(width="300px"))
    limit_int = widgets.BoundedIntText(value=default_limit, min=1, max=10000, description="Rows:", layout=widgets.Layout(width="200px"))
    run_btn = widgets.Button(description="Run", button_style="success")
    sql_box = widgets.Textarea(
        placeholder="Ask in plain English OR paste SQL…",
        description="Prompt:",
        layout=widgets.Layout(width="600px", height="120px"),
    )
    out = widgets.Output()

    # --- callbacks ---
    def _on_catalog(change):
        catalog = change["new"]
        schema_dd.options = _list_schemas(catalog)
        if schema_dd.options:
            schema_dd.value = schema_dd.options[0]

    def _on_schema(change):
        tables = _list_tables(catalog_dd.value, change["new"])
        table_sel.options = tables

    def _run(_):
        with out:
            clear_output(wait=True)
            prompt_text = sql_box.value.strip()

            catalog, schema = catalog_dd.value, schema_dd.value
            selected_tables = list(table_sel.value) or _list_tables(catalog, schema)

            # Decide if user input looks like SQL (crude heuristic)
            is_sql = prompt_text.lower().lstrip().startswith(("select", "with", "describe", "show"))

            if not prompt_text:
                print("⚠️ Please ask a question or enter SQL.")
                return

            if not is_sql:
                # Natural language → SQL via LLM
                try:
                    sql_query = _nl_to_sql(prompt_text, catalog, schema, selected_tables, limit_int.value)
                    print("-- Generated SQL --\n", sql_query)
                except Exception as llm_err:
                    print("LLM failed to produce SQL:", llm_err)
                    return
            else:
                sql_query = prompt_text
                # Append limit if missing
                if " limit " not in sql_query.lower():
                    sql_query += f" LIMIT {limit_int.value}"

            try:
                pdf = _spark().sql(sql_query).toPandas()
                display(pdf)
            except Exception as db_err:
                print("Query failed:", db_err)

    # Wire events
    catalog_dd.observe(_on_catalog, names="value")
    schema_dd.observe(_on_schema, names="value")
    run_btn.on_click(_run)

    # Populate initial lists
    _on_catalog({"new": catalog_dd.value})

    # Layout with blue background
    container = widgets.VBox([
        widgets.HBox([catalog_dd, schema_dd]),
        widgets.HBox([table_sel, widgets.VBox([limit_int, run_btn])]),
        sql_box,
        out,
    ])
    styled = widgets.Box([
        container
    ], layout=widgets.Layout(border="solid 1px gray", padding="10px", background_color="#e6f0ff"))
    return styled


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def launch_catalog_browser(default_limit: int = 20) -> None:
    """Display the catalog browser (with LLM SQL) in the current notebook cell."""
    display(_build_ui(default_limit))


# -----------------------------------------------------------------------------
# Entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    launch_catalog_browser()
