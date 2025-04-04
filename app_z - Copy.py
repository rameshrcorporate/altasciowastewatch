import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import os
from sqlalchemy import text
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from sqlalchemy import create_engine

POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL)

SHOW_TECHNICAL_DETAILS = False

def estimate_tokens(text, model="gpt-4-1106-preview"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def get_table_columns(engine, table_name="food_waste"):
    query = text(f"""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = '{table_name}';
    """)
    with engine.connect() as conn:
        result = conn.execute(query)
        columns = [row[0] for row in result.fetchall()]
    return columns

# def generate_required_column_patcher(df: pd.DataFrame):
    # related_columns = []

    # columns_lower = [col.lower() for col in df.columns]
    # if "weight" in columns_lower and "unit" in columns_lower:
        # related_columns.append(("weight", "unit"))
    # if "cost" in columns_lower and "currency" in columns_lower:
        # related_columns.append(("cost", "currency"))

    # def patch_sql(sql: str) -> str:
        # sql_lower = sql.lower()
        # select_index = sql_lower.find("select")
        # from_index = sql_lower.find("from")

        # if select_index == -1 or from_index == -1:
            # return sql  # invalid or non-standard SQL

        # select_clause = sql[select_index + 6:from_index].strip()

        # patched = False
        # additional_fields = []

        # for main_col, required_col in related_columns:
            # if main_col in sql_lower and required_col not in sql_lower:
                # additional_fields.append(required_col)
                # patched = True

        # if patched:
            # new_select_clause = ", ".join(additional_fields + [select_clause])
            # return f"SELECT {new_select_clause} {sql[from_index:]}"
        # return sql

    # return patch_sql
    
import re

def generate_required_column_patcher(df: pd.DataFrame):
    related_columns = []

    columns_lower_map = {col.lower(): col for col in df.columns}

    if "weight" in columns_lower_map and "unit" in columns_lower_map:
        related_columns.append(("weight", "unit"))
    if "cost" in columns_lower_map and "currency" in columns_lower_map:
        related_columns.append(("cost", "currency"))

    def patch_sql(sql: str) -> str:
        sql_lower = sql.lower()
        select_index = sql_lower.find("select")
        from_index = sql_lower.find("from")

        if select_index == -1 or from_index == -1:
            return sql  # invalid or non-standard SQL

        select_clause = sql[select_index + 6:from_index].strip()
        patched = False
        additional_fields = []

        # Match GROUP BY clause content
        groupby_match = re.search(r"group by (.+?)( order by| limit|;|$)", sql_lower)
        if groupby_match:
            groupby_raw = groupby_match.group(1)
            groupby_columns = [col.strip().strip('"') for col in groupby_raw.split(",")]
        else:
            groupby_columns = []

        for main_col, required_col in related_columns:
            if main_col in sql_lower and required_col not in sql_lower:
                required_col_real = columns_lower_map.get(required_col, required_col)
                required_col_quoted = f'"{required_col_real}"'
                additional_fields.append(required_col_quoted)

                if required_col_real not in groupby_columns:
                    groupby_columns.append(required_col_real)

                patched = True

        if patched:
            # Update SELECT clause
            new_select_clause = ", ".join(additional_fields + [select_clause])
            patched_sql = f"SELECT {new_select_clause} {sql[from_index:]}"

            # Update GROUP BY clause if it exists
            if groupby_match:
                new_groupby_clause = ", ".join(f'"{columns_lower_map.get(col.lower(), col)}"' for col in groupby_columns)
                patched_sql = re.sub(r"group by .+?( order by| limit|;|$)",
                                     f"GROUP BY {new_groupby_clause}\\1", patched_sql, flags=re.IGNORECASE)

            return patched_sql

        return sql

    return patch_sql





def generate_column_descriptions(df: pd.DataFrame) -> str:
    sample_values = {
        col: df[col].dropna().unique()[:3].tolist()
        for col in df.columns
    }

    descriptions = []
    columns_lower = [c.lower() for c in df.columns]

    for col in df.columns:
        col_lower = col.lower()

        if col_lower == "weight" and "unit" in columns_lower:
            descriptions.append(f"- '{col}': Quantity of food waste, measured in 'Unit'.")
        elif col_lower == "unit":
            descriptions.append(f"- '{col}': Measurement unit (e.g., kg, grams).")
        elif col_lower == "cost" and "currency" in columns_lower:
            descriptions.append(f"- '{col}': Monetary value of waste, expressed in 'Currency'.")
        elif col_lower == "currency":
            descriptions.append(f"- '{col}': Currency used for cost values (e.g., AED, USD).")
        elif col_lower == "loss reason":
            descriptions.append(f"- '{col}': Reason why the food was wasted.")
        elif col_lower == "disposition":
            descriptions.append(f"- '{col}': How the waste was handled (e.g., composted, disposed).")
        elif col_lower == "food item":
            descriptions.append(f"- '{col}': Specific food item being wasted.")
        elif col_lower == "food item category":
            descriptions.append(f"- '{col}': Group/category of the food item.")
        elif col_lower == "operator":
            descriptions.append(f"- '{col}': Person or staff member who logged the entry.")
        elif col_lower == "site":
            descriptions.append(f"- '{col}': Facility or location where waste occurred.")
        elif col_lower == "location":
            descriptions.append(f"- '{col}': Sub-location within the site.")
        elif col_lower == "device":
            descriptions.append(f"- '{col}': Device used to capture the entry.")
        elif col_lower == "date":
            descriptions.append(f"- '{col}': Date of the food waste entry.")
        elif col_lower == "time":
            descriptions.append(f"- '{col}': Time of the food waste entry.")
        elif col_lower == "region":
            descriptions.append(f"- '{col}': Geographic region.")
        elif col_lower == "country":
            descriptions.append(f"- '{col}': Country code or name.")
        elif col_lower == "stage of processing":
            descriptions.append(f"- '{col}': Phase of food processing (e.g., Pre-Consumer).")
        elif col_lower == "segment":
            descriptions.append(f"- '{col}': Business segment (e.g., Healthcare, Energy).")
        elif col_lower == "edited":
            descriptions.append(f"- '{col}': Indicates if the entry was edited (1 = yes, 0 = no).")
        elif col_lower == "quantity":
            descriptions.append(f"- '{col}': Number of items or units involved.")
        elif col_lower == "duration":
            descriptions.append(f"- '{col}': Duration (e.g., in seconds) possibly related to disposal or operation.")
        elif col_lower == "portions":
            descriptions.append(f"- '{col}': Number of food portions affected (if available).")
        elif col_lower == "source":
            descriptions.append(f"- '{col}': Source location or process where the waste came from.")
        elif col_lower == "container":
            descriptions.append(f"- '{col}': Type of container used (e.g., No Pan).")
        elif col_lower == "client location id":
            descriptions.append(f"- '{col}': Client-specific ID for the location.")
        elif col_lower == "client food item id":
            descriptions.append(f"- '{col}': Client-specific ID for the food item.")
        elif col_lower == "event order no.":
            descriptions.append(f"- '{col}': Identifier used for tracking specific disposal events.")
        else:
            # Generic fallback using sample values
            values = sample_values.get(col, [])
            if values:
                sample_snippet = ", ".join(map(str, values))
                descriptions.append(f"- '{col}': Sample values include {sample_snippet}.")
            else:
                descriptions.append(f"- '{col}': Column in the dataset (no sample values available).")

    return "\n".join(descriptions)


    
def extract_filters_from_question(question: str, table_columns: list):
    filters = {}

    for col in table_columns:
        col_lower = col.lower().replace("_", " ")
        match = re.search(rf"{col_lower}\s*[:=]?\s*([\w\s\-]+)", question, re.I)
        if match:
            value = match.group(1).strip()
            filters[col] = value

    # Handle common phrases like "last month"
    if "last month" in question.lower():
        today = datetime.today()
        first_day = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day = today.replace(day=1) - timedelta(days=1)
        filters["Date"] = (first_day.strftime("%Y-%m-%d"), last_day.strftime("%Y-%m-%d"))

    return filters


def build_sql_query(filters: dict, limit=100):
    base = 'SELECT * FROM "food_waste"'
    conditions = []

    for column, value in filters.items():
        if column == "Date" and isinstance(value, tuple):  # date range
            start, end = value
            conditions.append(f'"{column}" BETWEEN \'{start}\' AND \'{end}\'')
        else:
            conditions.append(f'"{column}" = \'{value}\'')

    if conditions:
        base += " WHERE " + " AND ".join(conditions)

    base += f' ORDER BY "Date" DESC LIMIT {limit};'
    return base

def generate_sql_from_question(question: str, table_columns: list) -> str:
    schema_str = ", ".join([f'"{col}"' for col in table_columns])
    
    sql_prompt = f"""
You are a helpful data analyst. The table is named "food_waste" and contains these columns:
{schema_str}

Write a PostgreSQL query (SELECT ...) to answer this question:
"{question}"

Do not explain the query. Only return the raw SQL statement. Do not wrap it in markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You write SQL queries based on user questions."},
            {"role": "user", "content": sql_prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )

    # Strip code blocks or backticks if GPT added them
    sql_raw = response.choices[0].message.content.strip()
    sql_clean = sql_raw.strip("`").replace("```sql", "").replace("```", "").strip()
    return sql_clean
    
from openai import OpenAI
import pandas as pd
from sqlalchemy import text

def generate_column_descriptions(df: pd.DataFrame) -> str:
    sample_values = {
        col: df[col].dropna().unique()[:3].tolist()
        for col in df.columns
    }

    descriptions = []
    columns_lower = [c.lower() for c in df.columns]

    for col in df.columns:
        col_lower = col.lower()

        if col_lower == "weight" and "unit" in columns_lower:
            descriptions.append(f"- '{col}': Quantity of food waste, measured in 'Unit'.")
        elif col_lower == "unit":
            descriptions.append(f"- '{col}': Measurement unit (e.g., kg, grams).")
        elif col_lower == "cost" and "currency" in columns_lower:
            descriptions.append(f"- '{col}': Monetary value of waste, expressed in 'Currency'.")
        elif col_lower == "currency":
            descriptions.append(f"- '{col}': Currency used for cost values (e.g., AED, USD).")
        elif col_lower == "loss reason":
            descriptions.append(f"- '{col}': Reason why the food was wasted.")
        elif col_lower == "disposition":
            descriptions.append(f"- '{col}': How the waste was handled (e.g., composted, disposed).")
        elif col_lower == "food item":
            descriptions.append(f"- '{col}': Specific food item being wasted.")
        elif col_lower == "food item category":
            descriptions.append(f"- '{col}': Group or category of the food item.")
        elif col_lower == "operator":
            descriptions.append(f"- '{col}': Person or staff member who logged the entry.")
        elif col_lower == "site":
            descriptions.append(f"- '{col}': Facility or location where waste occurred.")
        elif col_lower == "location":
            descriptions.append(f"- '{col}': Sub-location within the site.")
        elif col_lower == "device":
            descriptions.append(f"- '{col}': Device used to capture the entry.")
        elif col_lower == "date":
            descriptions.append(f"- '{col}': Date of the food waste entry.")
        elif col_lower == "time":
            descriptions.append(f"- '{col}': Time of the food waste entry.")
        elif col_lower == "region":
            descriptions.append(f"- '{col}': Geographic region.")
        elif col_lower == "country":
            descriptions.append(f"- '{col}': Country code or name.")
        elif col_lower == "stage of processing":
            descriptions.append(f"- '{col}': Phase of food processing (e.g., Pre-Consumer).")
        elif col_lower == "segment":
            descriptions.append(f"- '{col}': Business segment (e.g., Healthcare, Energy).")
        elif col_lower == "edited":
            descriptions.append(f"- '{col}': Indicates if the entry was edited (1 = yes, 0 = no).")
        elif col_lower == "quantity":
            descriptions.append(f"- '{col}': Number of items or units involved.")
        elif col_lower == "duration":
            descriptions.append(f"- '{col}': Duration in seconds (possibly related to operation time).")
        elif col_lower == "portions":
            descriptions.append(f"- '{col}': Number of food portions affected.")
        elif col_lower == "source":
            descriptions.append(f"- '{col}': Origin of the waste (e.g., Hot Line, Cold Prep).")
        elif col_lower == "container":
            descriptions.append(f"- '{col}': Type of container used (e.g., No Pan).")
        elif col_lower == "client location id":
            descriptions.append(f"- '{col}': Client-specific ID for the site.")
        elif col_lower == "client food item id":
            descriptions.append(f"- '{col}': Client-specific ID for the food item.")
        elif col_lower == "event order no.":
            descriptions.append(f"- '{col}': Order number used to track the waste event.")
        else:
            values = sample_values.get(col, [])
            if values:
                sample_snippet = ", ".join(map(str, values))
                descriptions.append(f"- '{col}': Sample values include {sample_snippet}.")
            else:
                descriptions.append(f"- '{col}': Column in the dataset (no sample values available).")

    return "\n".join(descriptions)


def ask_data_question_from_db(question: str, engine, client: OpenAI):
    table_columns = get_table_columns(engine)
    df_sample = pd.read_sql(text("SELECT * FROM food_waste LIMIT 50"), engine)
    sql = generate_sql_from_question(question, table_columns)
    patch_sql_func = generate_required_column_patcher(df_sample)  # df = actual full table or sample data
    sql = patch_sql_func(sql)
    df = pd.read_sql(text(sql), engine)

    if df.empty:
        return {"sql": sql, "answer": "⚠️ No data found for your query."}

    data_preview = df.head(10)
    preview_md = data_preview.to_markdown(index=False)
    column_context = generate_column_descriptions(df)

    explain_prompt = f"""
        You are analyzing food waste data from a PostgreSQL table named 'food_waste'.

        Here are descriptions of the table columns:
        {column_context}

        Below is a sample of the data returned from the query:

        {preview_md}

        Now, based on this data, answer the user's question:
        {question}
        """

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst who explains query results."},
            {"role": "user", "content": explain_prompt}
        ],
        max_tokens=400,
        temperature=0.3
    )

    return {"sql": sql, "answer": response.choices[0].message.content.strip()}


# def ask_data_question_from_db(question: str, engine) -> dict:
    # try:
        # from textwrap import dedent

        # table_columns = get_table_columns(engine)
        # sql = generate_sql_from_question(question, table_columns)
        # df = pd.read_sql(text(sql), engine)

        # if df.empty:
            # return {"sql": sql, "answer": "⚠️ No data found for your query."}

        # data_preview = df.head(10)
        # preview_md = data_preview.to_markdown(index=False)

        # # Optional: add semantic meaning to relevant columns
        # column_notes = []

        # if "Weight" in table_columns and "Unit" in table_columns:
            # column_notes.append("- 'Weight' indicates the quantity of food waste, measured in 'Unit' (e.g., kg, grams).")

        # if "Cost" in table_columns and "Currency" in table_columns:
            # column_notes.append("- 'Cost' reflects the monetary value of waste, using 'Currency' for the unit.")

        # if "Loss Reason" in table_columns:
            # column_notes.append("- 'Loss Reason' captures the reason for food waste (e.g., overproduction, spoilage).")

        # if "Disposition" in table_columns:
            # column_notes.append("- 'Disposition' describes how the waste was handled (e.g., disposed, composted).")

        # if "Food Item Category" in table_columns:
            # column_notes.append("- 'Food Item Category' groups items like Bread, Starch, Vegetables, etc.")

        # column_context = "\n".join(column_notes)

        # # 🧠 GPT-friendly prompt
        # explain_prompt = dedent(f"""
            # You are analyzing food waste data from a PostgreSQL table called 'food_waste'.
            # The table includes the following columns:

            # {', '.join(table_columns)}

            # {column_context}

            # Below is a sample of the data returned from a query:

            # {preview_md}

            # Now answer the user's question:
            # {question}
        # """)

        # response = client.chat.completions.create(
            # model="gpt-4-1106-preview",
            # messages=[
                # {"role": "system", "content": "You are a helpful data analyst who explains query results."},
                # {"role": "user", "content": explain_prompt}
            # ],
            # max_tokens=400,
            # temperature=0.3
        # )
        # # table_columns = get_table_columns(engine)
        # # sql = generate_sql_from_question(question, table_columns)
        # # df = pd.read_sql(text(sql), engine)

        # # if df.empty:
            # # return {"sql": sql, "answer": "⚠️ No data found for your query."}

        # # data_preview = df.head(10)
        # # preview_md = data_preview.to_markdown(index=False)

        # # explain_prompt = f"""
        # # Here is a sample of the data returned from the query:

        # # {preview_md}

        # # Now answer the user's question:
        # # {question}
        # # """

        # # response = client.chat.completions.create(
            # # model="gpt-4-1106-preview",
            # # messages=[
                # # {"role": "system", "content": "You are a helpful data analyst who explains query results."},
                # # {"role": "user", "content": explain_prompt}
            # # ],
            # # max_tokens=400,
            # # temperature=0.3
        # # )

        # return {
            # "sql": sql,
            # "table_preview": data_preview,
            # "answer": response.choices[0].message.content.strip()
        # }

    # except Exception as e:
        # return {"sql": None, "table_preview": None, "answer": f"❌ Error: {str(e)}"}



   
# def ask_data_question_from_db(question: str, engine) -> str:
    # table_columns = get_table_columns(engine)
    # filters = extract_filters_from_question(question, table_columns)
    # sql = build_sql_query(filters)

    # try:
        # df = pd.read_sql(sql, engine)

        # if df.empty:
            # return "⚠️ No relevant data found for your question."

        # data_sample = df.head(10).to_markdown(index=False)

        # prompt = f"""
        # You are a helpful data analyst. Use the following data to answer the user's question.

        # Data Sample:
        # {data_sample}

        # Question:
        # {question}
        # """

        # response = client.chat.completions.create(
            # model="gpt-4-1106-preview",
            # messages=[
                # {"role": "system", "content": "You are a helpful data analyst."},
                # {"role": "user", "content": prompt}
            # ],
            # max_tokens=500,
            # temperature=0.3
        # )
        # return response.choices[0].message.content.strip()

    # except Exception as e:
        # return f"❌ Error: {str(e)}"

    

# def ask_data_question(df: pd.DataFrame, question: str) -> str:
    # try:
        # data_json = df.to_json(orient="records")
        # prompt = f"""
# You are a helpful data analyst. Use the dataset below (in JSON format) to answer the user's question.

# Dataset:
# {data_json}

# Question:
# {question}

# Provide a clear and concise response.
# """

        # token_estimate = estimate_tokens(prompt)
        # if token_estimate > 100_000:
            # return f"⚠️ Dataset too large (estimated {token_estimate} tokens). Try using fewer rows or columns."

        # response = client.chat.completions.create(
            # model="gpt-4-1106-preview",
            # messages=[
                # {"role": "system", "content": "You are a helpful data analyst."},
                # {"role": "user", "content": prompt}
            # ],
            # max_tokens=500,
            # temperature=0.3
        # )
        # return response.choices[0].message.content.strip()

    # except Exception as e:
        # return f"❌ Error: {str(e)}"

# Load Data
@st.cache_data

# def load_raw_data(uploaded_file):
    
    # if uploaded_file:
        # df = pd.read_excel(uploaded_file)
        # df["Date"] = pd.to_datetime(df["Date"])
        # df["Month"] = df["Date"].dt.to_period("M").astype(str)
        # return df
    # else:
        # st.warning("Please upload a valid Excel file.")
        # return None
        
# @st.cache_data
def load_raw_data(uploaded_file):
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.to_period("M").astype(str)

        # Upload to PostgreSQL
        try:
            df.to_sql("food_waste", engine, if_exists="replace", index=False)
            # st.success("✅ Data uploaded to PostgreSQL.")
        except Exception as e:
            st.error(f"❌ Failed to upload to PostgreSQL: {e}")

        return df
    else:
        st.warning("Please upload a valid Excel file.")
        return None
        

# KPI Cards
def display_kpis(df, currency):
    total_cost = df["Cost"].sum()
    total_weight = df["Weight"].sum()
    avg_cost_per_kg = total_cost / total_weight if total_weight else 0
    top_loss_reason = df["Loss Reason"].mode()[0] if not df["Loss Reason"].isna().all() else "N/A"
    pre_consumer_pct = (df[df["Stage of Processing"] == "Pre-Consumer"].shape[0] / df.shape[0]) * 100

    col1, col2, col3 = st.columns(3)
    col4, col5 = st.columns(2)
    col1.metric("Total Cost", f"{currency} {total_cost:,.2f}")
    col2.metric("Total Weight", f"{total_weight:,.2f} kg")
    col3.metric("Avg Cost/KG", f"{currency} {avg_cost_per_kg:,.2f}")
    col4.metric("Top Loss Reason", top_loss_reason[:30])
    col5.metric("% Pre-Consumer", f"{pre_consumer_pct:.1f}%")

# Filters
def apply_filters(df):
    start_date = st.sidebar.date_input("Start Date", df["Date"].min().date())
    end_date = st.sidebar.date_input("End Date", df["Date"].max().date())
    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    regions = df["Region"].dropna().unique()
    # selected_region = st.sidebar.selectbox("Select Region", regions)
    # df = df[df["Region"] == selected_region]
    selected_regions = st.sidebar.multiselect("Select Region(s)", regions) #  default=list(regions)
    if selected_regions:
        df = df[df["Region"].isin(selected_regions)]


    sites = df["Site"].dropna().unique()
    # selected_site = st.sidebar.selectbox("Select Site", sites)
    # df = df[df["Site"] == selected_site]
    selected_sites = st.sidebar.multiselect("Select Site(s)", sites) # , default=list(sites)
    if selected_sites:
        df = df[df["Site"].isin(selected_sites)]

    locations = df["Location"].dropna().unique()
    # selected_location = st.sidebar.selectbox("Select Location", locations)
    # df = df[df["Location"] == selected_location]
    selected_location = st.sidebar.multiselect("Select Location(s)", locations) # , default=list(locations)
    if selected_location:
        df = df[df["Location"].isin(selected_location)]    

    operators = df["Operator"].dropna().unique()
    selected_operators = st.sidebar.multiselect("Select Operator(s)", operators) # , default=list(operators)

    if selected_operators:
        df = df[df["Operator"].isin(selected_operators)]

    return df

# Visualizations

def render_visualizations(df, currency):
    st.subheader("Wastage Trend Over Time")
    time_series = df.groupby("Date").agg({"Cost": "sum"}).reset_index()
    st.plotly_chart(px.line(time_series, x="Date", y="Cost", title="Wastage Cost Over Time ({currency})"))

    st.subheader("Wastage by Loss Reason")
    reason_chart = df["Loss Reason"].value_counts().reset_index()
    reason_chart.columns = ["Loss Reason", "Count"]
    st.plotly_chart(px.bar(reason_chart, x="Loss Reason", y="Count", title="Loss Reason Count"))

    st.subheader("Wastage by Food Category")
    metric_option = st.radio("Select metric for Food Category", ["Weight", "Cost"], horizontal=True)

    if metric_option == "Weight":
        category_data = df.groupby("Food Item Category")["Weight"].sum().reset_index().sort_values(by="Weight", ascending=False)
        y_col = "Weight"
        y_label = "Waste (kg)"
    elif metric_option == "Cost":
        category_data = df.groupby("Food Item Category")["Cost"].sum().reset_index().sort_values(by="Cost", ascending=False)
        y_col = "Cost"
        y_label = f"Cost ({currency})"

    selected_category = st.selectbox("Click to drill down by Food Category", category_data["Food Item Category"].unique())
    st.plotly_chart(px.bar(category_data, x="Food Item Category", y=y_col, title=f"Waste by Food Item Category ({y_label})", labels={y_col: y_label}))

    st.subheader(f"Food Items under '{selected_category}'")
    filtered_items = df[df["Food Item Category"] == selected_category]

    if metric_option == "Weight":
        item_chart = filtered_items.groupby("Food Item")["Weight"].sum().reset_index().sort_values(by="Weight", ascending=False)
        item_y = "Weight"
        item_label = "Waste (kg)"
    elif metric_option == "Cost":
        item_chart = filtered_items.groupby("Food Item")["Cost"].sum().reset_index().sort_values(by="Cost", ascending=False)
        item_y = "Cost"
        item_label = f"Cost ({currency})"

    st.plotly_chart(px.bar(item_chart, x="Food Item", y=item_y, title=f"Food Items in Category: {selected_category} ({item_label})", labels={item_y: item_label}))

    st.subheader("Disposition Distribution")
    disposition_chart = df["Disposition"].value_counts().reset_index()
    disposition_chart.columns = ["Disposition", "Count"]
    st.plotly_chart(px.pie(disposition_chart, names="Disposition", values="Count", title="Disposition Breakdown"))

    st.subheader("Stage of Processing")
    stage_chart = df["Stage of Processing"].value_counts().reset_index()
    stage_chart.columns = ["Stage", "Count"]
    st.plotly_chart(px.pie(stage_chart, names="Stage", values="Count", title="Processing Stage Breakdown"))

    st.subheader("Cost vs. Weight")
    st.plotly_chart(px.scatter(df, x="Weight", y="Cost", color="Loss Reason", title=f"Cost ({currency}) vs Weight (kg)", labels={"Weight": "Weight (kg)", "Cost": "Cost ($)"}))

    # st.subheader("Monthly Wastage Comparison")
    # monthly_chart = df.groupby("Month").agg({"Cost": "sum", "Weight": "sum"}).reset_index()
    # st.plotly_chart(px.bar(monthly_chart, x="Month", y=["Cost", "Weight"], barmode='group', title="Monthly Wastage Comparison"))
    st.subheader("Monthly Wastage Comparison")

    # Extract actual month for sorting
    df["MonthDate"] = pd.to_datetime(df["Date"]).dt.to_period("M").dt.to_timestamp()
    df["MonthLabel"] = df["MonthDate"].dt.strftime("%b %Y")

    # Group and sort by MonthDate
    monthly_chart = df.groupby(["MonthDate", "MonthLabel"]).agg({"Cost": "sum", "Weight": "sum"}).reset_index()
    monthly_chart = monthly_chart.sort_values("MonthDate")  # ✅ ensure correct order

    fig = px.bar(
        monthly_chart,
        x="MonthLabel",  # ✅ display clean month/year
        y=["Cost", "Weight"],
        barmode="group",
        title="Monthly Wastage Comparison"
    )

    fig.update_layout(xaxis_title="Month", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)



    st.subheader("Cost per KG by Site")
    site_cost_chart = df.groupby("Site").apply(lambda x: x["Cost"].sum() / x["Weight"].sum() if x["Weight"].sum() else 0).reset_index(name="Cost per KG")
    st.plotly_chart(px.bar(site_cost_chart, x="Site", y="Cost per KG", title="Cost per KG by Site"))

    st.subheader("Wastage Cost by Operator")
    operator_chart = df.groupby("Operator")["Cost"].sum().reset_index().sort_values(by="Cost", ascending=False)
    st.plotly_chart(px.bar(operator_chart, x="Operator", y="Cost", title="Wastage Cost by Operator"))
    
    st.subheader("Estimated CO2 Impact (Based on Weight)")
    df["Estimated CO2 (kg)"] = df["Weight"] * 2.5
    co2_chart = df.groupby("Date")["Estimated CO2 (kg)"].sum().reset_index()
    st.plotly_chart(px.area(co2_chart, x="Date", y="Estimated CO2 (kg)", title="Estimated CO2 Emissions from Food Waste"))
    
    st.subheader("CO2 Emissions by Disposition Method")
    co2_disp_chart = df.groupby("Disposition")["Estimated CO2 (kg)"].sum().reset_index()
    st.plotly_chart(px.bar(co2_disp_chart, x="Disposition", y="Estimated CO2 (kg)", title="CO2 Impact by Disposition Method"))
    
    # Forecast Section
    # import plotly.express as px
    # from prophet import Prophet

# import plotly.express as px
# from prophet import Prophet


    st.subheader("🔮 Forecast: Future Wastage Cost")

    forecast_days = st.radio("Select forecast period", [30, 60, 90], horizontal=True)
    
    try:
        from prophet import Prophet

        forecast_df = df.groupby("Date")["Cost"].sum().reset_index()
        forecast_df.columns = ["ds", "y"]

        model = Prophet()
        model.fit(forecast_df)

        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)

        # Merge forecast and actuals for Plotly
        plot_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        plot_df["type"] = ["Forecast"] * len(plot_df)
        actual_df = forecast_df.copy()
        actual_df.columns = ["ds", "y"]
        actual_df["type"] = "Actual"

        combined = pd.concat([
            actual_df[["ds", "y", "type"]],
            plot_df.rename(columns={"yhat": "y"})[["ds", "y", "type"]]
        ])

        fig = px.line(combined, x="ds", y="y", color="type", labels={"ds": "Date", "y": f"Cost ({currency})"})
        fig.update_layout(title=f"{forecast_days}-Day Forecast of Wastage Cost", xaxis_title="Date", yaxis_title=f"Cost ({currency})")
        st.plotly_chart(fig, use_container_width=True)
        
    

    except ImportError:
        st.error("Prophet is not installed. Run `pip install prophet`.")
    
    st.subheader("🔮 Forecast: Future Wastage Weight")

    try:
        # Prepare Weight data
        weight_df = df.groupby("Date")["Weight"].sum().reset_index()
        weight_df.columns = ["ds", "y"]

        model_w = Prophet()
        model_w.fit(weight_df)

        future_w = model_w.make_future_dataframe(periods=forecast_days)
        forecast_w = model_w.predict(future_w)

        # Merge forecast and actuals
        plot_w = forecast_w[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        plot_w["type"] = ["Forecast"] * len(plot_w)
        actual_w = weight_df.copy()
        actual_w.columns = ["ds", "y"]
        actual_w["type"] = "Actual"

        combined_w = pd.concat([
            actual_w[["ds", "y", "type"]],
            plot_w.rename(columns={"yhat": "y"})[["ds", "y", "type"]]
        ])

        fig_w = px.line(combined_w, x="ds", y="y", color="type", labels={"ds": "Date", "y": "Weight (kg)"})
        fig_w.update_layout(title=f"{forecast_days}-Day Forecast of Wastage Weight", xaxis_title="Date", yaxis_title="Weight (kg)")
        st.plotly_chart(fig_w, use_container_width=True)

    except Exception as e:
        st.error(f"Weight forecasting failed: {e}")


    
    except Exception as e:
        st.error(f"Forecasting failed: {e}")




















# Main App
# def main():
    # st.set_page_config(layout="wide", page_title="Waste Watch")
    # st.image("logo.png", width=150)
    # #st.title("\U0001F372 Waste Watch Analytics Dashboard")
    # st.title("Waste Watch Analytics Dashboard")

    # uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
    
    # if uploaded_file:
        # df = load_raw_data(uploaded_file)
        
        # # ✅ Correct placement of currency logic
        # if "Currency" in df.columns:
            # currency_values = df["Currency"].dropna().unique()
            # currency = currency_values[0] if len(currency_values) == 1 else ""
        # else:
            # currency = "{currency}"

        # if df is not None:
            # df_filtered = apply_filters(df)
            # display_kpis(df_filtered, currency)
            # render_visualizations(df_filtered, currency)
            
       # # 💬 Add GPT chat section at the bottom
    # st.markdown("---")
    # st.subheader("💬 Ask Questions About This Dataset")

    # user_question = st.text_input("Ask a question like 'What is the most common loss reason?'")

    # if user_question:
        # with st.spinner("Asking GPT..."):
            # answer = ask_data_question(df_filtered, user_question)
            # st.success(answer)     
            
    # else:
        # st.info("📂 Please upload an Excel file to get started.")

def main():
    st.set_page_config(layout="wide", page_title="Waste Watch")
    st.image("logo.png", width=150)
    st.title("Waste Watch Analytics Dashboard")

    uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
    
    if "gpt_question" in st.session_state and not uploaded_file:
        st.session_state.gpt_question = ""

    st.markdown("## 💬 Talk to your Data")
    user_question = st.text_input("Ask a question like 'What is the most common loss reason?'", key="gpt_question")

    df = None
    df_filtered = None
    currency = ""
    result = None  # Store GPT result

    # ✅ If file is uploaded, load it and allow GPT to run
    if uploaded_file:
        df = load_raw_data(uploaded_file)

        if "Currency" in df.columns:
            currency_values = df["Currency"].dropna().unique()
            currency = currency_values[0] if len(currency_values) == 1 else ""
        else:
            currency = "{currency}"

        df_filtered = apply_filters(df)

        # ✅ Run GPT based on question (after file load)
        if user_question:
            with st.spinner("🤖 GPT is thinking..."):
                #result = ask_data_question_from_db(user_question, engine)
                result = ask_data_question_from_db(user_question, engine, client)

    # ✅ Show GPT output (always below input)
    if result and result.get("answer"):
        if SHOW_TECHNICAL_DETAILS:
            st.markdown("#### 💬 GPT Insight")

            if result.get("sql"):
                st.markdown("#### 🧾 Generated SQL")
                st.code(result["sql"], language="sql")

            if result.get("table_preview") is not None:
                st.markdown("#### 📋 Query Result Sample")
                st.dataframe(result["table_preview"])

            st.success(result["answer"])
        else:
            # Business-friendly view only
            st.success(result["answer"])

# ✅ OUTSIDE the result block
    elif user_question and not uploaded_file:
        st.warning("📂 Please upload a file to query.")

    # ✅ Always render visuals below GPT block
    if df_filtered is not None:
        st.markdown("---")
        st.markdown("## 📊 Visualizations & Insights")
        display_kpis(df_filtered, currency)
        render_visualizations(df_filtered, currency)


# def main():
    # st.set_page_config(layout="wide", page_title="Waste Watch")
    # st.image("logo.png", width=150)
    # st.title("Waste Watch Analytics Dashboard")

    # uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])

    # st.markdown("## 💬 Talk to your data")
    # user_question = st.text_input("Ask a question like 'What is the most common loss reason?'")

    # df = None
    # df_filtered = None
    # currency = ""

    # if uploaded_file:
        # df = load_raw_data(uploaded_file)

        # if "Currency" in df.columns:
            # currency_values = df["Currency"].dropna().unique()
            # currency = currency_values[0] if len(currency_values) == 1 else ""
        # else:
            # currency = "{currency}"

        # df_filtered = apply_filters(df)

        # # ✅ Render visualizations regardless of GPT question
        # with st.container():
            # display_kpis(df_filtered, currency)
            # render_visualizations(df_filtered, currency)

        # # ✅ If GPT question is asked, show GPT answer block
        # if user_question:
            # with st.spinner("🤖 GPT is thinking..."):
                # result = ask_data_question_from_db(user_question, engine)

                # if result.get("sql"):
                    # st.markdown("#### 🧾 Generated SQL")
                    # st.code(result["sql"], language="sql")

                # if result.get("table_preview") is not None:
                    # st.markdown("#### 📋 Query Result Sample")
                    # st.dataframe(result["table_preview"])

                # if result.get("answer"):
                    # st.markdown("#### 💬 GPT Answer")
                    # st.success(result["answer"])

    # elif user_question and not uploaded_file:
        # st.warning("📂 Please upload and load data first.")

        
# def main():
    # st.set_page_config(layout="wide", page_title="Waste Watch")
    # st.image("logo.png", width=150)
    # st.title("Waste Watch Analytics Dashboard")

    # uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])

    # # 💬 GPT chat assistant (always visible in sidebar)
    # # st.sidebar.markdown("## 💬 Ask GPT")
    # # user_question = st.sidebar.text_input("Ask a question")
    # st.markdown("## 💬 Talk to your data")
    # user_question = st.text_input("Ask a question like 'What is the most common loss reason?'")    
    
    # df_filtered = None
    # if uploaded_file:
        # df = load_raw_data(uploaded_file)

        # if "Currency" in df.columns:
            # currency_values = df["Currency"].dropna().unique()
            # currency = currency_values[0] if len(currency_values) == 1 else ""
        # else:
            # currency = "{currency}"

        # if df is not None:
            # df_filtered = apply_filters(df)

        # if user_question and df_filtered is not None:
            # with st.spinner("🤖 GPT is thinking..."):
                # result = ask_data_question_from_db(user_question, engine)

                # if result.get("sql"):
                    # st.markdown("#### 🧾 Generated SQL")
                    # st.code(result["sql"], language="sql")

                # if result.get("table_preview") is not None:
                    # st.markdown("#### 📋 Query Result Sample")
                    # st.dataframe(result["table_preview"])

                # if result.get("answer"):
                    # st.markdown("#### 💬 GPT Answer")
                    # st.success(result["answer"])

        # elif user_question and df_filtered is None:
            # st.warning("📂 Please upload and load data first.")


            # # Scrollable section for visualizations
            # with st.container():
                # st.markdown(
                    # """
                    # <style>
                    # .scrollable-section {
                        # max-height: 75vh;
                        # overflow-y: auto;
                        # padding-right: 10px;
                    # }
                    # </style>
                    # <div class="scrollable-section">
                    # """,
                    # unsafe_allow_html=True
                # )

                # display_kpis(df_filtered, currency)
                # render_visualizations(df_filtered, currency)

                # st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown("---")
    # st.subheader("💬 Ask GPT About This Dataset")

    # user_question = st.text_input("Ask a question like 'What is the most common loss reason?'")

    # Process chat input
    # if user_question and df_filtered is not None:
        # with st.spinner("🤖 GPT is thinking..."):
            # result = ask_data_question_from_db(user_question, engine)

            # if result.get("sql"):
                # st.markdown("#### 🧾 Generated SQL")
                # st.code(result["sql"], language="sql")

            # if result.get("table_preview") is not None:
                # st.markdown("#### 📋 Query Result Sample")
                # st.dataframe(result["table_preview"])

            # if result.get("answer"):
                # st.markdown("#### 💬 GPT Answer")
                # st.success(result["answer"])

    # elif user_question and df_filtered is None:
        # st.warning("📂 Please upload and load data first.")


        

if __name__ == "__main__":
    main()