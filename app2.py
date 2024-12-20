import streamlit as st
import openai
import pandas as pd
import base64
import pandasql as ps
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine
from datetime import datetime
import re

# Preload the API key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Updated SQL Prompt Template with current date information
SQL_PROMPT_TEMPLATE = """
You are a highly skilled data analyst and SQL expert. Write SQL queries for the given question using the provided database schema.Do not include any Markdown formatting like ```sql in your response.
Today's date is: {current_date}

Important guidelines:
1. Make sure the SQL is compatible with SQLite
2. Avoid functions like MONTH() and CURRENT_DATE; instead, use strftime('%Y-%m-%d', date_column)
3. For column names with spaces or special characters, wrap them in square brackets or double quotes
4. Use the table name 'data'
5. When dealing with relative dates (e.g., "last 6 months"), use '{current_date}' as the reference point
6. Do not include any Markdown formatting in your response

Database schema:
{schema}

Question:
{question}

SQL Query:
"""

def generate_sql(query, schema):
    """Generate SQL query using OpenAI LLM with conversation history."""
    current_date = datetime.now().strftime('%Y-%m-%d')
    prompt = SQL_PROMPT_TEMPLATE.format(
        schema=schema,
        question=query,
        current_date=current_date
    )

    # Append the user prompt to the conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the OpenAI API with the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
        temperature=0
    )

    # Get the assistant's response
    sql_query = response['choices'][0]['message']['content']

    # Remove Markdown code block markers (```sql ... ```)
    #cleaned_sql_query = re.sub(r'```.*?```', '', sql_query, flags=re.DOTALL).strip()

    # Add the assistant's response to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": sql_query})

    return sql_query

def fetch_and_store_data(db_url, table_name):
    """Fetch data from the database and store it in session state."""
    try:
        engine = create_engine(db_url)
        query = f"SELECT * FROM {table_name}"
        with engine.connect() as conn:
            data = pd.read_sql(query, conn)
        st.session_state["data"] = data
        st.session_state["db_url"] = db_url
        st.session_state["table_name"] = table_name
        st.success("Data has been loaded and stored in the session.")
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None

def process_data(data):
    """Process data and store it in session state."""
    if "Date" in data.columns:
        try:
            data["Date"] = pd.to_datetime(data["Date"]).dt.strftime('%Y-%m-%d')
        except Exception:
            st.warning("Failed to convert 'Date' column to proper format.")
    st.session_state["data"] = data
    st.success("Data has been loaded and stored in the session.")

def explain_query_result(question, query_result):
    """Use GPT to generate a natural language explanation of the query result with conversation history."""
    if not query_result:
        prompt = f"""
You are a helpful assistant skilled in analyzing data and explaining results in natural language.

User Question:
{question}

Query Result:
The query returned no results (empty dataset).

Task: Explain what it means to have no results for this specific question. Do not make up or infer any data. 
Clearly state that no data was found and suggest possible reasons why (like data might not exist for the specified time period, 
or the specified conditions might not match any records).
"""
    else:
        formatted_result = "\n".join([str(row) for row in query_result])
        prompt = f"""
You are a helpful assistant skilled in analyzing data and explaining results in natural language.

User Question:
{question}

Query Result:
{formatted_result}

Task: Based on the query result shown above:
1. The data IS present and shows actual results - do not say there is no data
2. Analyze the numbers and patterns in the data shown
3. Provide insights about the Category and Repeat_Customers columns
4. Make clear statements about what the data shows (e.g., which categories have the highest/lowest numbers)
5. Do not make assumptions beyond what's directly shown in the data
6. If any interesting patterns or notable differences exist in the data, point them out
"""

    # Add the user's explanation prompt to the conversation history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the OpenAI API with the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages,
        temperature=0
    )

    explanation = response['choices'][0]['message']['content']

    # Add the assistant's explanation to the conversation history
    st.session_state.messages.append({"role": "assistant", "content": explanation})

    return explanation

def main():
    # Load your local logo file
    logo_path = "bblogoimages.png"  # Replace with the path to your logo file
    with open(logo_path, "rb") as image_file:
        logo_base64 = base64.b64encode(image_file.read()).decode()

    # Navigation Bar with Logo and Company Name
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; background-color: #039dfc; padding: 0px;">
            <img src="data:image/png;base64,{logo_base64}" style="height: 80px; margin-right: 1px;">
            <h1 style="color: white; font-size: 30px; margin-left: 10px;">Data Assistant</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set in the backend.")
        st.stop()

    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "db_url" not in st.session_state:
        st.session_state["db_url"] = None
    if "table_name" not in st.session_state:
        st.session_state["table_name"] = None

    # Initialize conversation history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "You are a helpful assistant skilled in generating SQL queries and explaining data results."}
        ]
    data_source = st.selectbox("Select Data Source", options=["Upload Excel File", "Database Connection"])

    if data_source == "Upload Excel File":
        uploaded_file = st.file_uploader("Upload an Excel File", type=["xlsx", "xls", "csv"])
        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            st.write("Preview of the uploaded file:")
            st.write(data.head())

            if st.button("Process Data"):
                process_data(data)

    elif data_source == "Database Connection":
        db_url = st.text_input("Enter Database Connection URL", placeholder="e.g., sqlite:///my_database.db")
        table_name = st.text_input("Enter Table Name", placeholder="e.g., sales_data")
        if st.button("Connect and Fetch Data"):
            if db_url and table_name:
                data = fetch_and_store_data(db_url, table_name)
                if data is not None:
                    st.write("Preview of the fetched data:")
                    st.write(data.head())
            else:
                st.warning("Please provide both the database URL and the table name.")

    if st.session_state["data"] is not None:
        st.header("Ask Questions about the Data")
        query = st.chat_input("Enter your question:")
        
        if query:
            # Display user message in chat message container
            #st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            schema = []
            for col, dtype in zip(st.session_state["data"].columns, st.session_state["data"].dtypes):
                needs_quotes = ' ' in col
                schema.append(f"{col}: {dtype} {'(requires quotes)' if needs_quotes else ''}")
            schema_str = "\n".join(schema)

            try:
                sql_query = generate_sql(query, schema_str)
                data = st.session_state["data"]
                result = ps.sqldf(sql_query, {"data": data})

                st.subheader("Query Result")
                if result.empty:
                    st.warning("The query returned no results.")
                else:
                    st.write(result)

                result_summary = result.to_dict(orient="records")
                gpt_response = explain_query_result(query, result_summary)
                with st.chat_message("assistant"):
                    #st.subheader("Answer")
                    st.write(gpt_response)
                    #st.session_state.messages.append({"role": "assistant", "content": gpt_response})

            except Exception as e:
                st.error(f"Failed to execute the query: {e}")
                st.error(f"Generated SQL query: {sql_query}")
                gpt_response = explain_query_result(query, [{"error": str(e)}])
                st.subheader("Error Analysis")
                st.write(gpt_response)

    # if st.button("Clear Conversation History"):
    #     st.session_state.messages = [
    #         {"role": "system", "content": "You are a helpful assistant skilled in generating SQL queries and explaining data results."}
    #     ]
    #     st.success("Conversation history cleared.")

if __name__ == "__main__":
    main()
