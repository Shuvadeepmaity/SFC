import streamlit as st
import openai
import pandas as pd
import pandasql as ps  # For running SQL on Pandas DataFrame
from langchain.prompts import PromptTemplate
from sqlalchemy import create_engine

# Preload the API key
OPENAI_API_KEY = "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"  # Replace with your actual API key
openai.api_key = OPENAI_API_KEY

# Custom SQL Prompt Template for SQLite
SQL_PROMPT_TEMPLATE = """
You are a highly skilled data analyst and SQL expert. Write SQL queries for the given question using the provided database schema.
Make sure the SQL is compatible with SQLite. Avoid functions like MONTH() and CURRENT_DATE; instead, use strftime('%Y-%m-%d', date_column) or similar for date filtering.
Use the table name 'data'. Do not include any Markdown formatting like ```sql in your response.

Database schema:
{schema}

Question:
{question}

SQL Query:
"""

def generate_sql(query, schema):
    """Generate SQL query using OpenAI LLM."""
    prompt = SQL_PROMPT_TEMPLATE.format(schema=schema, question=query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def fetch_and_store_data(db_url, table_name):
    """Fetch data from the database and store it in session state."""
    try:
        # Connect to the database
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
    # Convert date column to SQLite-compatible format if it exists
    if "Date" in data.columns:
        try:
            data["Date"] = pd.to_datetime(data["Date"]).dt.strftime('%Y-%m-%d')
        except Exception:
            st.warning("Failed to convert 'Date' column to proper format.")
    # Store the uploaded data in session state
    st.session_state["data"] = data
    st.success("Data has been loaded and stored in the session.")

def explain_query_result(question, query_result):
    """
    Use GPT to generate a natural language explanation of the query result.

    Args:
        question (str): The user's original question.
        query_result (list): Query result as a list of dictionaries.

    Returns:
        str: GPT-generated explanation.
    """
    formatted_result = "\n".join([str(row) for row in query_result])
    prompt = f"""
You are a helpful assistant skilled in analyzing data and explaining results in natural language.

User Question:
{question}

Query Result:
{formatted_result}

Based on the query result and the question, generate a concise and meaningful explanation of the result in plain English.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

# Streamlit App
def main():
    st.title("Data Assistant")
    
    # Ensure API key is set
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is not set in the backend.")
        st.stop()
    
    # Initialize session state variables
    if "data" not in st.session_state:
        st.session_state["data"] = None
    if "db_url" not in st.session_state:
        st.session_state["db_url"] = None
    if "table_name" not in st.session_state:
        st.session_state["table_name"] = None

    # Dropdown to select data source
    data_source = st.selectbox("Select Data Source", options=["Upload Excel File", "Database Connection"])
    
    # Handle Excel Upload Option
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
    
    # Handle Database Connection Option
    elif data_source == "Database Connection":
        db_url = st.text_input("Enter Database Connection URL", placeholder="e.g., sqlite:///my_database.db")
        table_name = st.text_input("Enter Table Name", placeholder="e.g., sales_data")
        if st.button("Connect and Fetch Data"):
            if db_url and table_name:
                try:
                    engine = create_engine(db_url)
                    query = f"SELECT * FROM {table_name}"  # Fetch all data from the specified table
                    with engine.connect() as conn:
                        data = pd.read_sql(query, conn)
                    st.session_state["data"] = data
                    st.session_state["db_url"] = db_url
                    st.session_state["table_name"] = table_name
                    st.success(f"Connected to database and loaded table '{table_name}'.")
                    st.write("Preview of the fetched data:")
                    st.write(data.head())
                except Exception as e:
                    st.error(f"Failed to connect or fetch data: {e}")

            else:
                st.warning("Please provide both the database URL and the table name.")
        else:
            # If the connection is already set up, show the details
            st.success(f"Connected to database: {st.session_state['db_url']} (Table: {st.session_state['table_name']})")
            st.write("Preview of the loaded data:")
            #st.write(st.session_state["data"].head())

    # Check if data has been loaded
    if st.session_state["data"] is not None:
        st.header("Ask Questions about the Data")
        query = st.text_input("Enter your question:")
        if query:
            # Generate SQL Query
            schema = "\n".join([f"{col}: {dtype}" for col, dtype in zip(st.session_state["data"].columns, st.session_state["data"].dtypes)])
            sql_query = generate_sql(query, schema)
            #st.subheader("Generated SQL Query")
            #st.code(sql_query, language="sql")
            
            # Execute the SQL Query using pandasql
            #if st.button("Execute Query"):
            try:
                # Assign the DataFrame to the default table name
                data = st.session_state["data"]
                result = ps.sqldf(sql_query, {"data": data})
                st.subheader("Query Result")
                st.write(result)
                # Format the result and question for GPT
                result_summary = result.to_dict(orient="records")  # Convert result to a list of dictionaries
                gpt_response = explain_query_result(query, result_summary)
                
                st.subheader("Answer")
                st.write(gpt_response)
            except Exception as e:
                st.error(f"Failed to execute the query: {e}")

if __name__ == "__main__":
    main()