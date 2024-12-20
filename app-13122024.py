import streamlit as st
import openai
import pandas as pd
import pandasql as ps
from sqlalchemy import create_engine
from datetime import datetime
import tiktoken
import json

# Preload the API key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# Initialize tokenizer for GPT-3.5-turbo
def num_tokens_from_string(string: str) -> int:
    """Calculate number of tokens in a string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

class ChatMessage:
    """Represents a single message in the chat interface."""
    def __init__(self, role: str, content: str, sql_query: str = None, data_preview = None):
        self.role = role
        self.content = content
        self.sql_query = sql_query
        self.data_preview = data_preview
        self.timestamp = datetime.now()

class ConversationMemory:
    """Manages conversation history with token limiting."""
    def __init__(self, max_tokens=12000):
        self.max_tokens = max_tokens
        self.messages = []
        self.current_tokens = 0
        self.chat_history = []
    
    def add_message(self, role: str, content: str, sql_query: str = None, data_preview = None):
        """Add a message to both conversation memory and chat history."""
        # Add to token-managed memory
        message = {"role": role, "content": content}
        message_tokens = num_tokens_from_string(content)
        
        while (self.current_tokens + message_tokens) > self.max_tokens and self.messages:
            removed_message = self.messages.pop(0)
            self.current_tokens -= num_tokens_from_string(removed_message["content"])
        
        self.messages.append(message)
        self.current_tokens += message_tokens
        
        # Add to chat history
        chat_message = ChatMessage(role, content, sql_query, data_preview)
        self.chat_history.append(chat_message)
    
    def get_messages(self):
        """Get all messages in the conversation history."""
        return self.messages
    
    def get_formatted_history(self):
        """Get conversation history formatted as a string."""
        return "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.messages
        ])

    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        self.current_tokens = 0
        self.chat_history = []

def render_chat_message(message: ChatMessage):
    """Render a single chat message with appropriate styling."""
    if message.role == "user":
        st.markdown("🧑‍💻 **You:**")
        st.markdown(message.content)
    else:
        st.markdown("🤖 **Assistant:**")
        st.markdown(message.content)
        if message.sql_query:
            with st.expander("View SQL Query"):
                st.code(message.sql_query, language="sql")
        if message.data_preview is not None:
            with st.expander("View Data"):
                st.dataframe(message.data_preview)

# SQL Template with conversation history
SQL_PROMPT_TEMPLATE = """
You are a highly skilled data analyst and SQL expert. Write SQL queries for the given question using the provided database schema.Do not include any Markdown formatting like ```sql in your response.
Today's date is: {current_date}

Previous conversation context:
{conversation_history}

Important guidelines:
1. Make sure the SQL is compatible with SQLite
2. Avoid functions like MONTH() and CURRENT_DATE; instead, use strftime('%Y-%m-%d', date_column)
3. For column names with spaces or special characters, wrap them in square brackets or double quotes
4. Use the table name 'data'
5. When dealing with relative dates (e.g., "last 6 months"), use '{current_date}' as the reference point
6. Do not include any Markdown formatting in your response
7. Consider previous questions and their context when generating new queries

Database schema:
{schema}

Question:
{question}

SQL Query:
"""

# Analysis Template with conversation history
ANALYSIS_TEMPLATE = """
You are a data analyst providing insights about query results.
Consider the conversation history and previous analyses when explaining new results.
You can also respond to general questions.

Previous conversation context:
{conversation_history}

Current question: {question}

Data to analyze:
{data}

Guidelines for analysis:
1. Analyze the numbers and patterns in the data
2. Reference relevant information from previous queries when applicable
3. Provide clear, data-driven insights
4. Highlight any trends or patterns, especially in relation to previous queries
5. Use specific numbers and percentages when relevant
6. If comparing to previous results, make explicit comparisons
7. Keep the explanation clear and concise

You can also respond to general questions.
"""

def generate_sql(query: str, schema: str, conversation_memory: ConversationMemory) -> str:
    """Generate SQL query using OpenAI with conversation context."""
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    prompt = SQL_PROMPT_TEMPLATE.format(
        schema=schema,
        question=query,
        current_date=current_date,
        conversation_history=conversation_memory.get_formatted_history()
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    
    sql_query = response['choices'][0]['message']['content']
    conversation_memory.add_message("user", query)
    return sql_query

def analyze_results(question: str, data: list, conversation_memory: ConversationMemory) -> str:
    """Analyze query results with conversation context."""
    prompt = ANALYSIS_TEMPLATE.format(
        conversation_history=conversation_memory.get_formatted_history(),
        question=question,
        data=json.dumps(data, indent=2)
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0
    )
    
    analysis = response['choices'][0]['message']['content']
    return analysis

def load_data_from_file(file):
    """Load data from uploaded file."""
    try:
        if file.name.endswith('.csv'):
            data = pd.read_csv(file)
        else:
            data = pd.read_excel(file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def load_data_from_db(db_url: str, table_name: str):
    """Load data from database."""
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        return data
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def get_schema_description(data: pd.DataFrame) -> str:
    """Generate schema description from DataFrame."""
    schema = []
    for col, dtype in zip(data.columns, data.dtypes):
        needs_quotes = ' ' in col
        schema.append(f"{col}: {dtype} {'(requires quotes)' if needs_quotes else ''}")
    return "\n".join(schema)

def initialize_session_state():
    """Initialize session state variables."""
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationMemory()
    if "data" not in st.session_state:
        st.session_state.data = None
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

def main():
    st.set_page_config(page_title="Marketing Assistant Chat", layout="wide")
    initialize_session_state()
    
    st.title("Marketing Assistant")
    
    # Sidebar setup
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear Chat History"):
            st.session_state.conversation_memory.clear()
            st.success("Chat history cleared!")
        
        st.subheader("Memory Usage")
        st.progress(st.session_state.conversation_memory.current_tokens / st.session_state.conversation_memory.max_tokens)
        st.text(f"Tokens: {st.session_state.conversation_memory.current_tokens}/{st.session_state.conversation_memory.max_tokens}")
        
        # Data Source Selection
        st.subheader("Data Source")
        data_source = st.radio("Select Data Source", ["Upload File", "Database Connection"])
        
        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"])
            if uploaded_file:
                data = load_data_from_file(uploaded_file)
                if data is not None:
                    st.session_state.data = data
                    st.success("Data loaded successfully!")
                    with st.expander("View Data Preview"):
                        st.write(data.head())
        
        else:  # Database Connection
            db_url = st.text_input("Database URL", placeholder="sqlite:///database.db")
            table_name = st.text_input("Table Name", placeholder="your_table")
            if st.button("Connect") and db_url and table_name:
                data = load_data_from_db(db_url, table_name)
                if data is not None:
                    st.session_state.data = data
                    st.success("Database connected successfully!")
                    with st.expander("View Data Preview"):
                        st.write(data.head())
    
    # Main chat interface
    if st.session_state.data is not None:
        # Chat history container
        chat_container = st.container()
        
        # Input form at the bottom
        with st.form(key="query_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            with col1:
                query = st.text_area("Ask a question about your data:", 
                                   key="query_input", 
                                   height=100)
            with col2:
                submit_button = st.form_submit_button("Send", use_container_width=True)
        
        # Display chat history
        with chat_container:
            for message in st.session_state.conversation_memory.chat_history:
                render_chat_message(message)
            
            if submit_button and query:
                try:
                    # Generate and execute SQL query
                    schema = get_schema_description(st.session_state.data)
                    sql_query = generate_sql(query, schema, st.session_state.conversation_memory)
                    
                    # Execute query
                    result = ps.sqldf(sql_query, {"data": st.session_state.data})
                    
                    # Analyze results
                    result_dict = result.to_dict(orient="records")
                    analysis = analyze_results(query, result_dict, st.session_state.conversation_memory)
                    
                    # Add to chat history
                    st.session_state.conversation_memory.add_message(
                        "assistant",
                        analysis,
                        sql_query=sql_query,
                        data_preview=result
                    )
                    
                    # Rerun to update chat display
                    st.rerun()
                    
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.session_state.conversation_memory.add_message(
                        "assistant",
                        error_message,
                        sql_query=sql_query if 'sql_query' in locals() else None
                    )
                    st.rerun()
    else:
        st.info("👈 Please load your data using the sidebar options to start the conversation.")

if __name__ == "__main__":
    main()