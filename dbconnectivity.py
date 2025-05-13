import requests
import csv
import io

from vanna.base import VannaBase
from vanna.chromadb import ChromaDB_VectorStore


# ---- Configuration ----
QUESTDB_IP = "40.81.240.69"
QUESTDB_PORT = "5000"


# ---- Helper Functions ----
def run_questdb_query(query: str) -> str:
    """Runs an SQL query against QuestDB via its /exec endpoint."""
    try:
        response = requests.get(
            f"http://{QUESTDB_IP}:{QUESTDB_PORT}/exec",
            params={"query": query.strip()},  # Send raw SQL only
            timeout=10
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Query execution failed: {e}"


def my_local_llm(prompt: str) -> str:
    """Calls local LLaMA model via Ollama API."""
    response = requests.post("http://135.13.34.15:11434/api/generate", json={
        "model": "llama3.1",
        "prompt": prompt,
        "stream": False
    })
    return response.json()['response']


def format_csv_response(csv_text: str):
    """Formats CSV response from QuestDB into readable text."""
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return "No data returned."
    return "\n".join([", ".join(row) for row in rows])


def clean_sql(sql: str) -> str:
    """
    Removes any markdown formatting, triple backticks, and extra whitespace.
    Ensures clean SQL output for execution.
    """
    return sql.replace("```sql", "").replace("```", "").strip()


# ---- Vector Store Implementation ----
class CustomChromaDB_VectorStore(ChromaDB_VectorStore):
    def __init__(self, config: dict):
        self.config = config
        persist_directory = self.config.get("persist_directory", "./chroma-persist")
        super().__init__(config)

    def assistant_message(self, message):
        print(f"[VectorStore] Assistant: {message}")

    def submit_prompt(self, prompt):
        print(f"[VectorStore] Prompt submitted: {prompt}")

    def system_message(self, message):
        print(f"[VectorStore] System: {message}")

    def user_message(self, message):
        print(f"[VectorStore] User: {message}")


# ---- Vanna Implementation ----
class CustomVanna(VannaBase):
    def __init__(self, model, vectorstore, llm, run_sql):
        self.model = model
        self.vectorstore = vectorstore
        self.llm = llm
        self.run_sql = run_sql
        self.dialect = self.vectorstore.config.get("dialect", "SQL")

    # --- Required Abstract Methods ---
    def add_question_sql(self, question: str, sql: str):
        print(f"Training - Question: '{question}', SQL: '{sql}'")
        self.vectorstore.add_question_sql(question=question, sql=sql)

    def add_ddl(self, ddl: str):
        print(f"Adding DDL: {ddl}")
        self.vectorstore.add_ddl(ddl=ddl)

    def add_documentation(self, documentation: str):
        print(f"Adding Documentation: {documentation}")
        self.vectorstore.add_documentation(documentation=documentation)

    def get_similar_question_sql(self, question: str):
        print(f"Finding similar questions to: {question}")
        return self.vectorstore.get_similar_question_sql(question)

    def get_related_ddl(self, table_name: str):
        print(f"Getting related DDL for: {table_name}")
        return self.vectorstore.get_related_ddl(table_name)

    def get_related_documentation(self, table_name: str):
        print(f"Getting related documentation for: {table_name}")
        return self.vectorstore.get_related_documentation(table_name)

    def generate_embedding(self, text: str):
        print(f"Generating embedding for: {text}")
        # Placeholder – replace with real embeddings later
        return [0.1] * 128

    def submit_prompt(self, prompt: str):
        print(f"Submitting prompt to LLM:\n{prompt}")
        return self.llm(prompt)

    def get_sql(self, question: str) -> str:
        print(f"Getting SQL for question: {question}")
        raw_sql = self.submit_prompt(self.generate_prompt(question))
        return clean_sql(raw_sql)

    def run_sql(self, sql: str):
        print(f"Executing SQL:\n{sql}")
        return self.run_sql(sql)

    def get_training_data(self):
        print("Fetching training data...")
        return []

    def remove_training_data(self):
        print("Removing training data...")

    def add_table_documentation(self, table_name: str, documentation: str):
        """Adds table-level documentation"""
        self.add_documentation(f"Table: {table_name}\n{documentation}")

    def generate_prompt(self, question: str) -> str:
        """
        Generates a prompt for the LLM based on the question and training data.
        """
        ddl_statements = self.get_related_ddl(question)
        doc_string = self.get_related_documentation(question)

        prompt = f"You are a SQL assistant. Generate a SQL query to answer the question: '{question}'\n\n"

        if ddl_statements:
            prompt += f"Use the following database schema:\n{ddl_statements}\n\n"

        if doc_string:
            prompt += f"Additional context:\n{doc_string}\n\n"

        prompt += "Only respond with the SQL query, no explanation, no markdown, no triple backticks."

        return prompt

    # --- Messaging Stubs ---
    def assistant_message(self, message):
        print(f"[Vanna] Assistant: {message}")

    def system_message(self, message):
        print(f"[Vanna] System: {message}")

    def user_message(self, message):
        print(f"[Vanna] User: {message}")


# ---- Initialize Vanna Instance ----
vn = CustomVanna(
    model='questdb-vanna',
    vectorstore=CustomChromaDB_VectorStore(config={
        "persist_directory": "./vanna-chroma",
        "dialect": "QuestDB SQL"
    }),
    llm=my_local_llm,
    run_sql=run_questdb_query
)

# ---- Train Vanna with Schema Info ----
vn.add_table_documentation(
    table_name="cost_tables_info",
    documentation="This table contains metadata about cost-related tables."
)

vn.add_ddl("CREATE TABLE cost_tables_info (cost_id INT, service_name TEXT, cost_value FLOAT, billing_date TIMESTAMP);")

vn.add_question_sql(
    question="Show me the first 5 rows of cost tables info",
    sql="SELECT * FROM cost_tables_info LIMIT 5;"
)

# ---- Ask a Natural Language Question ----
question = "How many clusters are there?"
sql_query = vn.get_sql(question)

print("\nGenerated SQL Query:\n", sql_query)

raw_result = run_questdb_query(sql_query)

print("\nQuery Result:")
print(format_csv_response(raw_result))


