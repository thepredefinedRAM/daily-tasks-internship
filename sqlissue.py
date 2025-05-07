from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_community.llms import Ollama  # Adjust based on your LLM
import requests

# ---- Step 1: Connect to QuestDB ----
QUESTDB_IP = "127.0.0.1.fff"
QUESTDB_PORT = "9000"

def run_questdb_query(query: str) -> str:
    try:
        response = requests.get(
            f"http://{QUESTDB_IP}:{QUESTDB_PORT}/exec",
            params={"query": query}
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Query execution failed: {e}"

# ---- Step 2: Prompt Templates ----
sql_prompt = PromptTemplate.from_template("""
You are a SQL expert. Given the user question and schema, write a correct SQL query.
Schema:
{schema}
Question:
{question}
SQL Query:
""")

validate_prompt = PromptTemplate.from_template("""
You are a SQL validator. Given a user question, SQL query, and schema, confirm if the query makes sense.
Schema: {schema}
Question: {question}
SQL Query: {sql_query}
Does the query make sense? Return only Yes or No with reason.
""")

table_selector_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Based on the user question and the following schema, decide which table(s) are most relevant.
Schema:
{schema}
Question: {question}
Relevant Table(s):
""")

# ---- Step 3: LLM Setup ----
llm = Ollama(model="llama3", base_url="http://localhost:11434ddddgg")  # Modify as needed

# ---- Step 4: Dynamic Schema Fetcher ----
def get_schema(_):
    query = "SELECT table_name, column_name, column_type FROM cost_table_info ORDER BY table_name;"
    try:
        raw_csv = run_questdb_query(query)
        lines = raw_csv.strip().split("\n")
        headers = lines[0].split(",")
        schema_lines = lines[1:]

        schema_dict = {}
        for line in schema_lines:
            parts = line.split(",")
            table, column, dtype = parts
            if table not in schema_dict:
                schema_dict[table] = []
            schema_dict[table].append(f"{column} {dtype}")

        formatted_schema = "\n".join(
            f"{table}({', '.join(columns)});" for table, columns in schema_dict.items()
        )
        return formatted_schema
    except Exception as e:
        return f"Failed to retrieve schema: {e}"

# ---- Step 5: Tools ----

# Tool 1: Schema Inspector
schema_tool = Tool.from_function(
    func=get_schema,
    name="SchemaInspector",
    description="Returns the available tables and their schema from cost_table_info"
)

# Tool 2: Table Selector
def select_table(input: dict) -> str:
    return llm.invoke(table_selector_prompt.format(**input))

table_selector_tool = Tool.from_function(
    func=select_table,
    name="TableSelector",
    description="Selects the most relevant table(s) for a given question using schema information"
)

# Tool 3: SQL Generator
def generate_sql(input: dict) -> str:
    return llm.invoke(sql_prompt.format(**input))

sql_gen_tool = Tool.from_function(
    func=generate_sql,
    name="SQLGenerator",
    description="Generates SQL query from natural language and schema"
)

# Tool 4: SQL Validator
def validate_sql(input: dict) -> str:
    return llm.invoke(validate_prompt.format(**input))

validator_tool = Tool.from_function(
    func=validate_sql,
    name="SQLValidator",
    description="Validates the generated SQL query"
)

# Tool 5: SQL Executor
sql_exec_tool = Tool.from_function(
    func=run_questdb_query,
    name="SQLExecutor",
    description="Executes SQL query on QuestDB and returns results"
)

# ---- Step 6: Agent Setup ----
tools = [schema_tool, table_selector_tool, sql_gen_tool, validator_tool, sql_exec_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,  # You can switch to AgentType.REACT if preferred
    verbose=True
)

# ---- Step 7: End-to-End Pipeline Execution ----
if __name__ == "__main__":
    user_question = "Show me the names of students who scored more than 80"

    # 1. Get schema
    schema = get_schema(None)

    # 2. Select relevant table(s)
    selected_table = select_table({"schema": schema, "question": user_question})
    print("Selected Table(s):\n", selected_table)

    # 3. Generate SQL
    intermediate_sql = generate_sql({"schema": selected_table, "question": user_question})
    print("Generated SQL:\n", intermediate_sql)

    # 4. Validate SQL
    validation_result = validate_sql({
        "schema": selected_table,
        "question": user_question,
        "sql_query": intermediate_sql
    })
    print("Validation Result:\n", validation_result)

    # 5. Execute SQL if valid
    if "yes" in validation_result.lower():
        result = run_questdb_query(intermediate_sql)
        print("Query Result:\n", result)
    else:
        print("Query is invalid.")
