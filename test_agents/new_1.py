from phi.agent import Agent
from phi.tools.postgres import PostgresTools
from phi.model.openai import OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
import uuid
import time

# Database connections
postgres_tools = PostgresTools(
    host="localhost", port=5432, db_name="parser", user="parser", password="parser123"
)

chat_tools = PostgresTools(
    host="localhost", port=6000, db_name="ai", user="ai", password="ai"
)

chat_url = "postgresql+psycopg://ai:ai@localhost:6000/ai"
storage = PgAgentStorage(table_name="chat_sessions", db_url=chat_url)

# Agent Roles
db_role = """
You are a smart database assistant for orders and line_items table. 
Execute queries safely, respect constraints, and return final results only.
"""

chat_role = """
You are a conversational agent. Store each message with order_id, default '000' for general chat.
Retrieve past conversation only for the given order_id for context.
"""

# Agents
db_agent = Agent(
    model=OpenAIChat(model="gpt-4.1", api_key="api key", temperature=0.2),
    tools=[postgres_tools],
    role=db_role
)

chat_agent = Agent(
    model=OpenAIChat(model="gpt-4.1", api_key="api key", temperature=0.2),
    tools=[chat_tools],
    storage=storage,
    add_history_to_messages=True,
    role=chat_role,
    read_chat_history=True,
    search_knowledge=True,
    update_knowledge=True,
    read_tool_call_history=True
)

agent_orchestrator = Agent(
    name="order_management_agent",
    model=OpenAIChat(model="gpt-4.1", 
    api_key="api key",   temperature=0.2),
    team=[db_agent, chat_agent],
    instructions=[
        "Step 1: Identify order_id in user query (default '000' if general).",
        "Step 2: Extract orders data via db_agent.",
        "Step 3: Retrieve past conversation from chat_agent for the same order_id.",
        "Step 4: Respond and store each message in conversation_data with order_id."
    ],
    show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,
    read_chat_history=True,
    search_knowledge=True,
    update_knowledge=True,
    read_tool_call_history=True
)

def store_conversation(order_id, user_name, message):
    query = """
    INSERT INTO conversation_data(order_id, user_name, user_info)
    VALUES (%s, %s, %s)
    """
    chat_tools.execute(query, (order_id, user_name, message))

def get_order_context(order_id):
    query = """
    SELECT user_info FROM conversation_data
    WHERE order_id = %s
    ORDER BY created_at
    """
    result = chat_tools.fetch_all(query, (order_id,))
    return "\n".join([r[0] for r in result]) if result else ""

while True:
    user_input = input("Enter your query: ")
   
    # Get agent response
    response = agent_orchestrator.print_response(user_input)