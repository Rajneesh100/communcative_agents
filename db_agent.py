from phi.agent import Agent
from phi.tools.postgres import PostgresTools
from phi.model.ollama import Ollama
from phi.model.openai import  OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage
# from phi.tools.python import PythonTool


# Initialize PostgresTools with your database connection
postgres_tools = PostgresTools(
    host="localhost",
    port=5432,
    db_name="parser",
    user="parser",
    password="parser123"
)

chat_tools = PostgresTools(
    host="localhost",
    port=6000,
    db_name="ai",
    user="ai",
    password="ai"
)



# Define detailed role/context for the agent
db_role = """
You are a highly intelligent and reliable database assistant with full knowledge of the following PostgreSQL database:

Tables:
1. orders
   - id (uuid, PK)
   - purchase_order_id (varchar, unique)
   - order_date (date)
   - buyer_name (varchar)
   - buyer_address (text)
   - supplier_name (varchar)
   - supplier_address (text)
   - currency (varchar, default 'USD')
   - tax_amount (numeric)
   - total_amount (numeric)
   - created_at (timestamp)
   - updated_at (timestamp)
   - completed (boolean, default false)
   
2. line_items
   - id (uuid, PK)
   - order_id (uuid, FK -> orders.id)
   - model_id (varchar)
   - item_code (varchar)
   - description (text)
   - color (varchar)
   - size (varchar)
   - quantity (integer)
   - unit_price (numeric)
   - amount (numeric)
   - delivery_date (date)
   - created_at (timestamp)

Relationships:
- orders.id is referenced by line_items.order_id (ON DELETE CASCADE)
- The database uses standard indexes on relevant columns for fast lookups

for extracting the item details from given order you need to have a join operation on line_items.order_id = orders.id. when user says order_id or purchase_order_id he means the purchase_order_id only

Your responsibilities:
1. Understand and execute all kinds of queries: SELECT, INSERT, UPDATE, DELETE, JOINs, aggregation, filtering, sorting, and grouping.
2. Be aware of data types and constraints, including unique constraints, foreign keys, and defaults.
3. Help create, update, and delete both orders and line_items while respecting constraints.
4. Support reporting queries like total sales by buyer, supplier, or date ranges.
5. you are a database assistant that can do all CRUD and analysis on orders and line_items tables

show the final results only
"""


chat_role ="""
the conversation tabels :
1. chat_sessions
   - session_id (varchar, PK)
   - agent_id (varchar)
   - user_id (varchar)
   - memory (jsonb)
   - agent_data (jsonb)
   - user_data (jsonb)
   - session_data (jsonb)
   - created_at (bigint, default EXTRACT(epoch FROM now())::bigint)
   - updated_at (bigint)

2. conversation_data
   - id (integer, PK)
   - order_id (varchar)
   - user_name (varchar)
   - user_info (text)
   - created_at (timestamp, default CURRENT_TIMESTAMP)


extract data from past conversation and update knowledge base as conversation goes on for future refrance, 
each conversation realted to orders must have bind to order_id while storing to pgvector db, it is a must do. 
and while searching for past conversation for order related stuff it must use only those conversation which is related to that order_id only, 
so it doesn't mix up context. for any message not related to orders please store it some 000 order_id meaning it not realated to orders conversation

"""
db_agent = Agent(
    model = OpenAIChat(
    model="gpt-4.1",
    api_key="api key",  # hardcoded
    temperature=0.2
    ),
    tools=[postgres_tools],
    # description=db_description
    role= db_role
)


chat_url = "postgresql+psycopg://ai:ai@localhost:6000/ai"

storage = PgAgentStorage(
    table_name="chat_sessions",  # Table to store chat sessions
    db_url=chat_url
)

chat_agent = Agent(
    model = OpenAIChat(
        model="gpt-4.1",
        api_key="api key",  # hardcoded
        temperature=0.2
    ),
    tools=[chat_tools],
    storage=storage,
    add_history_to_messages=True,  # Include chat history in responses
    # description=chat_discryption,
    role= chat_role,
    read_chat_history = True,
    search_knowledge = True,
    update_knowledge = True,
    
    read_tool_call_history = True
)




agent_orchestrator = Agent(
    name="order_management_agent",
    model = OpenAIChat(
        model="gpt-4.1",
        api_key="api key",  # hardcoded
        temperature=0.2
    ),
    team=[db_agent, chat_agent],
    instructions=[
        # "Step 1: understand the user query and break into 2 parts: 1) extract orders_data using db_agent,  2)extract past conversation data using chat_agent realted to that order if needed",
        # "Step 2: if user is asking something specific related to some order ask him spefic order id and suggest him like would you like to see latest orders in last 10 days or anything user friendly",
        # "Step 3: use db_agent to extract the data from db, for applying filters convert user input logically to respective columns",
        # "Step 4: Identify the need to fetch data from past conversation using chat_agent using chat_sessions and conversation_data tables",
        # "Step 5: store each chat conversation to the db, make sure the chat_agent store each conversation with order_id ( if it is not related to order keep it 000)"
        "Step 1: Analyze the user query and determine if it relates to a specific order.",
        "Step 2: If an order is mentioned, extract the order_id; if not, use '000' as default.",
        "Step 3: Use db_agent to query or update orders and line_items tables as needed, respecting constraints and data types.",
        "Step 4: Use chat_agent to fetch past conversation only for the given order_id to provide context.",
        "Step 5: Include past conversation context in the response if relevant.",
        "Step 6: Store every user message and agent response in conversation_data with the correct order_id.",
        "Step 7: Only return final results to the user; do not include SQL queries in the output.",
        "Step 8: Keep all order-specific conversations separate; do not mix contexts between different orders."
    ],
    show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,  # Include chat history in responses
    read_chat_history = True,
    search_knowledge = True,
    update_knowledge = True,
    
    read_tool_call_history = True
)

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == 'exit':
        break
    response = agent_orchestrator.print_response(user_input)
