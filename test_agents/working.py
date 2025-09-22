from phi.agent import Agent
from phi.tools.postgres import PostgresTools
from phi.model.ollama import Ollama
from phi.model.openai import  OpenAIChat
from phi.storage.agent.postgres import PgAgentStorage



# Initialize PostgresTools with your database connection
postgres_tools = PostgresTools(
    host="localhost",
    port=5432,
    db_name="parser",
    user="parser",
    password="parser123"
)

# Define detailed role/context for the agent
role_description = """
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

Your responsibilities:
1. Understand and execute all kinds of queries: SELECT, INSERT, UPDATE, DELETE, JOINs, aggregation, filtering, sorting, and grouping.
2. Be aware of data types and constraints, including unique constraints, foreign keys, and defaults.
3. Help create, update, and delete both orders and line_items while respecting constraints.
4. Support reporting queries like total sales by buyer, supplier, or date ranges.
5. you are a database assistant that can do all CRUD and analysis on orders and line_items tables

show the final results only
"""

db_agent = Agent(
    model = OpenAIChat(
    model="gpt-4.1",
    api_key="api key",  # hardcoded
    temperature=0.2
    ),
    tools=[postgres_tools],
    description=role_description
)




while True:
    user_input = input("Enter your query: ")
    if user_input.lower() == 'exit':
        break
    
    response = db_agent.print_response(user_input)





