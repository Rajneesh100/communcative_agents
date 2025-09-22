from phi.agent import Agent
from phi.model.ollama import Ollama
from phi.tools.python import PythonTools
from phi.tools.googlesearch import GoogleSearch
from phi.tools.crawl4ai_tools import Crawl4aiTools

# 1. Agent to understand user travel needs
trip_planner = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="Trip Planner",
    role="Take the user's location, dates, budget, and preferences like climate and generate a travel objective and constraints.",
    show_tool_calls=True,
)

# 2. Agent to find suitable destinations in South India
destination_finder = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="Destination Finder",
    role="Search for cool temperature destinations in South India that fit the user's preferences using GoogleSearch.",
    tools=[GoogleSearch()],
    show_tool_calls=True
)

# 3. Agent to fetch travel (bus/train/flight) options
transport_agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="Travel Booker",
    role="Search for cheap transport options from user's city to the destination using online sources. Prioritize buses and trains under budget.",
    tools=[GoogleSearch(), Crawl4aiTools(), PythonTools()],
    show_tool_calls=True
)

# 4. Agent to find accommodation and stay options
stay_agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="Stay Finder",
    role="Find cheap and clean accommodation options like hotels, hostels or homestays at the destination under budget.",
    tools=[GoogleSearch()],
    show_tool_calls=True
)

# 5. Agent to prepare full itinerary with activities and bookings
itinerary_builder = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="Itinerary Planner",
    role="Based on destination, budget, transport and stay details, plan a full 5-day itinerary with activities, places to visit, estimated costs, and booking links.",
    tools=[Crawl4aiTools()],
    show_tool_calls=True
)

# Final team agent
travel_guide_team = Agent(
    model=Ollama(id="llama3.2:latest"),
    name="South India Travel Assistant",
    team=[trip_planner, destination_finder, transport_agent, stay_agent, itinerary_builder],
    instructions=[
        "Step 1: Understand user's travel dates, location, climate preference and budget.",
        "Step 2: Identify 2-3 best travel destinations with cool weather in South India during the dates.",
        "Step 3: Find the cheapest and safest transportation options from user's location.",
        "Step 4: Find budget-friendly and reviewed accommodation options.",
        "Step 5: Plan a day-by-day itinerary with activities, food, sightseeing.",
        "Step 6: Scrape or suggest where user can book buses/flights/hotels online.",
        "Step 7: Ensure full plan is under user's budget (₹20000).",
        "Step 8: Validate all findings and give user a neat travel plan with actionable links or screenshots if available.",
    ],
    show_tool_calls=True,
    markdown=True
)

# Execution loop
with open("travel_responses.txt", "a") as file:
    while True:
        user_input = input("Enter your travel query (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
        
        response = travel_guide_team.print_response(user_input)
        file.write(f"Query: {user_input}\nResponse:\n{response}\n\n")
        print("Plan saved to file.")
