from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tools.budget_tool import estimate_budget_tool, TravelModel
from tools.weather_tool import weather_forecast_tool
from tools.rag_tool import get_rag_tool

load_dotenv()
app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for frontend input
class ChatRequest(BaseModel):
    message: str
    conversation_id: str = None

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

# Load agent logic
def get_integrated_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2)

    parser = PydanticOutputParser(pydantic_object=TravelModel)

    prompt = ChatPromptTemplate.from_messages([
       ("system", '''
You are a smart tourism assistant. Based on the user's query, perform one or more of the following tasks:

- If the user asks tourism or privacy policy-related questions, use the RAG tool.
- If the user needs a travel plan or cost estimate, call the budget tool and the weather tool.
Then:
• Give the average weather of that place.
• Generate a list of at least 3 hotel names matching the user's travel style (budget, midrange, luxury) and destination.
• Suggest 2-3 restaurants appropriate for the group (e.g., vegetarian, family-friendly, kid-friendly, upscale).
• Recommend activities and places to visit, covering all age groups in the user's group (children, elderly, adults).
• Include suggestions that account for children or elderly travelers when relevant.
• Provide a rough day-by-day travel timeline.
• Include emergency numbers (e.g., police, ambulance).
• Always show all prices in Indian Rupees (₹), even if the destination is international.

Your goal is to populate the TravelModel fields:
- hotels: hotel name strings
- restaurants: restaurant name strings
- activities: activity name strings
- places_to_visit: tourist attraction strings
        
Always return a complete and helpful plan in structured format.
'''),
        ("user", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    tools = [estimate_budget_tool, weather_forecast_tool, get_rag_tool()]
    agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
    return AgentExecutor(agent=agent, tools=tools)

agent_executor = get_integrated_agent()

# Chat endpoint for frontend
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = agent_executor.invoke({"query": request.message})
        return {"response": result["output"]}
    except Exception as e:
        print("Error:", e) #for showing error, (on terminal.)
        return {"response": "Sorry, something went wrong."}
