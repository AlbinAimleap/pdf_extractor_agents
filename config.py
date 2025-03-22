
from pydantic_ai import Agent

class Config:
    model_manager: str = "groq:llama-3.3-70b-versatile"
    model_sub_agent: str = "groq:llama-3.1-8b-instant"