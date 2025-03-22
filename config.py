
from pydantic_ai import Agent

class Config:
    model_manager: str = "groq:llama-3.3-70b-versatile"
    model_sub_agent: str = "groq:llama-3.3-70b-versatile"

    filepath: str  = r"C:\Users\DELL\Desktop\Aimleap\pdf_extractor_agents\input\JPM - x1004 - Statement (1).pdf"