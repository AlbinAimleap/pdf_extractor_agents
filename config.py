
from pydantic_ai import Agent

class Config:
    model_manager: str = "openai:gpt-4o"
    model_sub_agent: str = "openai:gpt-4o-mini"

    filepath: str  = r"C:\Users\Albia\Desktop\Aimleap\pdf_extraction\pydantic_agents_flow\input_files\JPM - x1004 - Statement (1).pdf"