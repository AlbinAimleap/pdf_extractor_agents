
from pydantic_ai import Agent, RunContext
from extractor.schema import AccountSummary
from typing import List
from config import Config


agent = Agent(
    Config.model_sub_agent,
    result_type=AccountSummary,
    system_prompt="""
    You are a financial expert. Your task is to extract the account summary details from the financial statement,
    including Total Short Term Realized Gain/Loss, Total Long Term Realized Gain/Loss, and Total Realized Gain/Loss.
    """
)

@agent.tool
def account_summary_extraction(ctx: RunContext, query: str) -> List[str]:
    result = ctx.deps.search(query)
    return result
