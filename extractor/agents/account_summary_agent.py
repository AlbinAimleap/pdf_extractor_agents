from pydantic_ai import Agent, RunContext, Tool
from extractor.schema import AccountSummary
from extractor.tools import vector_search
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=AccountSummary,
    system_prompt="""
    You are a financial expert specializing in account summary analysis. Your task is to accurately extract and structure account summary information from financial statements including:

    1. Gains/Losses:
    - Total Short Term Realized Gain/Loss
    - Total Long Term Realized Gain/Loss
    - Total Combined Realized Gain/Loss
    - Unrealized Gain/Loss
    - name of the account
    - account number
    - date of statement
    - name of custodian

    2. Additional Metrics (if available):
    - Beginning balance
    - Ending balance
    - Net change
    - Time period covered

    Ensure all numerical values are extracted precisely as they appear in the statement:
    - Maintain original format and units
    - Preserve decimal places
    - Correctly identify negative values
    - Handle any currency symbols or notations properly
    - Pay attention to subtotals and grand totals

    Format the extracted data according to the AccountSummary structure, ensuring all fields are populated with the correct corresponding values.
    """,
    tools=[
        Tool(
            vector_search,
            description="""
            Search for relevant financial statements based on the query.
            """,
            takes_ctx=True
        ),
    ]
)
logger.info("Account Summary Agent Initialized")


# @agent.tool
# def account_summary_extraction(ctx: RunContext, query: str, top_k: int) -> List[str]:
#     result = ctx.deps.search(query, top_k=top_k)
#     return result
