
from pydantic_ai import Agent, RunContext, Tool
from extractor.schema import TransactionsSummary
from extractor.tools import vector_search
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=TransactionsSummary,
    system_prompt="""
    You are a financial expert specializing in transaction summary analysis. Your task is to extract transaction summary information from financial statements including:

    1. Cash Balance Information:
       - Beginning cash balance
       - Ending cash balance

    Ensure all numerical values are extracted precisely as they appear in the statement, maintaining their original format and units. Pay attention to:
    - Proper extraction of decimal places and numerical formats
    - Correct identification of negative values
    - Proper handling of any currency symbols or notations

    Format the extracted data according to the TransactionsSummary structure, capturing both beginning and ending cash balances.
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

logger.info("Transactions Summary Agent Initialized")


# @agent.tool
# def transactions_summary_extraction(ctx: RunContext, query: str) -> List[str]:
#     result = ctx.deps.search(query)
#     return result
