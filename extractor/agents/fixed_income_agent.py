
from pydantic_ai import Agent, RunContext, Tool
from extractor.schema import FixedIncomeItem
from extractor.tools import vector_search
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=List[FixedIncomeItem],
    system_prompt="""
    You are a financial expert specializing in fixed income analysis and extraction. Your task is to accurately extract and structure detailed fixed income information from financial statements including:

    1. Security details:
        - Security names
        - Maturity dates
        - Coupon rates

    2. Position metrics:
        - Current prices
        - Quantities held
        - Market values
        - Unrealized gains/losses

    Ensure all numerical values are extracted precisely as they appear in the statement, maintaining their original format and units. Pay special attention to:
    - Proper matching of values to their corresponding securities
    - Accurate extraction of decimal places and numerical formats
    - Correct identification of negative values
    - Proper handling of any currency symbols or notations

    Format the extracted data according to the FixedIncomeItem structure, with each fixed income position as a separate entry in the list.
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

logger.info("Fixed Income Agent Initialized")

# @agent.tool
# def fixed_income_extraction(ctx: RunContext, query: str, top_k: int) -> List[str]:
#     result = ctx.deps.search(query)
#     return result
