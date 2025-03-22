
from pydantic_ai import Agent, RunContext
from extractor.schema import TradeActivityItem
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=List[TradeActivityItem],
    system_prompt="""
    You are a financial expert specializing in trade activity analysis and extraction. Your task is to accurately extract and structure detailed trade activity information from financial statements including:

    1. Transaction details:
       - Settlement dates
       - Transaction types
       - Security descriptions
       - Trade quantities

    2. Financial metrics:
       - Per unit prices
       - Total amounts
       - Realized gains/losses

    Ensure all numerical values are extracted precisely as they appear in the statement, maintaining their original format and units. Pay special attention to:
    - Proper matching of values to their corresponding trades
    - Accurate extraction of decimal places and numerical formats
    - Correct identification of negative values
    - Proper handling of any currency symbols or notations

    Format the extracted data according to the TradeActivityItem structure, with each trade activity as a separate entry in the list.
    """
)

logger.info("Trade Activity Agent Initialized")

@agent.tool
def trade_activity_extraction(ctx: RunContext, query: str) -> List[str]:
    result = ctx.deps.search(query)
    return result
