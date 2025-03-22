
from pydantic_ai import Agent, RunContext
from extractor.schema import PortfolioActivityDetailItem
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=List[PortfolioActivityDetailItem],
    system_prompt="""
    You are a financial expert specializing in portfolio activity analysis. Your task is to extract detailed portfolio activity information from financial statements including:

    1. Transaction details:
        - Settlement dates
        - Type and selection methods
        - Transaction descriptions
        - Quantities involved

    2. Financial metrics:
        - Per unit amounts
        - Total transaction amounts
        - Realized gains/losses

    Ensure all data is extracted precisely as it appears in the statement, maintaining:
    - Original date formats
    - Exact numerical values and decimal places
    - Proper identification of transaction types
    - Accurate matching of amounts to their corresponding transactions
    - Correct handling of positive and negative values
    - Any currency symbols or notations

    Format the extracted data according to the PortfolioActivityDetailItem structure, with each activity as a separate entry in the list.
    """
)

logger.info("Portfolio Activity Detail Agent Initialized")

@agent.tool
def portfolio_activity_extraction(ctx: RunContext, query: str) -> List[str]:
    result = ctx.deps.search(query)
    return result
