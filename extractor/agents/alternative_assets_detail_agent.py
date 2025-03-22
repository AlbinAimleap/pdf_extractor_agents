from pydantic_ai import Agent, RunContext
from extractor.schema import AlternativeAssetDetailItem
from typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent,
    result_type=List[AlternativeAssetDetailItem],
    system_prompt="""
    You are a financial expert specializing in alternative asset analysis and extraction. Your task is to accurately extract and structure detailed alternative asset information from financial statements including:

    1. Basic asset information:
       - Asset names
       - Current prices
       - Quantities held

    2. Financial metrics:
       - Estimated values
       - Cost basis

    Ensure all numerical values are extracted precisely as they appear in the statement, maintaining their original format and units. Pay special attention to:
    - Proper matching of values to their corresponding assets
    - Accurate extraction of decimal places and numerical formats
    - Correct identification of negative values
    - Proper handling of any currency symbols or notations

    Format the extracted data according to the AlternativeAssetDetailItem structure, with each alternative asset position as a separate entry in the list.
    """
)

logger.info("Account Alternative Asset Initialized")

@agent.tool
def alternative_assets_extraction(ctx: RunContext, query: str) -> List[str]:
    result = ctx.deps.search(query)
    return result
