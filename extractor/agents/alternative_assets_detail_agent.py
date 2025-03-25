from pydantic_ai import Agent, RunContext, Tool
from extractor.schema import AlternativeAssetDetailItem
from extractor.tools import vector_search
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

logger.info("Account Alternative Asset Initialized")

# @agent.tool
# def alternative_assets_extraction(ctx: RunContext, query: str, top_k: int) -> List[str]:
#    result = ctx.deps.search(query, top_k=top_k)
#    return result
