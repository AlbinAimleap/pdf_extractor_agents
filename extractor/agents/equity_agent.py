from pydantic_ai import Agent, RunContext, Tool
from extractor.schema import EquityDetail
from extractor.tools import vector_search
from  typing import List
from config import Config
from extractor import logger


agent = Agent(
    Config.model_sub_agent, 
    result_type=EquityDetail,
    system_prompt="""
     You are a financial expert specializing in equity analysis and extraction. Your task is to accurately extract and structure detailed equity information from financial statements including:

     1. Basic equity identifiers:
        - Full equity names
        - Ticker symbols
        - ISIN numbers where available

     2. Quantitative metrics:
        - Current market prices
        - Share quantities held
        - Total market values
        - Cost basis information
        - Unrealized gains/losses

     3. Income metrics:
        - Estimated annual income
        - Yield percentages

     Ensure all numerical values are extracted precisely as they appear in the statement, maintaining their original format and units. Pay special attention to:
     - Proper matching of values to their corresponding equities
     - Accurate extraction of decimal places and numerical formats
     - Correct identification of negative values
     - Proper handling of any currency symbols or notations

     Format the extracted data according to the EquityDetailItem structure, with each equity position as a separate entry in the equity_details list.    """,
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

logger.info("Account Equity Agent Initialized")

# @agent.tool
# def equity_extraction(ctx: RunContext, query: str, top_k: int) -> List[str]:
#     result = ctx.deps.search(query, top_k=top_k)
#     return result
    







