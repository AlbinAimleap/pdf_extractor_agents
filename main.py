from extractor.agents.equity_agent import agent as equity_agent
from extractor.agents.account_summary_agent import agent as account_summary_agent
from extractor.schema import EquityDetail, AccountSummary
from extractor.vector_db  import DocumentProcessor
from dataclasses import dataclass
import asyncio
import json
from typing  import Callable

from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from config import Config


class FinalResult(BaseModel):
    equity_details: EquityDetail
    account_summary: AccountSummary


@dataclass
class Deps:
    file_path: str = r"C:\Users\Albia\Desktop\Aimleap\pdf_extraction\pydantic_agents_flow\input_files\JPM - x1004 - Statement (1).pdf"
    doc_processor: DocumentProcessor = DocumentProcessor()
    search: Callable = doc_processor.vector_db.search

    def __post_init__(self):
        self.doc_processor.process_document(self.file_path)
        
              
manager_agent = Agent(
    Config.model_manager,
    system_prompt="""
    You are a highly skilled financial analyst specializing in investment statement analysis. Your expertise lies in:
    1. Extracting and interpreting detailed equity information including market values, cost basis, and performance metrics
    2. Analyzing account summaries with focus on realized and unrealized gains/losses
    3. Evaluating alternative investments and their performance metrics
    4. Understanding complex transaction histories and portfolio activities
    5. Interpreting fixed income securities and their key characteristics
    6. Analyzing trading activities and their impact on portfolio performance

    Your task is to thoroughly extract and analyze all financial records from the provided statement, ensuring accuracy and completeness in the data extraction process. Pay special attention to numerical values, dates, and financial metrics.
    """
)


@manager_agent.tool
async def equity_extraction(ctx: RunContext, query: str) -> EquityDetail:
    result = await equity_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def account_summary_extraction(ctx: RunContext, query: str) -> AccountSummary:
    result = await account_summary_agent.run(query, deps=ctx.deps)
    return result.data



async def main():
    result = await manager_agent.run("""Please extract all available financial records from the statement including:
    - Equity details (names, tickers, prices, quantities, values, cost basis, unrealized gains/losses, estimated income, yields)
    - Account summary (short term and long term realized gains/losses, total realized gains/losses, unrealized gains/losses)
    - Alternative asset details (names, prices, quantities, estimated values, costs)
    - Transaction summaries (beginning and ending cash balances)
    - Portfolio activity details (settlement dates, types, descriptions, quantities, amounts, realized gains/losses)
    - Fixed income securities (names, maturity dates, coupon rates, prices, quantities, market values)
    - Trade activity (dates, transaction types, descriptions, quantities, prices, amounts)""", 
            deps=Deps(),
            result_type=FinalResult
    )
    
    result = json.dumps(result.data.model_dump(), indent=4)
    print(result)
    
if __name__ == "__main__":
    asyncio.run(main())