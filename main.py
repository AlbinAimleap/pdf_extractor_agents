from extractor.agents.equity_agent import agent as equity_agent
from extractor.agents.account_summary_agent import agent as account_summary_agent
from extractor.agents.alternative_assets_detail_agent import agent as alternative_assets_agent
from extractor.agents.portfolio_activity_detail_agent import agent as portfolio_activity_agent
from extractor.agents.transactions_summary_agent import agent as transactions_summary_agent
from extractor.agents.fixed_income_agent import agent as fixed_income_agent
from extractor.agents.trade_activity_agent import agent as trade_activity_agent
from extractor.schema import EquityDetail, AccountSummary, AlternativeAssetDetailItem, PortfolioActivityDetailItem, TransactionsSummary, FixedIncomeItem, TradeActivityItem
from extractor.vector_db  import DocumentProcessor
from dataclasses import dataclass
import asyncio
import json
from typing  import Callable, List
from devtools import debug
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel
from config import Config
from extractor import logger
from pydantic_ai.usage import UsageLimits
from pathlib import Path


class FinalResult(BaseModel):
    equity_details: EquityDetail
    account_summary: AccountSummary
    alternative_assets: List[AlternativeAssetDetailItem]
    portfolio_activity: List[PortfolioActivityDetailItem]
    transactions_summary: List[TransactionsSummary]
    fixed_income: List[FixedIncomeItem]
    trade_activity: List[TradeActivityItem]


@dataclass
class Deps:
    file_path: str = Config.filepath
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

logger.info("Account Manager Agent Initialized")



@manager_agent.tool
async def equity_extraction(ctx: RunContext, query: str) -> EquityDetail:
    result = await equity_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def account_summary_extraction(ctx: RunContext, query: str) -> AccountSummary:
    result = await account_summary_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def alternative_assets_extraction(ctx: RunContext, query: str) -> List[AlternativeAssetDetailItem]:
    result = await alternative_assets_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def portfolio_activity_extraction(ctx: RunContext, query: str) -> List[PortfolioActivityDetailItem]:
    result = await portfolio_activity_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def transactions_summary_extraction(ctx: RunContext, query: str) -> List[TransactionsSummary]:
    result = await transactions_summary_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def fixed_income_extraction(ctx: RunContext, query: str) -> List[FixedIncomeItem]:
    result = await fixed_income_agent.run(query, deps=ctx.deps)
    return result.data

@manager_agent.tool
async def trade_activity_extraction(ctx: RunContext, query: str) -> List[TradeActivityItem]:
    result = await trade_activity_agent.run(query, deps=ctx.deps)
    return result.data


async def main():
    result = await manager_agent.run(
            """Please extract all available financial records from the statement including:
            - Equity details (names, tickers, prices, quantities, values, cost basis, unrealized gains/losses, estimated income, yields)
            - Account summary (short term and long term realized gains/losses, total realized gains/losses, unrealized gains/losses)
            - Alternative asset details (names, prices, quantities, estimated values, costs)
            - Transaction summaries (beginning and ending cash balances)
            - Portfolio activity details (settlement dates, types, descriptions, quantities, amounts, realized gains/losses)
            - Fixed income securities (names, maturity dates, coupon rates, prices, quantities, market values)
            - Trade activity (dates, transaction types, descriptions, quantities, prices, amounts)""", 
            deps=Deps(),
            result_type=FinalResult,
            usage_limits=UsageLimits(request_tokens_limit=10000,response_tokens_limit=5000)
    )

    debug(result)
    result = json.dumps(result.data.model_dump(), indent=4)
    print(result)
    return result


async def extractor_agent(file: str):
    filepath: Path = Path(file)
    result = await manager_agent.run(
            """Please extract all available financial records from the statement including:
            - Equity details (names, tickers, prices, quantities, values, cost basis, unrealized gains/losses, estimated income, yields)
            - Account summary (short term and long term realized gains/losses, total realized gains/losses, unrealized gains/losses)
            - Alternative asset details (names, prices, quantities, estimated values, costs)
            - Transaction summaries (beginning and ending cash balances)
            - Portfolio activity details (settlement dates, types, descriptions, quantities, amounts, realized gains/losses)
            - Fixed income securities (names, maturity dates, coupon rates, prices, quantities, market values)
            - Trade activity (dates, transaction types, descriptions, quantities, prices, amounts)""", 
            deps=Deps(file_path=filepath),
            result_type=FinalResult,
            usage_limits=UsageLimits(request_tokens_limit=10000,response_tokens_limit=5000)
    )
    result = result.data.model_dump()
    return result


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
