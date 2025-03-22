from pydantic import BaseModel, Field
from typing import Optional, List

class EquityDetailItem(BaseModel):
    equity_name: Optional[str] = None
    ticker: Optional[str] = None
    isin: Optional[str] = None
    price: Optional[str] = None
    quantity: Optional[str] = None
    value: Optional[str] = None
    cost_basis: Optional[str] = None
    unrealized_gain_loss: Optional[str] = None
    estimated_annual_income: Optional[str] = None
    _yield: Optional[str] = None

class EquityDetail(BaseModel):
    equity_details: List[EquityDetailItem] = Field(default_factory=list)

class AlternativeAssetDetailItem(BaseModel):
    name: Optional[str] = None
    price: Optional[str] = None
    quantity: Optional[str] = None
    estimated_value: Optional[str] = None
    cost: Optional[str] = None

class TransactionsSummary(BaseModel):
    beginning_cash_balance: Optional[str] = None
    ending_cash_balance: Optional[str] = None

class PortfolioActivityDetailItem(BaseModel):
    settle_date: Optional[str] = None
    type_selection_method: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[str] = None
    per_unit_amount: Optional[str] = None
    amount: Optional[str] = None
    realized_gain_loss: Optional[str] = None

class AccountSummary(BaseModel):
    total_st_realized_gain_loss: float
    total_lt_realized_gain_loss: float
    total_realized_gain_loss: float
    unrealized_gain_loss: float

class FixedIncomeItem(BaseModel):
    security_name: Optional[str] = None
    maturity_date: Optional[str] = None
    coupon_rate: Optional[str] = None
    price: Optional[str] = None
    quantity: Optional[str] = None
    market_value: Optional[str] = None
    unrealized_gain_loss: Optional[str] = None

class TradeActivityItem(BaseModel):
    settle_date: Optional[str] = None
    transaction_type: Optional[str] = None
    description: Optional[str] = None
    quantity: Optional[str] = None
    per_unit_price: Optional[str] = None
    amount: Optional[str] = None
    realized_gain_loss: Optional[str] = None

class FinancialStatement(BaseModel):
    name_of_account: str
    account_number: str
    date_of_statement: str
    account_summary: AccountSummary
    name_of_the_custodian: Optional[str] = None
    unrealized_gain_loss_total: Optional[str] = None
    equity_detail: Optional[List[EquityDetailItem]] = None
    alternative_assets_detail: Optional[List[AlternativeAssetDetailItem]] = None
    transactions_summary: Optional[TransactionsSummary] = None
    portfolio_activity_detail: Optional[List[PortfolioActivityDetailItem]] = None
    fixed_income: Optional[List[FixedIncomeItem]] = None
    trade_activity: Optional[List[TradeActivityItem]] = None