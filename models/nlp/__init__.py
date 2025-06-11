"""
NLP models for financial analysis and explanation.

This package contains natural language processing models for:
- Financial jargon translation to simple terms
- News sentiment analysis
- Market analysis explanation
- Real-time financial data interpretation
"""

from .financial_explainer import FinancialExplainer
from .news_analyzer import NewsAnalyzer
from .market_interpreter import MarketInterpreter

__all__ = ['FinancialExplainer', 'NewsAnalyzer', 'MarketInterpreter']