"""
Financial News Analyzer - Analyzes financial news sentiment and explains market impact.
"""

import re
import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


@dataclass
class NewsAnalysis:
    """Result of news sentiment analysis."""
    sentiment_score: float  # -1 to 1
    sentiment_label: str    # Positive, Negative, Neutral
    key_topics: List[str]
    market_impact: str
    summary: str
    confidence: float


@dataclass
class NewsArticle:
    """Represents a financial news article."""
    title: str
    content: str
    source: str
    published_date: datetime
    url: Optional[str] = None


class NewsAnalyzer:
    """
    Analyzes financial news for sentiment and market impact.
    Provides simple explanations of how news might affect investments.
    """
    
    def __init__(self):
        """Initialize the news analyzer."""
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        self.financial_keywords = self._load_financial_keywords()
        self.impact_patterns = self._load_impact_patterns()
        
    def _load_positive_words(self) -> List[str]:
        """Load positive financial sentiment words."""
        return [
            'profit', 'profits', 'profitable', 'gain', 'gains', 'growth', 'growing',
            'increase', 'increases', 'rising', 'rise', 'up', 'surge', 'surged',
            'boost', 'strong', 'stronger', 'solid', 'robust', 'healthy',
            'positive', 'optimistic', 'bullish', 'outperform', 'beat', 'exceed',
            'success', 'successful', 'record', 'high', 'peak', 'improved',
            'improving', 'expansion', 'expand', 'momentum', 'rally', 'soar',
            'breakthrough', 'achievement', 'milestone', 'upgrade', 'upgraded',
            'buy', 'recommend', 'favorable', 'excellence', 'excellent'
        ]
    
    def _load_negative_words(self) -> List[str]:
        """Load negative financial sentiment words."""
        return [
            'loss', 'losses', 'lose', 'losing', 'decline', 'declining', 'fall',
            'falling', 'drop', 'dropped', 'plunge', 'plunged', 'crash', 'crashed',
            'weak', 'weaker', 'poor', 'disappointing', 'missed', 'miss',
            'concern', 'concerns', 'worried', 'worry', 'fear', 'fears',
            'negative', 'bearish', 'pessimistic', 'underperform', 'struggle',
            'struggling', 'challenge', 'challenges', 'difficult', 'pressure',
            'pressured', 'downgrade', 'downgraded', 'sell', 'avoid',
            'recession', 'crisis', 'bankruptcy', 'debt', 'deficit', 'risk',
            'risks', 'volatility', 'uncertain', 'uncertainty', 'slowdown'
        ]
    
    def _load_financial_keywords(self) -> Dict[str, List[str]]:
        """Load financial keywords by category."""
        return {
            'earnings': [
                'earnings', 'revenue', 'sales', 'income', 'profit', 'eps',
                'earnings per share', 'quarterly results', 'financial results'
            ],
            'market': [
                'market', 'stock', 'stocks', 'shares', 'trading', 'volume',
                'price', 'valuation', 'market cap', 'index', 'dow', 'nasdaq', 's&p'
            ],
            'economy': [
                'economy', 'economic', 'gdp', 'inflation', 'unemployment',
                'fed', 'federal reserve', 'interest rates', 'monetary policy'
            ],
            'company': [
                'company', 'corporation', 'business', 'management', 'ceo',
                'executive', 'board', 'strategy', 'operations', 'merger',
                'acquisition', 'ipo', 'dividend'
            ],
            'sector': [
                'technology', 'healthcare', 'finance', 'energy', 'real estate',
                'consumer', 'industrial', 'materials', 'utilities', 'telecom'
            ]
        }
    
    def _load_impact_patterns(self) -> Dict[str, str]:
        """Load patterns that indicate market impact."""
        return {
            'earnings_beat': 'Companies beating earnings expectations often see stock price increases',
            'earnings_miss': 'Missing earnings expectations typically leads to stock price declines',
            'fed_rate_cut': 'Federal Reserve rate cuts usually boost stock markets',
            'fed_rate_hike': 'Federal Reserve rate increases often pressure stock markets',
            'merger_announcement': 'Merger announcements usually increase target company stock prices',
            'product_launch': 'Major product launches can boost company stock if well-received',
            'leadership_change': 'CEO changes can create uncertainty and stock volatility',
            'regulatory_approval': 'Regulatory approvals often lead to positive stock reactions',
            'recession_warning': 'Recession warnings typically cause broad market declines'
        }
    
    def analyze_news_sentiment(self, articles: List[NewsArticle]) -> NewsAnalysis:
        """
        Analyze sentiment of financial news articles.
        
        Args:
            articles: List of news articles to analyze
            
        Returns:
            NewsAnalysis with sentiment and impact assessment
        """
        if not articles:
            return NewsAnalysis(
                sentiment_score=0.0,
                sentiment_label="Neutral",
                key_topics=[],
                market_impact="No news to analyze",
                summary="No financial news provided for analysis.",
                confidence=0.0
            )
        
        # Combine all article text
        combined_text = " ".join([
            f"{article.title} {article.content}" for article in articles
        ])
        
        # Calculate sentiment score
        sentiment_score = self._calculate_sentiment_score(combined_text)
        sentiment_label = self._get_sentiment_label(sentiment_score)
        
        # Extract key topics
        key_topics = self._extract_key_topics(combined_text)
        
        # Assess market impact
        market_impact = self._assess_market_impact(combined_text, sentiment_score)
        
        # Generate summary
        summary = self._generate_news_summary(articles, sentiment_score, key_topics)
        
        # Calculate confidence
        confidence = self._calculate_confidence(combined_text, len(articles))
        
        return NewsAnalysis(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            key_topics=key_topics,
            market_impact=market_impact,
            summary=summary,
            confidence=confidence
        )
    
    def explain_news_impact(self, news_analysis: NewsAnalysis) -> str:
        """
        Explain how the news might impact investments in simple terms.
        
        Args:
            news_analysis: Results from news sentiment analysis
            
        Returns:
            Simple explanation of potential market impact
        """
        explanations = []
        
        # Sentiment explanation
        if news_analysis.sentiment_score > 0.3:
            explanations.append(
                "The overall news sentiment is positive, which typically means "
                "investors are feeling good about the market. This often leads to "
                "stock prices going up as more people want to buy."
            )
        elif news_analysis.sentiment_score < -0.3:
            explanations.append(
                "The overall news sentiment is negative, which usually means "
                "investors are worried. This often leads to stock prices going down "
                "as more people want to sell than buy."
            )
        else:
            explanations.append(
                "The news sentiment is neutral, suggesting mixed signals. "
                "Markets might not have a strong reaction either way."
            )
        
        # Topic-specific impacts
        if news_analysis.key_topics:
            explanations.append("Key topics that could affect your investments:")
            for topic in news_analysis.key_topics[:3]:  # Limit to top 3
                if 'earnings' in topic:
                    explanations.append(
                        "• Earnings news: Companies reporting profits usually see stock increases, "
                        "while losses often lead to decreases."
                    )
                elif 'fed' in topic or 'interest' in topic:
                    explanations.append(
                        "• Interest rate news: Lower rates are usually good for stocks, "
                        "higher rates can be challenging."
                    )
                elif 'merger' in topic or 'acquisition' in topic:
                    explanations.append(
                        "• Merger news: Target companies often see stock price jumps, "
                        "while acquiring companies might see mixed reactions."
                    )
        
        # Market impact explanation
        explanations.append(news_analysis.market_impact)
        
        # Confidence note
        if news_analysis.confidence < 0.5:
            explanations.append(
                "Note: The confidence in this analysis is moderate because "
                "the news signals are mixed or limited. Be cautious with decisions."
            )
        
        return " ".join(explanations)
    
    def get_sample_news(self) -> List[NewsArticle]:
        """Get sample financial news for demonstration."""
        sample_articles = [
            NewsArticle(
                title="Tech Giants Report Strong Quarterly Earnings",
                content="Major technology companies exceeded earnings expectations this quarter, "
                        "with Apple, Microsoft, and Google all posting record profits. "
                        "Strong consumer demand and cloud services growth drove the results.",
                source="Financial Times",
                published_date=datetime.now() - timedelta(hours=2)
            ),
            NewsArticle(
                title="Federal Reserve Signals Potential Rate Cut",
                content="The Federal Reserve hinted at possible interest rate cuts in response "
                        "to slowing economic growth. Chairman Powell emphasized the Fed's "
                        "commitment to supporting economic stability.",
                source="Reuters",
                published_date=datetime.now() - timedelta(hours=5)
            ),
            NewsArticle(
                title="Energy Sector Faces Volatility Amid Supply Concerns",
                content="Oil prices fluctuated as geopolitical tensions raised supply concerns. "
                        "Energy companies are closely monitoring the situation while maintaining "
                        "production levels.",
                source="Bloomberg",
                published_date=datetime.now() - timedelta(hours=8)
            )
        ]
        return sample_articles
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score from -1 (negative) to 1 (positive)."""
        text_lower = text.lower()
        
        # Count positive and negative words
        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)
        
        # Weight by word importance (financial terms get higher weight)
        positive_weight = positive_count
        negative_weight = negative_count
        
        # Calculate sentiment score
        total_words = len(text_lower.split())
        if total_words == 0:
            return 0.0
        
        # Normalize by text length and apply sigmoid function
        raw_score = (positive_weight - negative_weight) / max(total_words * 0.1, 1)
        
        # Apply sigmoid to bound between -1 and 1
        sentiment_score = 2 / (1 + np.exp(-5 * raw_score)) - 1
        
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.2:
            return "Positive"
        elif score < -0.2:
            return "Negative"
        else:
            return "Neutral"
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key financial topics from text."""
        text_lower = text.lower()
        topics = []
        
        for category, keywords in self.financial_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count > 0:
                topics.append(f"{category} ({keyword_count} mentions)")
        
        # Sort by frequency and return top topics
        return sorted(topics, key=lambda x: int(x.split('(')[1].split(' ')[0]), reverse=True)[:5]
    
    def _assess_market_impact(self, text: str, sentiment_score: float) -> str:
        """Assess potential market impact of the news."""
        text_lower = text.lower()
        
        # Check for specific impact patterns
        for pattern, explanation in self.impact_patterns.items():
            pattern_words = pattern.replace('_', ' ').split()
            if all(word in text_lower for word in pattern_words):
                return explanation
        
        # General impact based on sentiment
        if sentiment_score > 0.5:
            return "Strong positive news often leads to market rallies and increased investor confidence"
        elif sentiment_score > 0.2:
            return "Moderately positive news may support current market trends"
        elif sentiment_score < -0.5:
            return "Strong negative news often causes market sell-offs and increased volatility"
        elif sentiment_score < -0.2:
            return "Moderately negative news may create downward pressure on markets"
        else:
            return "Mixed or neutral news typically has limited immediate market impact"
    
    def _generate_news_summary(self, articles: List[NewsArticle], 
                             sentiment_score: float, key_topics: List[str]) -> str:
        """Generate a summary of the news analysis."""
        summary_parts = []
        
        # Article count and timeframe
        if len(articles) == 1:
            summary_parts.append("Analyzed 1 financial news article.")
        else:
            summary_parts.append(f"Analyzed {len(articles)} financial news articles.")
        
        # Sentiment summary
        if sentiment_score > 0.3:
            summary_parts.append("The news is generally positive for markets.")
        elif sentiment_score < -0.3:
            summary_parts.append("The news contains concerning developments for markets.")
        else:
            summary_parts.append("The news presents a mixed picture for markets.")
        
        # Key topics
        if key_topics:
            main_topic = key_topics[0].split(' (')[0]
            summary_parts.append(f"Main focus area: {main_topic}.")
        
        # Investment implication
        if sentiment_score > 0.2:
            summary_parts.append("Consider this as potentially supportive for your investments.")
        elif sentiment_score < -0.2:
            summary_parts.append("Consider this as a potential risk factor for your investments.")
        else:
            summary_parts.append("This news is unlikely to significantly impact your investments.")
        
        return " ".join(summary_parts)
    
    def _calculate_confidence(self, text: str, article_count: int) -> float:
        """Calculate confidence in the sentiment analysis."""
        # Base confidence on article count and text length
        text_length = len(text.split())
        
        # More articles and longer text = higher confidence
        article_factor = min(article_count / 5.0, 1.0)  # Max boost from 5+ articles
        length_factor = min(text_length / 1000.0, 1.0)  # Max boost from 1000+ words
        
        # Check for financial keyword density
        financial_word_count = 0
        text_lower = text.lower()
        for keywords in self.financial_keywords.values():
            financial_word_count += sum(1 for keyword in keywords if keyword in text_lower)
        
        keyword_density = min(financial_word_count / max(text_length * 0.1, 1), 1.0)
        
        # Combine factors
        base_confidence = 0.4
        confidence = base_confidence + (0.6 * (article_factor + length_factor + keyword_density) / 3)
        
        return np.clip(confidence, 0.1, 0.95)
    
    def fetch_live_news(self, query: str = "financial markets", 
                       max_articles: int = 10) -> List[NewsArticle]:
        """
        Fetch live financial news (placeholder for real API integration).
        
        Args:
            query: Search query for news
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news articles
        """
        # This is a placeholder - in production, you would integrate with:
        # - Alpha Vantage News API
        # - NewsAPI
        # - Financial Times API
        # - Reuters API
        # - Bloomberg API
        
        # For now, return sample articles with current timestamps
        sample_news = [
            NewsArticle(
                title=f"Market Update: {query.title()} Show Mixed Signals",
                content="Recent market activity shows mixed signals as investors weigh "
                       "economic data against corporate earnings reports. Trading volumes "
                       "remain elevated as market participants assess ongoing developments.",
                source="Market News API",
                published_date=datetime.now() - timedelta(minutes=30),
                url="https://example.com/news/1"
            ),
            NewsArticle(
                title="Economic Indicators Point to Steady Growth",
                content="Latest economic indicators suggest steady but moderate growth, "
                       "with employment figures remaining stable and consumer confidence "
                       "showing resilience despite global uncertainties.",
                source="Economic Wire",
                published_date=datetime.now() - timedelta(hours=2),
                url="https://example.com/news/2"
            )
        ]
        
        return sample_news[:max_articles]