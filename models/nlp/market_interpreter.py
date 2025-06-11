"""
Market Interpreter - Analyzes real-time macro finance data and explains it in simple terms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import json


@dataclass
class MarketInsight:
    """Market analysis insight."""
    metric_name: str
    current_value: float
    change_from_previous: float
    interpretation: str
    impact_level: str  # Low, Medium, High
    recommendation: str


@dataclass
class MacroAnalysis:
    """Complete macro finance analysis."""
    overall_sentiment: str
    key_insights: List[MarketInsight]
    market_summary: str
    risk_assessment: str
    investment_implications: List[str]
    confidence_score: float


class MarketInterpreter:
    """
    Interprets real-time macro finance data and explains market conditions in simple terms.
    """
    
    def __init__(self):
        """Initialize the market interpreter."""
        self.metric_thresholds = self._define_metric_thresholds()
        self.correlation_matrix = self._build_correlation_matrix()
        self.interpretation_templates = self._load_interpretation_templates()
        
    def _define_metric_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Define thresholds for different financial metrics."""
        return {
            'vix': {
                'very_low': 12,
                'low': 16,
                'normal': 20,
                'high': 25,
                'very_high': 30
            },
            'yield_spread': {
                'inverted': 0,
                'flat': 0.5,
                'normal': 1.5,
                'steep': 2.5,
                'very_steep': 3.5
            },
            'unemployment_rate': {
                'very_low': 3.0,
                'low': 4.0,
                'normal': 5.5,
                'high': 7.0,
                'very_high': 9.0
            },
            'inflation_rate': {
                'deflation': 0,
                'very_low': 1.0,
                'target': 2.0,
                'elevated': 3.0,
                'high': 5.0,
                'very_high': 7.0
            },
            'gdp_growth': {
                'recession': 0,
                'slow': 1.0,
                'moderate': 2.5,
                'strong': 4.0,
                'very_strong': 6.0
            }
        }
    
    def _build_correlation_matrix(self) -> Dict[str, Dict[str, str]]:
        """Build correlation relationships between metrics."""
        return {
            'vix_high': {
                'market_impact': 'negative',
                'explanation': 'High fear usually means stock prices drop'
            },
            'yield_curve_inverted': {
                'market_impact': 'very_negative',
                'explanation': 'Inverted yield curve often predicts recession'
            },
            'unemployment_rising': {
                'market_impact': 'negative',
                'explanation': 'Rising unemployment hurts consumer spending and corporate profits'
            },
            'inflation_high': {
                'market_impact': 'mixed',
                'explanation': 'High inflation can hurt stocks but may help some commodities'
            },
            'gdp_strong': {
                'market_impact': 'positive',
                'explanation': 'Strong economic growth usually supports stock markets'
            }
        }
    
    def _load_interpretation_templates(self) -> Dict[str, str]:
        """Load templates for interpreting different scenarios."""
        return {
            'market_rally': "Markets are in rally mode - most investors are optimistic and buying",
            'market_decline': "Markets are declining - investors are selling due to concerns",
            'high_volatility': "Markets are very volatile - prices are swinging wildly up and down",
            'low_volatility': "Markets are calm - prices are moving in small, predictable ranges",
            'economic_expansion': "The economy is growing - businesses are doing well and hiring",
            'economic_contraction': "The economy is shrinking - businesses are struggling",
            'recession_warning': "Warning signs suggest a possible recession ahead",
            'recovery_signs': "Signs point to economic recovery and improvement"
        }
    
    def analyze_macro_data(self, macro_data: Dict[str, float]) -> MacroAnalysis:
        """
        Analyze macro finance data and provide simple explanations.
        
        Args:
            macro_data: Dictionary of macro finance metrics
            
        Returns:
            MacroAnalysis with insights and explanations
        """
        insights = []
        
        # Analyze each metric
        for metric, value in macro_data.items():
            if metric in self.metric_thresholds:
                insight = self._analyze_single_metric(metric, value)
                if insight:
                    insights.append(insight)
        
        # Generate overall assessment
        overall_sentiment = self._assess_overall_sentiment(insights)
        market_summary = self._generate_market_summary(insights, overall_sentiment)
        risk_assessment = self._assess_market_risk(insights)
        investment_implications = self._generate_investment_implications(insights)
        confidence_score = self._calculate_confidence(macro_data, insights)
        
        return MacroAnalysis(
            overall_sentiment=overall_sentiment,
            key_insights=insights,
            market_summary=market_summary,
            risk_assessment=risk_assessment,
            investment_implications=investment_implications,
            confidence_score=confidence_score
        )
    
    def explain_economic_indicators(self, indicators: Dict[str, float]) -> str:
        """
        Explain economic indicators in simple, human terms.
        
        Args:
            indicators: Dictionary of economic indicators
            
        Returns:
            Simple explanation of what the indicators mean
        """
        explanations = []
        
        for indicator, value in indicators.items():
            indicator_lower = indicator.lower().replace('_', ' ')
            
            if 'gdp' in indicator_lower:
                if value > 3:
                    explanations.append(
                        f"GDP growth is {value:.1f}%, which is strong - the economy is expanding well, "
                        "like a healthy person getting stronger."
                    )
                elif value > 1:
                    explanations.append(
                        f"GDP growth is {value:.1f}%, which is moderate - the economy is growing slowly "
                        "but steadily, like a gradual recovery."
                    )
                else:
                    explanations.append(
                        f"GDP growth is {value:.1f}%, which is concerning - the economy might be "
                        "stagnating or shrinking, like someone losing strength."
                    )
            
            elif 'unemployment' in indicator_lower:
                if value < 4:
                    explanations.append(
                        f"Unemployment is {value:.1f}%, which is very low - almost everyone who wants "
                        "a job can find one. This is great for workers but might cause wage inflation."
                    )
                elif value < 6:
                    explanations.append(
                        f"Unemployment is {value:.1f}%, which is healthy - most people can find jobs "
                        "without too much pressure on wages."
                    )
                else:
                    explanations.append(
                        f"Unemployment is {value:.1f}%, which is high - many people are struggling "
                        "to find work, which hurts the economy."
                    )
            
            elif 'inflation' in indicator_lower:
                if value < 1:
                    explanations.append(
                        f"Inflation is {value:.1f}%, which is very low - prices are barely rising, "
                        "which might signal economic weakness."
                    )
                elif value < 3:
                    explanations.append(
                        f"Inflation is {value:.1f}%, which is healthy - prices are rising at a "
                        "reasonable pace that supports economic growth."
                    )
                else:
                    explanations.append(
                        f"Inflation is {value:.1f}%, which is concerning - prices are rising fast, "
                        "making everything more expensive for consumers."
                    )
            
            elif 'vix' in indicator_lower:
                if value > 25:
                    explanations.append(
                        f"The fear index (VIX) is {value:.1f}, which is high - investors are very "
                        "nervous and expect big market swings."
                    )
                elif value > 15:
                    explanations.append(
                        f"The fear index (VIX) is {value:.1f}, which is moderate - normal level "
                        "of market uncertainty."
                    )
                else:
                    explanations.append(
                        f"The fear index (VIX) is {value:.1f}, which is low - investors are calm "
                        "and confident about the market."
                    )
            
            elif 'yield' in indicator_lower and 'spread' in indicator_lower:
                if value < 0:
                    explanations.append(
                        f"The yield curve is inverted (spread: {value:.2f}) - this is a strong "
                        "recession warning signal that has predicted past economic downturns."
                    )
                elif value < 1:
                    explanations.append(
                        f"The yield curve is flat (spread: {value:.2f}) - this suggests economic "
                        "growth might be slowing down."
                    )
                else:
                    explanations.append(
                        f"The yield curve is normal (spread: {value:.2f}) - this suggests healthy "
                        "economic conditions with normal growth expectations."
                    )
        
        if not explanations:
            return "No recognizable economic indicators to explain."
        
        return " ".join(explanations)
    
    def get_sample_macro_data(self) -> Dict[str, float]:
        """Get sample macro finance data for demonstration."""
        return {
            'vix': 18.5,
            'yield_spread': 1.2,
            'unemployment_rate': 4.1,
            'inflation_rate': 2.8,
            'gdp_growth': 2.3,
            'dollar_index': 102.5,
            'oil_price': 75.2,
            'gold_price': 1950.0
        }
    
    def interpret_market_conditions(self, market_data: Dict[str, Any]) -> str:
        """
        Interpret current market conditions in simple terms.
        
        Args:
            market_data: Dictionary containing various market metrics
            
        Returns:
            Simple interpretation of market conditions
        """
        conditions = []
        
        # Market direction
        if 'market_trend' in market_data:
            trend = market_data['market_trend']
            if trend > 0.05:
                conditions.append("Markets are in a strong upward trend - most stocks are rising.")
            elif trend > 0.01:
                conditions.append("Markets have a slight upward bias - gentle gains overall.")
            elif trend < -0.05:
                conditions.append("Markets are in a downward trend - most stocks are falling.")
            elif trend < -0.01:
                conditions.append("Markets have a slight downward bias - gentle declines overall.")
            else:
                conditions.append("Markets are moving sideways - no clear direction.")
        
        # Volatility
        if 'volatility' in market_data:
            vol = market_data['volatility']
            if vol > 0.25:
                conditions.append(
                    "Volatility is very high - expect big price swings and unpredictable moves."
                )
            elif vol > 0.15:
                conditions.append(
                    "Volatility is elevated - more price movement than usual."
                )
            else:
                conditions.append(
                    "Volatility is normal - typical day-to-day price movements."
                )
        
        # Trading activity
        if 'volume' in market_data:
            volume = market_data['volume']
            if volume > 1.5:  # Assuming normalized volume
                conditions.append("Trading volume is very high - lots of buying and selling activity.")
            elif volume > 1.1:
                conditions.append("Trading volume is above average - increased market activity.")
            elif volume < 0.8:
                conditions.append("Trading volume is low - fewer people are actively trading.")
            else:
                conditions.append("Trading volume is normal - typical market activity.")
        
        # Sector performance
        if 'sector_performance' in market_data:
            sectors = market_data['sector_performance']
            if isinstance(sectors, dict):
                best_sector = max(sectors.items(), key=lambda x: x[1])
                worst_sector = min(sectors.items(), key=lambda x: x[1])
                
                conditions.append(
                    f"Best performing sector: {best_sector[0]} (+{best_sector[1]:.1%}). "
                    f"Worst performing: {worst_sector[0]} ({worst_sector[1]:.1%})."
                )
        
        if not conditions:
            return "Market conditions appear normal with no significant unusual activity."
        
        return " ".join(conditions)
    
    def generate_daily_market_briefing(self, all_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive daily market briefing in simple language.
        
        Args:
            all_data: Dictionary containing all market and economic data
            
        Returns:
            Simple, comprehensive market briefing
        """
        briefing_parts = []
        
        # Opening
        briefing_parts.append("Here's your daily market briefing in simple terms:")
        
        # Market overview
        if 'market_data' in all_data:
            market_conditions = self.interpret_market_conditions(all_data['market_data'])
            briefing_parts.append(f"Market Overview: {market_conditions}")
        
        # Economic indicators
        if 'economic_indicators' in all_data:
            economic_explanation = self.explain_economic_indicators(all_data['economic_indicators'])
            briefing_parts.append(f"Economic Health: {economic_explanation}")
        
        # Key developments
        if 'macro_analysis' in all_data:
            macro_analysis = all_data['macro_analysis']
            if isinstance(macro_analysis, MacroAnalysis):
                briefing_parts.append(f"Overall Sentiment: {macro_analysis.overall_sentiment}")
                briefing_parts.append(f"Risk Level: {macro_analysis.risk_assessment}")
                
                if macro_analysis.investment_implications:
                    briefing_parts.append("What this means for your investments:")
                    for implication in macro_analysis.investment_implications:
                        briefing_parts.append(f"• {implication}")
        
        # Bottom line
        briefing_parts.append(
            "Bottom line: Markets are complex, but understanding these basics helps you make "
            "better investment decisions. Always consider your personal financial situation "
            "and risk tolerance when making investment choices."
        )
        
        return "\n\n".join(briefing_parts)
    
    def _analyze_single_metric(self, metric: str, value: float) -> Optional[MarketInsight]:
        """Analyze a single metric and generate insight."""
        thresholds = self.metric_thresholds.get(metric)
        if not thresholds:
            return None
        
        # Determine level and interpretation
        interpretation = ""
        impact_level = "Medium"
        recommendation = ""
        
        if metric == 'vix':
            if value > thresholds['very_high']:
                interpretation = "Market fear is extremely high - investors are panicking"
                impact_level = "High"
                recommendation = "Consider defensive positions and avoid high-risk investments"
            elif value > thresholds['high']:
                interpretation = "Market fear is elevated - increased uncertainty"
                impact_level = "Medium"
                recommendation = "Be cautious with new positions"
            elif value < thresholds['low']:
                interpretation = "Market fear is low - investors are confident"
                impact_level = "Low"
                recommendation = "Good environment for growth investments"
            else:
                interpretation = "Market fear is at normal levels"
                impact_level = "Low"
                recommendation = "Normal market conditions"
        
        elif metric == 'unemployment_rate':
            if value > thresholds['high']:
                interpretation = "Unemployment is high - economy is struggling"
                impact_level = "High"
                recommendation = "Focus on defensive stocks and essential services"
            elif value < thresholds['low']:
                interpretation = "Unemployment is very low - strong job market"
                impact_level = "Medium"
                recommendation = "Good for consumer discretionary stocks"
            else:
                interpretation = "Unemployment is at healthy levels"
                impact_level = "Low"
                recommendation = "Balanced approach to investments"
        
        # Add more metric-specific logic as needed
        
        return MarketInsight(
            metric_name=metric.replace('_', ' ').title(),
            current_value=value,
            change_from_previous=0.0,  # Would need historical data
            interpretation=interpretation,
            impact_level=impact_level,
            recommendation=recommendation
        )
    
    def _assess_overall_sentiment(self, insights: List[MarketInsight]) -> str:
        """Assess overall market sentiment from individual insights."""
        if not insights:
            return "Neutral"
        
        positive_count = sum(1 for insight in insights 
                           if 'good' in insight.interpretation.lower() or 
                              'strong' in insight.interpretation.lower() or
                              'confident' in insight.interpretation.lower())
        
        negative_count = sum(1 for insight in insights 
                           if 'high' in insight.interpretation.lower() and 'fear' in insight.interpretation.lower() or
                              'struggling' in insight.interpretation.lower() or
                              'elevated' in insight.interpretation.lower())
        
        if positive_count > negative_count:
            return "Positive"
        elif negative_count > positive_count:
            return "Negative"
        else:
            return "Mixed"
    
    def _generate_market_summary(self, insights: List[MarketInsight], sentiment: str) -> str:
        """Generate overall market summary."""
        if sentiment == "Positive":
            return "Markets are showing positive signals with supportive economic conditions."
        elif sentiment == "Negative":
            return "Markets face headwinds with concerning economic indicators."
        else:
            return "Markets show mixed signals with both positive and negative factors."
    
    def _assess_market_risk(self, insights: List[MarketInsight]) -> str:
        """Assess overall market risk level."""
        high_risk_count = sum(1 for insight in insights if insight.impact_level == "High")
        
        if high_risk_count >= 2:
            return "High Risk"
        elif high_risk_count == 1:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def _generate_investment_implications(self, insights: List[MarketInsight]) -> List[str]:
        """Generate investment implications from insights."""
        implications = []
        
        for insight in insights:
            if insight.recommendation and insight.recommendation not in implications:
                implications.append(insight.recommendation)
        
        return implications[:3]  # Limit to top 3
    
    def _calculate_confidence(self, macro_data: Dict[str, float], 
                            insights: List[MarketInsight]) -> float:
        """Calculate confidence in the analysis."""
        # Base confidence on data completeness and consistency
        data_completeness = min(len(macro_data) / 8.0, 1.0)  # Assuming 8 key metrics
        insight_consistency = len(insights) / max(len(macro_data), 1)
        
        confidence = 0.5 + 0.5 * (data_completeness + insight_consistency) / 2
        return np.clip(confidence, 0.3, 0.9)