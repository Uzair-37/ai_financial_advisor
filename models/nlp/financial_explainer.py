"""
Financial Explainer - Translates complex financial terms and analysis into simple language.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    """Result of financial explanation."""
    simple_explanation: str
    key_insights: List[str]
    risk_level: str
    confidence: float
    recommendations: List[str]


class FinancialExplainer:
    """
    Explains complex financial concepts, analysis results, and market data in simple terms.
    """
    
    def __init__(self):
        """Initialize the financial explainer."""
        self.jargon_dictionary = self._build_jargon_dictionary()
        self.risk_levels = ["Very Low", "Low", "Moderate", "High", "Very High"]
        
    def _build_jargon_dictionary(self) -> Dict[str, str]:
        """Build a dictionary of financial terms and their simple explanations."""
        return {
            # Basic Terms
            "volatility": "how much a stock's price jumps up and down",
            "market cap": "the total value of all company shares",
            "p/e ratio": "how expensive a stock is compared to its earnings",
            "dividend": "cash payment companies give to shareholders",
            "yield": "percentage return you get from an investment",
            "bull market": "when stock prices are generally going up",
            "bear market": "when stock prices are generally going down",
            "portfolio": "your collection of different investments",
            
            # Technical Analysis
            "moving average": "average price over a specific time period",
            "rsi": "indicator showing if a stock is overbought or oversold",
            "support level": "price where a stock usually stops falling",
            "resistance level": "price where a stock usually stops rising",
            "breakout": "when price moves past a key resistance or support level",
            "trend": "general direction a stock price is moving",
            "momentum": "speed and strength of price movement",
            
            # Risk and Returns
            "beta": "how much a stock moves compared to the overall market",
            "alpha": "how much better (or worse) a stock performs than expected",
            "sharpe ratio": "measure of return versus risk",
            "standard deviation": "measure of how spread out returns are",
            "correlation": "how much two investments move together",
            "diversification": "spreading investments to reduce risk",
            
            # Market Indicators
            "vix": "fear index - measures market anxiety",
            "yield curve": "comparison of short-term vs long-term interest rates",
            "inflation": "when prices of goods and services increase over time",
            "gdp": "total value of everything a country produces",
            "unemployment rate": "percentage of people looking for work but can't find it",
            
            # Trading
            "bid": "highest price someone is willing to pay for a stock",
            "ask": "lowest price someone is willing to sell a stock for",
            "spread": "difference between bid and ask prices",
            "volume": "number of shares traded",
            "liquidity": "how easily you can buy or sell without affecting the price",
            
            # Advanced
            "hedge": "investment that protects against losses in other investments",
            "leverage": "using borrowed money to invest",
            "margin": "borrowing money from your broker to buy stocks",
            "short selling": "betting that a stock price will go down",
            "options": "contracts giving the right to buy or sell at a specific price",
            "futures": "agreements to buy or sell something at a future date",
            
            # Uncertainty Terms
            "heteroscedastic": "uncertainty that changes over time",
            "monte carlo": "using random simulations to predict possible outcomes",
            "confidence interval": "range where the true value is likely to be",
            "probability distribution": "all possible outcomes and their chances",
        }
    
    def explain_predictions(self, predictions: Dict[str, Any], 
                          model_type: str = "heteroscedastic") -> ExplanationResult:
        """
        Explain model predictions in simple terms.
        
        Args:
            predictions: Dictionary containing model predictions
            model_type: Type of uncertainty model used
            
        Returns:
            ExplanationResult with simple explanation
        """
        explanation_parts = []
        key_insights = []
        recommendations = []
        
        # Analyze prediction structure
        if 'mean' in str(predictions).lower() or 'var' in str(predictions).lower():
            # Uncertainty predictions
            explanation_parts.append("The AI model has analyzed the financial data and made predictions with uncertainty estimates.")
            
            if model_type == "heteroscedastic":
                explanation_parts.append(
                    "This means the model knows that some predictions are more reliable than others, "
                    "just like how weather forecasts are more accurate for tomorrow than next week."
                )
            elif model_type == "mc_dropout":
                explanation_parts.append(
                    "The model ran thousands of slightly different predictions and averaged them out, "
                    "similar to asking multiple experts and combining their opinions."
                )
            elif model_type == "bayesian":
                explanation_parts.append(
                    "The model considers multiple possible scenarios and weights them by probability, "
                    "like a smart betting system that considers all possibilities."
                )
                
        else:
            # Simple predictions
            explanation_parts.append("The AI model has analyzed the financial data and made straightforward predictions.")
        
        # Extract insights from predictions
        if isinstance(predictions, dict):
            for key, value in predictions.items():
                if 'price' in key.lower() or 'return' in key.lower():
                    if hasattr(value, 'mean'):
                        avg_value = float(np.mean(value))
                    elif isinstance(value, (list, np.ndarray)):
                        avg_value = float(np.mean(value))
                    else:
                        avg_value = float(value)
                    
                    if avg_value > 0:
                        key_insights.append(f"The model expects {key.replace('_', ' ')} to increase")
                        recommendations.append("Consider holding or buying positions")
                    else:
                        key_insights.append(f"The model expects {key.replace('_', ' ')} to decrease")
                        recommendations.append("Consider reducing exposure or setting stop losses")
        
        # Determine risk level
        risk_level = self._assess_risk_level(predictions)
        
        # Add risk explanation
        if risk_level in ["High", "Very High"]:
            explanation_parts.append(
                f"The risk level is {risk_level.lower()}, which means there's significant uncertainty "
                "in these predictions. Think of it like driving in heavy fog - you need to be extra careful."
            )
            recommendations.append("Consider smaller position sizes due to high uncertainty")
        else:
            explanation_parts.append(
                f"The risk level is {risk_level.lower()}, indicating relatively stable predictions."
            )
        
        return ExplanationResult(
            simple_explanation=" ".join(explanation_parts),
            key_insights=key_insights,
            risk_level=risk_level,
            confidence=self._calculate_confidence(predictions),
            recommendations=recommendations
        )
    
    def explain_technical_indicators(self, indicators: Dict[str, float]) -> str:
        """
        Explain technical indicators in simple terms.
        
        Args:
            indicators: Dictionary of technical indicator values
            
        Returns:
            Simple explanation of what the indicators mean
        """
        explanations = []
        
        for indicator, value in indicators.items():
            indicator_lower = indicator.lower()
            
            # Moving Averages
            if 'ma_' in indicator_lower and 'ratio' not in indicator_lower:
                period = indicator_lower.split('_')[1]
                explanations.append(
                    f"The {period}-day moving average is ${value:.2f}, which shows the average "
                    f"price over the last {period} days - this helps smooth out daily price jumps."
                )
            
            elif 'ma_' in indicator_lower and 'ratio' in indicator_lower:
                if value > 1.05:
                    explanations.append(
                        "The stock is trading above its recent average price, suggesting upward momentum."
                    )
                elif value < 0.95:
                    explanations.append(
                        "The stock is trading below its recent average price, suggesting downward pressure."
                    )
                else:
                    explanations.append(
                        "The stock is trading close to its recent average price, showing stability."
                    )
            
            # RSI
            elif 'rsi' in indicator_lower:
                if value > 70:
                    explanations.append(
                        f"RSI is {value:.1f}, which is high - this might mean the stock is overpriced "
                        "and could drop soon (like a rubber band stretched too far)."
                    )
                elif value < 30:
                    explanations.append(
                        f"RSI is {value:.1f}, which is low - this might mean the stock is underpriced "
                        "and could bounce back (like a spring compressed too much)."
                    )
                else:
                    explanations.append(
                        f"RSI is {value:.1f}, which is in the normal range - the stock appears fairly valued."
                    )
            
            # Volatility
            elif 'volatility' in indicator_lower:
                if value > 0.03:
                    explanations.append(
                        f"Volatility is {value:.1%}, which is high - expect bigger price swings "
                        "(like a roller coaster ride)."
                    )
                elif value < 0.01:
                    explanations.append(
                        f"Volatility is {value:.1%}, which is low - expect smaller, steadier price movements "
                        "(like a calm lake)."
                    )
                else:
                    explanations.append(
                        f"Volatility is {value:.1%}, which is moderate - expect normal price fluctuations."
                    )
        
        if not explanations:
            return "No technical indicators to explain."
        
        return " ".join(explanations)
    
    def simplify_financial_text(self, text: str) -> str:
        """
        Replace financial jargon with simple explanations.
        
        Args:
            text: Text containing financial jargon
            
        Returns:
            Text with jargon replaced by simple explanations
        """
        simplified_text = text.lower()
        
        # Replace jargon with simple explanations
        for jargon, simple in self.jargon_dictionary.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(jargon) + r'\b'
            replacement = f"{simple}"
            simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
        
        # Capitalize first letter of sentences
        sentences = simplified_text.split('. ')
        capitalized_sentences = [s.capitalize() for s in sentences]
        
        return '. '.join(capitalized_sentences)
    
    def explain_market_conditions(self, market_data: Dict[str, Any]) -> str:
        """
        Explain overall market conditions in simple terms.
        
        Args:
            market_data: Dictionary containing market metrics
            
        Returns:
            Simple explanation of market conditions
        """
        explanations = []
        
        # Market direction
        if 'trend' in market_data:
            trend = market_data['trend']
            if trend > 0.02:
                explanations.append("The market is in an upward trend - most stocks are generally rising.")
            elif trend < -0.02:
                explanations.append("The market is in a downward trend - most stocks are generally falling.")
            else:
                explanations.append("The market is moving sideways - no clear direction up or down.")
        
        # Volatility
        if 'volatility' in market_data:
            vol = market_data['volatility']
            if vol > 0.25:
                explanations.append("Market volatility is high - expect big price swings and uncertainty.")
            elif vol < 0.15:
                explanations.append("Market volatility is low - expect calmer, more predictable movements.")
            else:
                explanations.append("Market volatility is moderate - normal fluctuations expected.")
        
        # Volume
        if 'volume' in market_data:
            volume = market_data['volume']
            if volume > 1.2:  # Assuming normalized volume
                explanations.append("Trading volume is high - lots of people are buying and selling.")
            elif volume < 0.8:
                explanations.append("Trading volume is low - fewer people are actively trading.")
            else:
                explanations.append("Trading volume is normal - typical level of market activity.")
        
        return " ".join(explanations) if explanations else "Market conditions appear normal."
    
    def _assess_risk_level(self, predictions: Dict[str, Any]) -> str:
        """Assess risk level based on prediction uncertainty."""
        try:
            # Look for variance or uncertainty measures
            uncertainties = []
            
            for key, value in predictions.items():
                if 'var' in key.lower() or 'std' in key.lower() or 'uncertainty' in key.lower():
                    if hasattr(value, 'mean'):
                        uncertainties.append(float(np.mean(value)))
                    elif isinstance(value, (list, np.ndarray)):
                        uncertainties.append(float(np.mean(value)))
                    else:
                        uncertainties.append(float(value))
            
            if not uncertainties:
                return "Moderate"
            
            avg_uncertainty = np.mean(uncertainties)
            
            # Classify risk based on uncertainty level
            if avg_uncertainty > 0.5:
                return "Very High"
            elif avg_uncertainty > 0.3:
                return "High"
            elif avg_uncertainty > 0.15:
                return "Moderate"
            elif avg_uncertainty > 0.05:
                return "Low"
            else:
                return "Very Low"
                
        except Exception:
            return "Moderate"
    
    def _calculate_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate confidence score from predictions."""
        try:
            # Look for uncertainty measures and invert them for confidence
            uncertainties = []
            
            for key, value in predictions.items():
                if 'var' in key.lower() or 'std' in key.lower():
                    if hasattr(value, 'mean'):
                        uncertainties.append(float(np.mean(value)))
                    elif isinstance(value, (list, np.ndarray)):
                        uncertainties.append(float(np.mean(value)))
                    else:
                        uncertainties.append(float(value))
            
            if not uncertainties:
                return 0.7  # Default moderate confidence
            
            avg_uncertainty = np.mean(uncertainties)
            # Convert uncertainty to confidence (0-1 scale)
            confidence = max(0.1, min(0.95, 1.0 - avg_uncertainty))
            
            return confidence
            
        except Exception:
            return 0.7  # Default moderate confidence
    
    def generate_investment_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive investment summary in simple language.
        
        Args:
            analysis_results: Complete analysis results
            
        Returns:
            Simple, comprehensive investment summary
        """
        summary_parts = []
        
        # Opening
        summary_parts.append("Here's what the AI analysis found in simple terms:")
        
        # Main findings
        if 'predictions' in analysis_results:
            pred_explanation = self.explain_predictions(analysis_results['predictions'])
            summary_parts.append(pred_explanation.simple_explanation)
            
            if pred_explanation.key_insights:
                summary_parts.append("Key insights:")
                for insight in pred_explanation.key_insights:
                    summary_parts.append(f"• {insight}")
        
        # Technical analysis
        if 'technical_indicators' in analysis_results:
            tech_explanation = self.explain_technical_indicators(analysis_results['technical_indicators'])
            if tech_explanation != "No technical indicators to explain.":
                summary_parts.append("Technical analysis shows:")
                summary_parts.append(tech_explanation)
        
        # Market context
        if 'market_conditions' in analysis_results:
            market_explanation = self.explain_market_conditions(analysis_results['market_conditions'])
            summary_parts.append("Market conditions:")
            summary_parts.append(market_explanation)
        
        # Risk assessment
        if 'predictions' in analysis_results:
            pred_explanation = self.explain_predictions(analysis_results['predictions'])
            summary_parts.append(f"Risk level: {pred_explanation.risk_level}")
            summary_parts.append(f"Confidence in predictions: {pred_explanation.confidence:.0%}")
        
        # Recommendations
        if 'predictions' in analysis_results:
            pred_explanation = self.explain_predictions(analysis_results['predictions'])
            if pred_explanation.recommendations:
                summary_parts.append("Recommendations:")
                for rec in pred_explanation.recommendations:
                    summary_parts.append(f"• {rec}")
        
        # Disclaimer
        summary_parts.append(
            "Remember: This is AI analysis based on historical data. "
            "Real markets can be unpredictable, so always do your own research "
            "and consider consulting with a financial advisor."
        )
        
        return "\n\n".join(summary_parts)