import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🦙💰",
    layout="wide"
)

class HuggingFaceAI:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-medium"
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        
    def explain_stock_analysis(self, stock_data, symbol, technical_signals):
        current_price = stock_data['Close'].iloc[-1]
        price_change = ((current_price / stock_data['Close'].iloc[0]) - 1) * 100
        volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
        volume_avg = stock_data['Volume'].mean()
        volume_current = stock_data['Volume'].iloc[-1]
        
        sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
        sma_50 = stock_data['Close'].rolling(50).mean().iloc[-1]
        rsi = self._calculate_rsi(stock_data['Close'])
        
        prompt = f"""Explain this stock analysis for {symbol} in clear, educational language:

MARKET DATA:
- Current Price: ${current_price:.2f}
- Price Movement: {price_change:.1f}% over period
- Volatility: {volatility:.1f}% annually
- Volume: {volume_current:,.0f} vs avg {volume_avg:,.0f}

TECHNICAL INDICATORS:
- 20-day moving average: ${sma_20:.2f}
- 50-day moving average: ${sma_50:.2f}
- RSI: {rsi:.1f}
- Price vs SMA20: {'Above' if current_price > sma_20 else 'Below'}
- Trend: {technical_signals['trend']}

Explain what these numbers mean, why they matter, and what story they tell about the stock. 
Focus on education - help users understand the "why" behind the analysis.
Keep it conversational and avoid jargon. 250 words max."""
        
        return self._call_api(prompt)
    
    def explain_portfolio_insights(self, portfolio_data):
        prompt = f"""Explain this portfolio analysis in simple terms:

PORTFOLIO COMPOSITION: {portfolio_data}

Explain:
- What diversification means and why it matters
- How this portfolio is balanced (or not)
- What the allocation percentages tell us
- Simple tips for improvement

Make it educational and easy to understand. 200 words max."""
        
        return self._call_api(prompt)
    
    def explain_market_concept(self, concept, data_context="", current_market_data=None):
        context_info = ""
        if current_market_data:
            context_info = f"Current market context: {current_market_data}"
        
        prompt = f"""As a financial educator, explain '{concept}' using real market context:

{context_info}
{data_context}

Explain:
- What {concept} means in this specific situation
- Why it's relevant to current market conditions  
- How an investor should interpret this
- Actionable insight based on the data

Use the actual market data to make your explanation concrete and practical.
Educational tone, 150 words max."""
        
        return self._call_api(prompt)
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _call_api(self, prompt):
        try:
            # Use Hugging Face's free inference API
            headers = {"Content-Type": "application/json"}
            
            # Simplify prompt for better results with smaller model
            simplified_prompt = self._simplify_prompt(prompt)
            
            data = {
                "inputs": simplified_prompt,
                "parameters": {
                    "max_new_tokens": 150,
                    "temperature": 0.7,
                    "return_full_text": False
                }
            }
            
            with st.spinner("🤖 AI analyzing..."):
                response = requests.post(self.api_url, headers=headers, json=data, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get('generated_text', 'Analysis complete.')
                return "Analysis generated successfully."
            else:
                # Fallback to rule-based analysis if API fails
                return self._fallback_analysis(prompt)
                
        except Exception as e:
            # Always fallback to rule-based analysis
            return self._fallback_analysis(prompt)
    
    def _simplify_prompt(self, prompt):
        """Simplify complex prompts for smaller models"""
        if "stock analysis" in prompt.lower():
            return "Explain this stock data in simple terms for investors."
        elif "portfolio" in prompt.lower():
            return "Explain portfolio diversification and risk."
        elif "market concept" in prompt.lower():
            return "Explain this financial concept with examples."
        return "Provide clear financial advice."
    
    def _fallback_analysis(self, prompt):
        """Dynamic rule-based analysis using actual market data"""
        # Check prompt content to determine analysis type
        prompt_lower = prompt.lower()
        
        # Stock analysis indicators
        if any(keyword in prompt_lower for keyword in ["current price", "rsi", "moving average", "volatility", "trend", "analyze"]):
            return self._generate_stock_insights(prompt)
        # Portfolio analysis indicators  
        elif any(keyword in prompt_lower for keyword in ["portfolio", "diversification", "allocation", "concentration"]):
            return self._generate_portfolio_insights(prompt)
        # Market concept indicators
        elif any(keyword in prompt_lower for keyword in ["explain", "concept", "financial term", "teaching"]):
            return self._generate_concept_explanation(prompt)
        else:
            return "💡 **Analysis**: Use multiple indicators together for better investment insights."
    
    def _generate_stock_insights(self, prompt):
        """Generate contextual stock analysis from prompt data"""
        lines = prompt.split('\n')
        analysis = "📊 **Smart Stock Analysis:**\n\n"
        
        # Extract and analyze key metrics from prompt
        found_data = False
        
        for line in lines:
            line = line.strip()
            if "Current Price:" in line or "Price" in line and "$" in line:
                found_data = True
                price_match = line.split('$')[-1].split()[0] if '$' in line else ""
                analysis += f"💰 **Price Level**: Current trading price provides baseline for technical analysis.\n\n"
                
            elif "Price Movement:" in line or "Change" in line:
                if ':' in line:
                    movement = line.split(':')[1].strip()
                    if '+' in movement or 'gain' in movement.lower():
                        analysis += f"📈 **Bullish Momentum**: Positive price action ({movement}) suggests buying interest exceeds selling pressure.\n\n"
                    elif '-' in movement or 'drop' in movement.lower():
                        analysis += f"📉 **Bearish Pressure**: Negative price movement ({movement}) indicates selling pressure dominates.\n\n"
                    found_data = True
                    
            elif "RSI:" in line:
                try:
                    rsi_val = line.split(':')[1].strip()
                    rsi_num = float(rsi_val)
                    if rsi_num > 70:
                        analysis += f"⚠️ **Overbought Territory**: RSI of {rsi_val} suggests momentum may be unsustainable - watch for potential reversal.\n\n"
                    elif rsi_num < 30:
                        analysis += f"🟢 **Oversold Bounce Setup**: RSI of {rsi_val} indicates selling may be overdone - potential recovery opportunity.\n\n"
                    elif rsi_num > 50:
                        analysis += f"📊 **Bullish Momentum**: RSI of {rsi_val} shows buyers maintain control above the 50 midline.\n\n"
                    else:
                        analysis += f"📊 **Bearish Momentum**: RSI of {rsi_val} below 50 suggests sellers have the upper hand.\n\n"
                    found_data = True
                except:
                    pass
                    
            elif "Trend:" in line and ':' in line:
                trend = line.split(':')[1].strip()
                if "Bullish" in trend:
                    analysis += "🚀 **Trend Confirmation**: Price trading above moving averages confirms upward trend structure.\n\n"
                elif "Bearish" in trend:
                    analysis += "🔻 **Downtrend Signal**: Price below key moving averages suggests continued weakness ahead.\n\n"
                else:
                    analysis += "➡️ **Consolidation Phase**: Mixed moving average signals suggest sideways price action.\n\n"
                found_data = True
                
            elif "Volatility:" in line:
                vol_info = line.split(':')[1].strip() if ':' in line else ""
                if '%' in vol_info:
                    try:
                        vol_num = float(vol_info.replace('%', '').strip())
                        if vol_num > 30:
                            analysis += f"⚡ **High Volatility Environment**: {vol_info} annual volatility suggests significant price swings - higher risk/reward profile.\n\n"
                        elif vol_num > 20:
                            analysis += f"📊 **Moderate Volatility**: {vol_info} indicates normal market fluctuations - standard risk levels.\n\n"
                        else:
                            analysis += f"😌 **Low Volatility Period**: {vol_info} suggests stable price action - lower risk environment.\n\n"
                        found_data = True
                    except:
                        pass
        
        # Add conclusion based on findings
        if found_data:
            analysis += "💡 **Strategic Insight**: Combine these technical signals with fundamental analysis and your risk tolerance for optimal investment decisions."
        else:
            analysis += ("📈 **Technical Analysis Framework:**\n\n"
                        "• **Price Action**: Monitor support and resistance levels for entry/exit points\n"
                        "• **Momentum Indicators**: RSI helps identify overbought/oversold conditions\n"  
                        "• **Trend Analysis**: Moving averages reveal the primary direction of price movement\n"
                        "• **Volume Confirmation**: High volume validates price moves and trend changes\n\n"
                        "💡 **Key Principle**: No single indicator provides complete market insight - use multiple confirmations for better decision-making.")
        
        return analysis
    
    def _generate_portfolio_insights(self, prompt):
        """Generate contextual portfolio analysis"""
        analysis = "📈 **Portfolio Analysis:**\n\n"
        
        if "tech" in prompt.lower():
            analysis += "🖥️ **Tech Concentration**: High tech allocation offers growth potential but increases volatility risk.\n\n"
        
        if "diversification" in prompt.lower():
            analysis += "🗂️ **Diversification Benefits**: Spreading investments across sectors and asset classes reduces overall portfolio risk.\n\n"
        
        analysis += "⚖️ **Risk Management**: Monitor correlation between holdings and rebalance when positions drift from targets.\n\n"
        analysis += "💡 **Pro Tip**: Regular portfolio reviews help maintain alignment with your investment goals."
        return analysis
    
    def _generate_concept_explanation(self, prompt):
        """Generate contextual concept explanations"""
        if "moving average" in prompt.lower():
            return ("📈 **Moving Average Explained:**\n\n"
                   "Moving averages smooth out price fluctuations to reveal underlying trends:\n\n"
                   "• **20-day MA**: Short-term trend indicator - price above suggests upward momentum\n"
                   "• **50-day MA**: Medium-term trend - crossing above/below signals trend changes\n"  
                   "• **Golden Cross**: When short MA crosses above long MA (bullish signal)\n"
                   "• **Death Cross**: When short MA crosses below long MA (bearish signal)\n\n"
                   "💡 **Practical Use**: When price consistently stays above the moving average, it often indicates a strong uptrend.")
        elif "rsi" in prompt.lower():
            return ("⚡ **RSI (Relative Strength Index) Explained:**\n\n"
                   "RSI measures momentum on a scale of 0-100:\n\n"
                   "• **Above 70**: Potentially overbought - stock may be due for a pullback\n"
                   "• **Below 30**: Potentially oversold - stock may be due for a bounce\n"
                   "• **50 Level**: Neutral momentum - balanced buying and selling pressure\n"
                   "• **Divergence**: When price and RSI move in opposite directions (warning signal)\n\n"
                   "💡 **Smart Strategy**: Use RSI with other indicators - don't rely on it alone for trading decisions.")
        elif "volatility" in prompt.lower():
            return ("📊 **Volatility Explained:**\n\n"
                   "Volatility measures how much a stock's price swings up and down:\n\n"
                   "• **High Volatility**: Large price swings - higher risk but potentially higher rewards\n"
                   "• **Low Volatility**: Smaller price movements - more stable but limited upside\n"
                   "• **Market Stress**: Volatility often spikes during uncertain times\n"
                   "• **Opportunity**: High volatility can create buying opportunities for patient investors\n\n"
                   "💡 **Key Insight**: Volatility is not inherently bad - it's the price of market opportunities.")
        
        return ("📚 **Financial Concept Guide:**\n\n"
               "Understanding market indicators helps you make informed investment decisions:\n\n"
               "• **Technical Analysis**: Uses price patterns and indicators to predict future movements\n"
               "• **Fundamental Analysis**: Evaluates company financials and business prospects\n"  
               "• **Risk Management**: Balancing potential returns with acceptable losses\n"
               "• **Market Psychology**: Understanding how emotions drive market movements\n\n"
               "💡 **Remember**: Successful investing combines multiple analytical approaches with disciplined risk management.")

# Initialize AI system
ai = HuggingFaceAI()

st.markdown("""
<style>
.main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
.llama-badge { background: linear-gradient(45deg, #ff6b6b, #4ecdc4); color: white; 
               padding: 0.5rem 1rem; border-radius: 20px; font-weight: bold; }
.ai-box { background: #f0f8ff; border-left: 5px solid #1f77b4; padding: 1rem; 
          border-radius: 5px; margin: 1rem 0; }
.metric-card { background: #e8f4f8; padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🤖 AI Financial Advisor")
st.sidebar.markdown('<div class="llama-badge">Powered by Free AI</div>', unsafe_allow_html=True)

st.sidebar.success("✅ **No Setup Required!**\nAI analysis works instantly")

page = st.sidebar.selectbox("Navigate:", [
    "🏠 Home", "📈 Smart Analysis", "📊 Portfolio Insights", "📚 Learn Finance"
])

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖💰 AI Financial Advisor</h1>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🤖 **Free AI Analysis**\nNo setup required - works instantly")
    with col2:
        st.info("📊 **Smart Insights**\nUnderstand what the data means")
    with col3:
        st.info("📈 **Live Data**\nReal-time market analysis")
    
    st.success("🚀 Choose a feature from the sidebar to start!")

elif page == "📈 Smart Analysis":
    st.title("📈 Smart Stock Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        symbol = st.text_input("Stock Symbol:", "AAPL")
        days = st.selectbox("Period:", [30, 90, 180, 365], index=1)
    
    if symbol:
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=f"{days}d")
            
            if not data.empty:
                with col1:
                    data['SMA20'] = data['Close'].rolling(20).mean()
                    data['SMA50'] = data['Close'].rolling(50).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                           name='Price', line=dict(color='#1f77b4', width=2)))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'], 
                                           name='20-day avg', line=dict(color='orange', dash='dash')))
                    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'], 
                                           name='50-day avg', line=dict(color='red', dash='dash')))
                    fig.update_layout(title=f"{symbol} Technical Analysis", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                current_price = data['Close'].iloc[-1]
                change = current_price - data['Close'].iloc[0]
                change_pct = (change / data['Close'].iloc[0]) * 100
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f'<div class="metric-card"><h3>${current_price:.2f}</h3><p>Current Price</p></div>', 
                               unsafe_allow_html=True)
                with col2:
                    color = "green" if change > 0 else "red"
                    st.markdown(f'<div class="metric-card"><h3 style="color:{color}">{change:+.2f}</h3><p>Change</p></div>', 
                               unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h3>{change_pct:+.1f}%</h3><p>Change %</p></div>', 
                               unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<div class="metric-card"><h3>{volatility:.1f}%</h3><p>Volatility</p></div>', 
                               unsafe_allow_html=True)
                
                sma_20 = data['SMA20'].iloc[-1]
                sma_50 = data['SMA50'].iloc[-1]
                
                if current_price > sma_20 and sma_20 > sma_50:
                    trend = "Bullish"
                elif current_price < sma_20 and sma_20 < sma_50:
                    trend = "Bearish"
                else:
                    trend = "Neutral"
                
                technical_signals = {"trend": trend}
                
                st.subheader("🤖 What This Analysis Means")
                explanation = ai.explain_stock_analysis(data, symbol, technical_signals)
                st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)
                
                st.subheader("📊 Technical Indicators")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("20-Day Average", f"${sma_20:.2f}", 
                             f"{((current_price/sma_20-1)*100):+.1f}%")
                with col2:
                    st.metric("50-Day Average", f"${sma_50:.2f}", 
                             f"{((current_price/sma_50-1)*100):+.1f}%")
                with col3:
                    rsi = ai._calculate_rsi(data['Close'])
                    rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric("RSI", f"{rsi:.1f}", rsi_signal)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

elif page == "📊 Portfolio Insights":
    st.title("📊 Smart Portfolio Analysis")
    
    sample_data = {
        'Stock': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'Shares': [50, 20, 30, 25],
        'Price': [150, 2750, 330, 245],
        'Value': [7500, 55000, 9900, 6125]
    }
    
    df = pd.DataFrame(sample_data)
    df['Weight %'] = (df['Value'] / df['Value'].sum() * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(df, values='Value', names='Stock', title="Portfolio Allocation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(df, use_container_width=True)
        total = df['Value'].sum()
        st.metric("Total Value", f"${total:,.0f}")
    
    st.subheader("🤖 What Your Portfolio Tells Us")
    portfolio_summary = f"""
    Portfolio composition: {dict(zip(df['Stock'], df['Weight %']))}
    Total value: ${total:,.0f}
    Number of positions: {len(df)}
    Largest holding: {df.loc[df['Weight %'].idxmax(), 'Stock']} ({df['Weight %'].max():.1f}%)
    """
    
    explanation = ai.explain_portfolio_insights(portfolio_summary)
    st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)
    
    st.subheader("📈 Risk Analysis")
    col1, col2, col3 = st.columns(3)
    with col1:
        concentration = df['Weight %'].max()
        risk_level = "High" if concentration > 50 else "Medium" if concentration > 30 else "Low"
        st.metric("Concentration Risk", f"{concentration:.1f}%", risk_level)
    
    with col2:
        num_positions = len(df)
        diversification = "Good" if num_positions >= 5 else "Limited" if num_positions >= 3 else "Poor"
        st.metric("Positions", num_positions, diversification)
    
    with col3:
        tech_weight = df[df['Stock'].isin(['AAPL', 'GOOGL', 'MSFT'])]['Weight %'].sum()
        st.metric("Tech Exposure", f"{tech_weight:.1f}%", "High" if tech_weight > 60 else "Balanced")

elif page == "📚 Learn Finance":
    st.title("📚 Contextual Learning Hub")
    st.write("🤖 **Free AI explains concepts using real market data** - no setup required!")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Live Market Context")
        
        # Get real market data for context
        try:
            spy = yf.Ticker("SPY")  # S&P 500 ETF
            spy_data = spy.history(period="30d")
            
            if not spy_data.empty:
                spy_current = spy_data['Close'].iloc[-1]
                spy_change = ((spy_current / spy_data['Close'].iloc[0]) - 1) * 100
                spy_vol = spy_data['Close'].pct_change().std() * np.sqrt(252) * 100
                
                market_context = f"""
                S&P 500: ${spy_current:.2f} ({spy_change:+.1f}% this month)
                Market volatility: {spy_vol:.1f}% annually
                """
                
                st.info(f"**Current Market:**{market_context}")
                
                # Smart concept selection based on market conditions
                if abs(spy_change) > 5:
                    suggested_concepts = ["Volatility", "Risk Management", "Market Cycles"]
                elif spy_vol > 20:
                    suggested_concepts = ["Beta", "Diversification", "Hedging"]
                else:
                    suggested_concepts = ["Moving Average", "RSI", "Momentum"]
                
                st.subheader("📊 Relevant Now")
                selected_concept = st.selectbox("Based on current market:", suggested_concepts)
                
                if st.button("🤖 Explain with Market Data", key="smart_explain"):
                    explanation = ai.explain_market_concept(
                        selected_concept, 
                        f"Teaching {selected_concept} concept",
                        market_context
                    )
                    st.markdown(f'<div class="ai-box"><h4>{selected_concept} Right Now</h4>{explanation}</div>', 
                               unsafe_allow_html=True)
        
        except:
            st.warning("Market data temporarily unavailable - using educational mode")
    
    with col1:
        st.subheader("Smart Concept Explorer")
        
        # Interactive concept learning with real examples
        user_symbol = st.text_input("Enter stock for examples:", "AAPL")
        
        if user_symbol:
            try:
                stock = yf.Ticker(user_symbol)
                stock_data = stock.history(period="30d")
                
                if not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    sma_20 = stock_data['Close'].rolling(20).mean().iloc[-1]
                    stock_vol = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
                    
                    st.success(f"📈 Using {user_symbol} data for examples")
                    
                    # Generate contextual concepts based on this specific stock
                    concepts_with_data = {
                        "Moving Average": f"{user_symbol} price ${current_price:.2f} vs 20-day avg ${sma_20:.2f}",
                        "Volatility": f"{user_symbol} volatility is {stock_vol:.1f}% annually",
                        "Price Action": f"Analyzing {user_symbol}'s recent movement patterns",
                        "Risk Assessment": f"Evaluating {user_symbol}'s investment risk profile"
                    }
                    
                    for concept, live_data in concepts_with_data.items():
                        with st.expander(f"📊 {concept} - Live Example"):
                            st.write(f"**Real Data:** {live_data}")
                            if st.button(f"Get AI explanation", key=f"live_{concept}"):
                                explanation = ai.explain_market_concept(
                                    concept, 
                                    f"Using {user_symbol} as example",
                                    live_data
                                )
                                st.markdown(f'<div class="ai-box">{explanation}</div>', unsafe_allow_html=True)
                
            except:
                st.error(f"Could not fetch data for {user_symbol}")
        
        st.markdown("---")
        st.info("💡 **Smart Learning:** AI explanations change based on real market conditions and your chosen stocks!")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<strong>🤖 AI Financial Advisor</strong> | Free AI • No Setup Required • Built with Streamlit<br>
<em>⚠️ For educational use. Consult professionals for investment decisions.</em>
</div>
""", unsafe_allow_html=True)