import yfinance as yf
import pandas as pd
from datetime import datetime
from crewai_tools import tool


@tool
def yf_fundamental_analysis(ticker: str):
    """
    Perform a comprehensive fundamental analysis on the given stock symbol.
    
    Args:
        ticker (str): The stock symbol to analyze.
    
    Returns:
        dict: A dictionary with the detailed fundamental analysis results.
    """
    try:
        # Fetch the stock data
        stock = yf.Ticker(ticker)
        info = stock.info

        # Retrieve financial data
        try:
            financials = stock.financials.fillna(0).infer_objects(copy=False)
            balance_sheet = stock.balance_sheet.fillna(0).infer_objects(copy=False)
            cash_flow = stock.cashflow.fillna(0).infer_objects(copy=False)
        except Exception as e:
            return f"Error fetching financial data: {e}"

        # Fill missing values
        financials = financials.ffill()
        balance_sheet = balance_sheet.ffill()
        cash_flow = cash_flow.ffill()

        # Key Ratios and Metrics
        ratios = {
            "P/E Ratio": info.get('trailingPE', 0),
            "Forward P/E": info.get('forwardPE', 0),
            "P/B Ratio": info.get('priceToBook', 0),
            "P/S Ratio": info.get('priceToSalesTrailing12Months', 0),
            "PEG Ratio": info.get('pegRatio', 0),
            "Debt to Equity": info.get('debtToEquity', 0),
            "Current Ratio": info.get('currentRatio', 0),
            "Quick Ratio": info.get('quickRatio', 0),
            "ROE": info.get('returnOnEquity', 0),
            "ROA": info.get('returnOnAssets', 0),
            "ROIC": info.get('returnOnCapital', 0),
            "Gross Margin": info.get('grossMargins', 0),
            "Operating Margin": info.get('operatingMargins', 0),
            "Net Profit Margin": info.get('profitMargins', 0),
            "Dividend Yield": info.get('dividendYield', 0),
            "Payout Ratio": info.get('payoutRatio', 0),
        }

        # Growth Rates
        revenue = financials.loc['Total Revenue'] if 'Total Revenue' in financials.index else pd.Series()
        net_income = financials.loc['Net Income'] if 'Net Income' in financials.index else pd.Series()

        revenue_growth = revenue.pct_change(periods=-1).iloc[0] if len(revenue) > 1 else 0
        net_income_growth = net_income.pct_change(periods=-1).iloc[0] if len(net_income) > 1 else 0

        growth_rates = {
            "Revenue Growth (YoY)": revenue_growth,
            "Net Income Growth (YoY)": net_income_growth,
        }

        # Valuation
        market_cap = info.get('marketCap', 0)
        enterprise_value = info.get('enterpriseValue', 0)

        valuation = {
            "Market Cap": market_cap,
            "Enterprise Value": enterprise_value,
            "EV/EBITDA": info.get('enterpriseToEbitda', 0),
            "EV/Revenue": info.get('enterpriseToRevenue', 0),
        }

        # Future Estimates
        estimates = {
            "Next Year EPS Estimate": info.get('forwardEps', 0),
            "Next Year Revenue Estimate": info.get('revenueEstimates', {}).get('avg', 0),
            "Long-term Growth Rate": info.get('longTermPotentialGrowthRate', 0.03),
        }

        # Simple DCF Valuation
        free_cash_flow = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0
        wacc = 0.1  # Assumed Weighted Average Cost of Capital
        growth_rate = info.get('longTermPotentialGrowthRate', 0.03)

        def simple_dcf(fcf, growth_rate, wacc, years=5):
            if fcf == 0 or growth_rate == 0:
                return 0
            terminal_value = fcf * (1 + growth_rate) / (wacc - growth_rate)
            dcf_value = sum([fcf * (1 + growth_rate) ** i / (1 + wacc) ** i for i in range(1, years + 1)])
            dcf_value += terminal_value / (1 + wacc) ** years
            return dcf_value

        dcf_value = simple_dcf(free_cash_flow, growth_rate, wacc)

        # Prepare the results
        analysis = {
            "Company Name": info.get('longName', ticker),
            "Sector": info.get('sector', 'N/A'),
            "Industry": info.get('industry', 'N/A'),
            "Key Ratios": ratios,
            "Growth Rates": growth_rates,
            "Valuation Metrics": valuation,
            "Future Estimates": estimates,
            "Simple DCF Valuation": dcf_value,
            "Last Updated": datetime.fromtimestamp(info.get('lastFiscalYearEnd', 0)).strftime('%Y-%m-%d'),
            "Data Retrieval Date": datetime.now().strftime('%Y-%m-%d'),
        }

        # Add interpretations
        interpretations = {
            "P/E Ratio": "High P/E might indicate overvaluation or high growth expectations" if ratios.get('P/E Ratio', 0) > 20 else "Low P/E might indicate undervaluation or low growth expectations",
            "Debt to Equity": "High leverage" if ratios.get('Debt to Equity', 0) > 2 else "Conservative capital structure",
            "ROE": "Strong returns" if ratios.get('ROE', 0) > 0.15 else "Potential profitability issues",
            "Revenue Growth": "Strong growth" if growth_rates.get('Revenue Growth (YoY)', 0) > 0.1 else "Slowing growth",
        }

        analysis["Interpretations"] = interpretations

        return analysis

    except Exception as e:
        return f"An error occurred during the analysis: {str(e)}"

