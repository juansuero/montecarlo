import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime, timedelta
import json
import base64
import requests

# Streamlit App Configuration
st.title("Monte Carlo Portfolio Simulator")

# Function to convert configuration to JSON string
def export_config(config):
    return base64.b64encode(json.dumps(config).encode()).decode()

# Function to import configuration from JSON string
def import_config(config_string):
    try:
        return json.loads(base64.b64decode(config_string.encode()).decode())
    except:
        st.error("Invalid configuration string")
        return None

# Configuration Management
st.sidebar.header("Configuration Management")
config_option = st.sidebar.radio(
    "Configuration",
    ["Create New", "Import Existing"],
    key="config_option"
)

if config_option == "Import Existing":
    config_string = st.sidebar.text_input("Paste configuration string")
    if config_string:
        imported_config = import_config(config_string)
        if imported_config:
            st.sidebar.success("Configuration imported successfully!")

# Step 1: User Inputs with Enhanced Controls
st.sidebar.header("User Input Parameters")

# Date Range Selection
st.sidebar.subheader("Historical Data Range")
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=1825))  # Default to ~10 years

if start_date >= end_date:
    st.error("Start date must be before end date")
    st.stop()

# Ticker input and validation
default_tickers = imported_config["tickers"] if config_option == "Import Existing" and imported_config else "VOO, VGT, FBND"
tickers_input = st.sidebar.text_input("Enter the tickers (comma-separated)", default_tickers)
tickers = [tick.strip() for tick in tickers_input.split(",") if tick.strip()]

# Weight input and validation
default_weights = imported_config["weights"] if config_option == "Import Existing" and imported_config else "0.4, 0.4, 0.2"
weights_input = st.sidebar.text_input("Enter the corresponding weights (comma-separated)", default_weights)
try:
    weights = [float(w.strip()) for w in weights_input.split(",") if w.strip()]
    
    if len(weights) != len(tickers):
        st.error(f"Number of weights ({len(weights)}) must match number of tickers ({len(tickers)})")
        st.stop()
    
    if not 0.99 <= sum(weights) <= 1.01:
        st.warning(f"Weights sum to {sum(weights):.2f}. They should sum to 1.0")
        weights = [w/sum(weights) for w in weights]
        st.write("Weights have been normalized to:", [f"{w:.2f}" for w in weights])
    
except ValueError as e:
    st.error("Please enter valid numerical weights (e.g., 0.4, 0.3, 0.3)")
    st.stop()

# Enhanced Simulation Period Selection
st.sidebar.subheader("Simulation Period")
period_unit = st.sidebar.selectbox(
    "Select period unit",
    ["Trading Days", "Months", "Years"]
)

if period_unit == "Trading Days":
    time_horizon = st.sidebar.number_input(
        "Number of trading days",
        min_value=1,
        max_value=252*10,  # 10 years max
        value=252,
        step=1
    )
elif period_unit == "Months":
    months = st.sidebar.number_input(
        "Number of months",
        min_value=1,
        max_value=120,  # 10 years max
        value=12,
        step=1
    )
    time_horizon = int(months * 21)  # Approximate trading days per month
else:  # Years
    years = st.sidebar.number_input(
        "Number of years",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )
    time_horizon = int(years * 252)  # Trading days per year

# Other simulation parameters
num_simulations = st.sidebar.number_input(
    "Number of simulations",
    min_value=100,
    max_value=5000,
    step=100,
    value=imported_config.get("num_simulations", 1000) if config_option == "Import Existing" and imported_config else 1000
)

# Fixed type consistency for initial_value input
initial_value = st.sidebar.number_input(
    "Initial portfolio value",
    min_value=1000.0,
    max_value=100000000.0,
    value=float(imported_config.get("initial_value", 10000.0)) if config_option == "Import Existing" and imported_config else 10000.0,
    step=1000.0,
    format="%.2f"
)

# Create configuration dictionary
current_config = {
    "tickers": tickers_input,
    "weights": weights_input,
    "start_date": start_date.strftime("%Y-%m-%d"),
    "end_date": end_date.strftime("%Y-%m-%d"),
    "period_unit": period_unit,
    "time_horizon": time_horizon,
    "num_simulations": num_simulations,
    "initial_value": initial_value
}

# Export configuration
st.sidebar.subheader("Export Configuration")
config_string = export_config(current_config)
st.sidebar.text_area("Configuration String (Copy to share)", config_string, height=100)

# Step 2: Download Historical Data with Error Handling
st.write(f"Fetching historical data for {tickers}...")
try:
    # Validate tickers individually to catch invalid symbols
    for t in tickers:
        try:
            yf.Ticker(t).history(period="1d")
        except Exception as ticker_err:
            st.error(f"Ticker '{t}' appears invalid or data is not available: {ticker_err}")
            st.stop()

    data = yf.download(tickers, start=start_date, end=end_date, group_by="column")

    # yfinance may return either 'Adj Close' or only 'Close'.
    if 'Adj Close' in data.columns:
        data = data['Adj Close']
    elif 'Close' in data.columns:
        data = data['Close']
    else:
        raise KeyError("Neither 'Adj Close' nor 'Close' data available")

    # yf.download returns a Series when only one ticker is provided. Convert it
    # to a DataFrame so downstream operations work consistently.
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0])
    
    if data.empty:
        st.error("No data retrieved. Please check if the ticker symbols are correct.")
        st.stop()
    
    missing_data = data.isnull().sum()
    if missing_data.any():
        st.warning(f"Missing data found:\n{missing_data[missing_data > 0]}")
        data = data.fillna(method='ffill')
    
    st.write("Data fetched successfully!")
    
except Exception as e:
    if isinstance(e, requests.exceptions.RequestException):
        st.error(
            "Network request failed while downloading data. "
            "Please check your internet connection and try again."
        )
    else:
        st.error(f"Error downloading data: {e}")
    st.stop()

# Step 3: Calculate Daily Returns
daily_returns = data.pct_change().dropna()

# Step 4: Mean Returns and Covariance Matrix
mean_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Step 5: Monte Carlo Simulation Function
@st.cache_data
def run_monte_carlo_simulations(num_simulations, time_horizon, weights, mean_returns, cov_matrix, initial_value):
    simulation_results = np.zeros((num_simulations, time_horizon))
    
    for i in range(num_simulations):
        random_returns = np.random.multivariate_normal(mean_returns, cov_matrix, time_horizon)
        portfolio_values = [initial_value]
        
        for t in range(time_horizon):
            portfolio_return = np.dot(weights, random_returns[t])
            if t == 0:
                portfolio_values.append(initial_value * (1 + portfolio_return))
            else:
                portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))
        
        simulation_results[i, :] = portfolio_values[1:]
    
    return simulation_results

# Run Simulations
try:
    simulation_results = run_monte_carlo_simulations(
        num_simulations, 
        time_horizon, 
        weights, 
        mean_returns, 
        cov_matrix,
        initial_value
    )

    # Step 6: Display Results
    final_values = simulation_results[:, -1]

    # Risk Metrics
    final_returns = final_values / initial_value - 1
    var_5 = np.percentile(final_returns, 5)
    cvar_5 = final_returns[final_returns <= var_5].mean()
    portfolio_mean = np.dot(weights, mean_returns)
    portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_mean / portfolio_std) * np.sqrt(252)
    
    # Create two columns for statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Summary Statistics")
        st.write(f"Mean Final Value: ${np.mean(final_values):,.2f}")
        st.write(f"Median Final Value: ${np.median(final_values):,.2f}")
        st.write(f"Standard Deviation: ${np.std(final_values):,.2f}")
        st.write(f"Minimum Value: ${np.min(final_values):,.2f}")
        st.write(f"Maximum Value: ${np.max(final_values):,.2f}")

        st.subheader("Risk Metrics")
        st.write(f"5% Value at Risk (VaR): {var_5*100:.2f}%")
        st.write(f"5% Conditional VaR (CVaR): {cvar_5*100:.2f}%")
        st.write(f"Sharpe Ratio: {sharpe_ratio:.2f}")

    with col2:
        st.subheader("Percentile Analysis")
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = np.percentile(final_values, percentiles)
        for p, v in zip(percentiles, percentile_values):
            st.write(f"{p}th percentile: ${v:,.2f}")

    # Visualization
    st.subheader("Portfolio Value Distribution")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=final_values, kde=True, ax=ax)
    ax.set_title('Distribution of Final Portfolio Values')
    ax.set_xlabel('Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    plt.close()

    # Add time series plot of some sample paths
    st.subheader("Sample Simulation Paths")
    fig, ax = plt.subplots(figsize=(12, 6))
    num_paths = 50  # Number of sample paths to display
    for i in range(min(num_paths, num_simulations)):
        ax.plot(simulation_results[i, :])
    ax.set_title('Sample Portfolio Value Paths')
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Portfolio Value ($)')
    st.pyplot(fig)
    plt.close()

    # Allow users to download simulation results as CSV
    st.subheader("Export Results")
    download_df = pd.DataFrame(
        simulation_results,
        columns=[f"Step_{i+1}" for i in range(time_horizon)]
    )
    download_df.insert(0, "Final Value", final_values)
    csv_data = download_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="simulation_results.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error(f"Error during simulation: {e}")
    st.stop()

