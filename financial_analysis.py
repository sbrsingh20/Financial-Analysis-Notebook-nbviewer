# app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pandas import ExcelWriter

# Set style for seaborn and matplotlib
sns.set(style="whitegrid")

# Title of the app
st.title("Financial Analysis Dashboard")

# Editable code section
code = """
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import ExcelWriter

# Set style for seaborn and matplotlib
sns.set(style="whitegrid")

# Function definitions for profitability and growth analysis
def profitability_analysis(df):
    df['Gross Margin'] = (df['Total Revenue/Income'] - df['Total Operating Expense']) / df['Total Revenue/Income'] * 100
    df['Operating Margin'] = df['Operating Income/Profit'] / df['Total Revenue/Income'] * 100
    df['Net Profit Margin'] = df['Net Income'] / df['Total Revenue/Income'] * 100
    return df[['Date', 'Gross Margin', 'Operating Margin', 'Net Profit Margin']]

def growth_analysis(df):
    df['Revenue Growth'] = df['Total Revenue/Income'].pct_change() * 100
    df['Net Income Growth'] = df['Net Income'].pct_change() * 100
    return df[['Date', 'Revenue Growth', 'Net Income Growth']]
"""

# File upload section
uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

# Initialize data containers
all_profit_data = {}
all_growth_data = {}
comparative_metrics = {}

if uploaded_files:
    # Display editable code section
    st.subheader("Editable Code Section")
    editable_code = st.text_area("Modify the code below:", code, height=300)

    # Check if the code was modified and execute the analysis
    try:
        exec(editable_code)
        
        for uploaded_file in uploaded_files:
            data = pd.read_csv(uploaded_file)

            # Convert Date column and sort
            data['Date'] = pd.to_datetime(data['Date'], format="%b-%y", errors='coerce')
            data.dropna(subset=['Date'], inplace=True)
            data.sort_values('Date', inplace=True)

            # Perform analyses
            profit_data = profitability_analysis(data)
            growth_data = growth_analysis(data)

            # Add file identifier
            file_id = uploaded_file.name.split('.')[0]

            # Store data in dictionaries with file-based keys
            all_profit_data[f"{file_id} Combined Profitability"] = profit_data
            all_growth_data[f"{file_id} Combined Growth"] = growth_data

            # Store comparative metrics
            comparative_metrics[file_id] = {
                "Avg Gross Margin": profit_data['Gross Margin'].mean(),
                "Avg Operating Margin": profit_data['Operating Margin'].mean(),
                "Avg Net Profit Margin": profit_data['Net Profit Margin'].mean(),
                "Avg Revenue Growth": growth_data['Revenue Growth'].mean(),
                "Avg Net Income Growth": growth_data['Net Income Growth'].mean()
            }

        # Convert comparative metrics to DataFrame
        comparative_df = pd.DataFrame(comparative_metrics).T
        comparative_df.index.name = "File"
        comparative_df.reset_index(inplace=True)

        # Display the comparative metrics DataFrame
        st.subheader("Comparative Metrics")
        st.dataframe(comparative_df)

        # Plot comparison graphs
        plt.figure(figsize=(12, 8))
        comparative_df.plot(x="File", kind="bar", stacked=False)
        plt.title("Comparison of Key Financial Metrics Across Files")
        plt.ylabel("Metric Averages")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("financial_metric_comparison.png")
        plt.close()

        # Show the plot in the Streamlit app
        st.subheader("Comparison of Key Financial Metrics")
        img = plt.imread('financial_metric_comparison.png')
        st.image(img)

        # Provide download link for the Excel file
        output_file = "financial_comparison_analysis.xlsx"
        with ExcelWriter(output_file) as writer:
            for sheet_name, df in all_profit_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            for sheet_name, df in all_growth_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            comparative_df.to_excel(writer, sheet_name="Comparative Metrics", index=False)

        st.success(f"Analysis complete. Results saved to {output_file}.")
        st.markdown(f"[Download Excel File]({output_file})")
    except Exception as e:
        st.error(f"Error executing the code: {str(e)}")
else:
    st.warning("Please upload CSV files to perform analysis.")
