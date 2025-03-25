import streamlit as st 
from main import extractor_agent
import asyncio
from tempfile import NamedTemporaryFile
import pandas as pd
# from sample_output import result

def init():
    st.set_page_config(
        page_title="Financial Statement Extractor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    session_state_items = [
        "equity_details", "account_summary", "alternative_assets",
        "portfolio_activity", "transactions_summary", "fixed_income",
        "trade_activity", "active_tab", "extracted"
    ]
    
    for item in session_state_items:
        if item not in st.session_state:
            if item == "extracted":
                st.session_state[item] = False
            else:
                st.session_state[item] = [] if item != "active_tab" else "Overview"

init()

st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 2rem;
        border-radius: 15px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Financial Statement Extractor")
st.markdown("---")

async def process_file(temp_file):
    try:
        result = await extractor_agent(temp_file.name)
        state_items = [
            'equity_details', 'account_summary', 'alternative_assets',
            'portfolio_activity', 'transactions_summary', 'fixed_income',
            'trade_activity'
        ]
        for item in state_items:
            st.session_state[item] = result[item]
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        temp_file.close()

def display_metric(label, value, delta=""):
    st.metric(label, f"${value:,.2f}" if isinstance(value, (int, float)) else value, delta)

def display_dataframe(data, message):
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info(f"🔍 No {message} found in the document")

with st.container():
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded_file = st.file_uploader("📄 Upload your financial statement (PDF)", key="input_file")
    with col2:
        if uploaded_file:
            st.success("File uploaded successfully!")
            if st.button("🚀 Extract Data", key="extract_button"):
                with st.spinner("Processing your document..."):
                    temp_file = NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(process_file(temp_file))
                st.session_state.extracted = True

if uploaded_file and st.session_state.extracted:
    tabs = st.tabs(["Overview", "Equity Details", "Account Summary", "Alternative Assets", 
                    "Portfolio Activity", "Transactions", "Fixed Income", "Trade Activity"])
    
    with tabs[0]:
        st.subheader("📈 Financial Overview")
        if st.session_state.account_summary:
            metrics_container = st.container()
            with metrics_container:
                col1, col2, col3 = st.columns(3)
                metrics = [
                    ("Name", st.session_state.account_summary['account_name']),
                    ("Account Number", st.session_state.account_summary['account_name']),
                    ("Date of statement", st.session_state.account_summary['date_of_statement']),
                    ("Name of the Custodian", st.session_state.account_summary['name_of_custodian']),
                    ("Total Realized Gain/Loss", st.session_state.account_summary['total_realized_gain_loss']),
                    ("Short Term G/L", st.session_state.account_summary['total_st_realized_gain_loss']),
                    ("Long Term G/L", st.session_state.account_summary['total_lt_realized_gain_loss']),
                ]
                for col, (label, value) in zip([col1, col2, col3], metrics):
                    with col:
                        display_metric(label, value)
    
    with tabs[1]:
        st.subheader("📊 Equity Details")
        display_dataframe(
            st.session_state.equity_details["equity_details"],
            "equity details"
        )
    
    with tabs[2]:
        st.subheader("💰 Account Summary")
        if st.session_state.account_summary:
            col1, col2 = st.columns(2)
            with col1:
                display_metric("Total Realized Gain/Loss", 
                             st.session_state.account_summary['total_realized_gain_loss'])
            with col2:
                display_metric("Unrealized Gain/Loss", 
                             st.session_state.account_summary['unrealized_gain_loss'])
        else:
            st.info("🔍 No account summary found in the document")
    
    with tabs[3]:
        st.subheader("🏢 Alternative Assets")
        display_dataframe(st.session_state.alternative_assets, "alternative assets")
    
    with tabs[4]:
        st.subheader("📈 Portfolio Activity")
        display_dataframe(st.session_state.portfolio_activity, "portfolio activity")
    
    with tabs[5]:
        st.subheader("💳 Transactions Summary")
        if st.session_state.transactions_summary:
            col1, col2 = st.columns(2)
            with col1:
                display_metric("Beginning Cash Balance", st.session_state.transactions_summary["beginning_cash_balance"])
            with col2:
                display_metric("Ending Cash Balance", st.session_state.transactions_summary["ending_cash_balance"])
        else:
            st.info("🔍 No transactions summary found in the document")
    
    with tabs[6]:
        st.subheader("📋 Fixed Income")
        display_dataframe(st.session_state.fixed_income, "fixed income securities")
    
    with tabs[7]:
        st.subheader("🔄 Trade Activity")
        display_dataframe(st.session_state.trade_activity, "trade activity")