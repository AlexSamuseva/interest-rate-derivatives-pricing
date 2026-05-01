"""Streamlit dashboard for displaying interest rate yield curves from FRED."""

import streamlit as st
import plotly.graph_objects as go
import datetime
from interest_rate_derivatives.market_data import MarketDataClient

# Page configuration
st.set_page_config(
    page_title="Interest Rate Yield Curve",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("📈 Interest Rate Yield Curve Dashboard")
st.markdown(
    "Real-time treasury yield curve data fetched from **FRED** (Federal Reserve Economic Data)"
)

# Sidebar controls
st.sidebar.header("Configuration")

provider = st.sidebar.selectbox(
    "Data Provider",
    options=["fred"],
    help="Select the data provider for interest rate data"
)

# API Key input
st.sidebar.subheader("FRED API Key")
api_key_input = st.sidebar.text_input(
    "Enter your FRED API Key (optional)",
    type="password",
    help="Get a free API key at https://fred.stlouisfed.org/user/register. Leave blank to use .env file or placeholder data."
)

date_input = st.sidebar.date_input(
    "Select Date",
    value=datetime.date.today(),
    help="Choose the date for which to fetch the yield curve"
)

# Fetch data button
if st.sidebar.button("🔄 Fetch Data", use_container_width=True):
    st.session_state.fetch_requested = True

# Display API key status
if api_key_input:
    st.sidebar.success("✓ API key configured (via UI)")
else:
    st.sidebar.info("ℹ️ Using .env file or placeholder data")

# Fetch data
@st.cache_data(ttl=3600)
def get_yield_curve(provider: str, date: str, api_key: str = None):
    """Fetch yield curve data with caching."""
    client = MarketDataClient(provider=provider, api_key=api_key)
    return client.get_term_structure(date=date if date != datetime.date.today().isoformat() else None)

# Display content
if st.sidebar.button("Refresh", use_container_width=True) or True:  # Always fetch on load
    with st.spinner("Fetching data from FRED..."):
        try:
            date_str = date_input.isoformat()
            # Pass API key if provided
            curve_data = get_yield_curve(provider, date_str, api_key=api_key_input if api_key_input else None)
            
            # Create two columns
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Create interactive plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=curve_data['Maturity'],
                    y=curve_data['Rate'] * 100,  # Convert to percentage
                    mode='lines+markers',
                    name='Yield Curve',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=10, color='#1f77b4'),
                    hovertemplate='<b>Maturity:</b> %{x:.2f} years<br><b>Yield:</b> %{y:.3f}%<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"Treasury Yield Curve - {date_input.strftime('%B %d, %Y')}",
                    xaxis_title="Maturity (Years)",
                    yaxis_title="Yield (%)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    font=dict(size=12),
                    margin=dict(l=60, r=40, t=60, b=60),
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Curve Statistics")
                st.metric(
                    "Shortest Maturity Yield",
                    f"{curve_data['Rate'].min() * 100:.3f}%",
                    delta=None
                )
                st.metric(
                    "Longest Maturity Yield",
                    f"{curve_data['Rate'].max() * 100:.3f}%",
                    delta=None
                )
                st.metric(
                    "Curve Slope (30Y - 3M)",
                    f"{(curve_data['Rate'].max() - curve_data['Rate'].min()) * 100:.3f}%",
                    delta=None
                )
            
            # Display data table
            st.subheader("Detailed Data")
            display_df = curve_data.copy()
            display_df['Rate'] = (display_df['Rate'] * 100).round(3).astype(str) + '%'
            display_df['Maturity'] = display_df['Maturity'].round(2).astype(str) + ' years'
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Maturity": st.column_config.Column(width="medium"),
                    "Rate": st.column_config.Column(width="medium"),
                }
            )
            
            # Data info
            st.info(
                f"✓ Data fetched successfully from {provider.upper()} for {date_input.strftime('%B %d, %Y')}"
            )
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.info(
                "💡 **Tip:** Enter your FRED API key in the sidebar for real market data, "
                "or configure it in `.env` file. Without an API key, you'll see placeholder data."
            )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
    Data source: <a href='https://fred.stlouisfed.org/'>FRED - Federal Reserve Economic Data</a>
    </div>
    """,
    unsafe_allow_html=True
)
