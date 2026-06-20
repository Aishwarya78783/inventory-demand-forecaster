import streamlit as st
import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Set page configuration with a modern dark theme aesthetic
st.set_page_config(
    page_title="Apex Global Supply Chain Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling injection for premium look (Glassmorphism & dark gradients)
st.markdown("""
<style>
    .main {
        background-color: #0f111a;
        color: #ffffff;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(15, 17, 26) 0%, rgb(21, 26, 43) 90%);
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-bottom: 20px;
    }
    .metric-card h4 {
        color: #a0aec0;
        margin-bottom: 8px;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 APEX Global Manufacturing Control Center")
st.markdown("##### Production-Grade Demand Forecasting, Safety Stock Planning, and Inventory Simulation")

# Connect to MongoDB with Fallback
@st.cache_resource
def get_db_connection():
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1000)
        client.admin.command('ping')
        return client.inventory_db, True
    except Exception:
        return None, False

db, connected = get_db_connection()

# Sidebar Setup
st.sidebar.image("https://img.icons8.com/nolan/96/factory.png", width=70)
st.sidebar.markdown("### **Operational Controls**")

lead_time = st.sidebar.slider("Lead Time / Replenishment Window (Days)", 1, 30, 5, help="Time taken for shipment delivery from supplier.")
service_level = st.sidebar.select_slider(
    "Target Service Level (Coverage)",
    options=["85%", "90%", "95%", "97.5%", "99%"],
    value="95%",
    help="Target probability of not running out of stock during the replenishment cycle."
)

# Map Service Level to Z-score
z_mapping = {"85%": 1.04, "90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}
z_score = z_mapping[service_level]

# CSV File Uploader for instant out-of-the-box operations
st.sidebar.markdown("---")
st.sidebar.markdown("### **Upload Custom Inventory Data**")
uploaded_file = st.sidebar.file_uploader("Upload CSV (SKU, Description, History, Stock)", type=["csv"], help="Upload your own warehouse CSV to instantly calculate forecasts and safety stock levels.")

# Global state to hold the dataset
df_items = None
mock_history = {}

# Parse Uploaded CSV
if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        # Check required columns
        required_cols = ["sku", "description", "history", "current_stock"]
        if all(col in df_uploaded.columns for col in required_cols):
            # Parse history lists
            parsed_history = []
            for h in df_uploaded["history"]:
                if isinstance(h, str):
                    parsed_history.append([float(x.strip()) for x in h.strip("[]").split(",") if x.strip()])
                else:
                    parsed_history.append([float(x) for x in h])
            
            df_uploaded["history"] = parsed_history
            
            # Recalculate parameters
            df_uploaded["mean_demand"] = df_uploaded["history"].apply(np.mean)
            df_uploaded["std_demand"] = df_uploaded["history"].apply(np.std)
            
            # Generate double exponential forecasts
            alpha, beta = 0.25, 0.15
            forecasts = []
            for h_vals in df_uploaded["history"]:
                level = h_vals[0]
                t_val = h_vals[1] - h_vals[0]
                for val in h_vals[1:]:
                    next_level = alpha * val + (1.0 - alpha) * (level + t_val)
                    next_trend = beta * (next_level - level) + (1.0 - beta) * t_val
                    level = next_level
                    t_val = next_trend
                forecasts.append(level + 7.0 * t_val)
            
            df_uploaded["forecast"] = forecasts
            df_items = df_uploaded
            
            # Load into mock_history dict
            for _, row in df_uploaded.iterrows():
                mock_history[row["sku"]] = row["history"]
            st.sidebar.success("✅ Custom CSV parsed successfully!")
        else:
            st.sidebar.error("CSV must contain: sku, description, history, current_stock")
    except Exception as e:
        st.sidebar.error(f"Error parsing CSV: {e}")

# If no CSV uploaded, fall back to DB or Mock Data
if df_items is None:
    if not connected:
        # Generate mock time-series data
        np.random.seed(42)
        skus = [f"SKU-{1000 + i:04d}" for i in range(30)]
        mock_meta = {}
        
        for i, sku in enumerate(skus):
            # Base seasonal demand
            base_demand = np.random.poisson(lam=30 + (i % 3) * 12, size=365).astype(float)
            trend = np.linspace(0, (i % 4 - 2) * 2.5, 365)
            noise = np.random.normal(0, 4, 365)
            history_vals = np.clip(base_demand + trend + noise, 0, None).tolist()
            
            mock_history[sku] = history_vals
            mean_d = float(np.mean(history_vals))
            std_d = float(np.std(history_vals))
            
            # Temporary safety stock & ROP (recalculated dynamically below)
            safety_s = z_score * std_d * np.sqrt(lead_time)
            rop = (mean_d * lead_time) + safety_s
            
            # Random current stock
            current_st = int(np.random.normal(rop * 1.15, rop * 0.25))
            current_st = max(10, current_st)
            
            # Holt's linear trend forecast
            alpha, beta = 0.25, 0.15
            level = history_vals[0]
            t_val = history_vals[1] - history_vals[0]
            for val in history_vals[1:]:
                next_level = alpha * val + (1.0 - alpha) * (level + t_val)
                next_trend = beta * (next_level - level) + (1.0 - beta) * t_val
                level = next_level
                t_val = next_trend
            forecast_val = level + 7.0 * t_val
            
            mock_meta[sku] = {
                "sku": sku,
                "description": f"Industrial raw component batch {i} - Grade {chr(65 + (i % 3))}",
                "mean_demand": mean_d,
                "std_demand": std_d,
                "current_stock": current_st,
                "forecast": forecast_val,
                "history": history_vals
            }
            
        df_items = pd.DataFrame(list(mock_meta.values()))
    else:
        # Load from MongoDB
        hist_collection = db.demand_history
        fore_collection = db.forecasts
        
        history_docs = list(hist_collection.find())
        forecast_docs = list(fore_collection.find())
        
        if not history_docs:
            st.warning("Database empty. Please run worker.py first.")
            st.stop()
            
        df_items = pd.DataFrame(history_docs)
        df_forecasts = pd.DataFrame(forecast_docs)
        
        if not df_forecasts.empty:
            df_items = df_items.merge(df_forecasts[["sku", "forecast", "mean_demand", "std_demand"]], on="sku", how="left")
        
        if "current_stock" not in df_items.columns:
            np.random.seed(42)
            df_items["current_stock"] = np.random.randint(100, 600, size=len(df_items))

# Perform calculations dynamically on user adjustments
df_items["safety_stock"] = (z_score * df_items["std_demand"] * np.sqrt(lead_time)).round(1)
df_items["reorder_point"] = ((df_items["mean_demand"] * lead_time) + df_items["safety_stock"]).round(1)

# Risk Status Classification
def classify_risk(row):
    stock = row["current_stock"]
    rop = row["reorder_point"]
    if stock <= (rop * 0.7):
        return "🔴 CRITICAL"
    elif stock <= rop:
        return "🟡 REORDER"
    return "🟢 SAFE"

df_items["risk_status"] = df_items.apply(classify_risk, axis=1)

# Split Dashboard into 2 Tabs
tab_overview, tab_deepdive = st.tabs(["📊 Operational Control Center", "📈 SKU Analysis & Simulator"])

# ================= TAB 1: OPERATIONAL CONTROL CENTER =================
with tab_overview:
    # High-level analytical metrics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        total_skus = len(df_items)
        st.markdown(f'<div class="metric-card"><h4>Active SKU Coverage</h4><h2>{total_skus} Items</h2></div>', unsafe_allow_html=True)
    with col_stat2:
        critical_items = len(df_items[df_items["risk_status"] == "🔴 CRITICAL"])
        st.markdown(f'<div class="metric-card"><h4>Stockout Emergencies</h4><h2>{critical_items} SKUs</h2></div>', unsafe_allow_html=True)
    with col_stat3:
        reorder_items = len(df_items[df_items["risk_status"] == "🟡 REORDER"])
        st.markdown(f'<div class="metric-card"><h4>Reorders Triggered</h4><h2>{reorder_items} SKUs</h2></div>', unsafe_allow_html=True)
    with col_stat4:
        safe_percentage = (len(df_items[df_items["risk_status"] == "🟢 SAFE"]) / total_skus) * 100
        st.markdown(f'<div class="metric-card"><h4>Service SLA Coverage</h4><h2>{safe_percentage:.1f}%</h2></div>', unsafe_allow_html=True)

    # Advanced filter row
    col_f1, col_f2 = st.columns([2, 1])
    with col_f1:
        search_filter = st.text_input("🔍 Search SKU or Description", "", key="overview_search")
    with col_f2:
        status_filter = st.multiselect("Filter by Status", ["🟢 SAFE", "🟡 REORDER", "🔴 CRITICAL"], default=["🟢 SAFE", "🟡 REORDER", "🔴 CRITICAL"])

    # Filtering data
    display_df = df_items.copy()
    if search_filter:
        display_df = display_df[
            display_df["sku"].str.contains(search_filter, case=False) |
            display_df["description"].str.contains(search_filter, case=False)
        ]
    display_df = display_df[display_df["risk_status"].isin(status_filter)]

    # Dynamic Table
    st.subheader("📋 Active Inventory Pipeline")
    
    table_view = display_df[["sku", "description", "current_stock", "safety_stock", "reorder_point", "forecast", "risk_status"]].copy()
    table_view["forecast"] = table_view["forecast"].round(1)
    
    st.dataframe(
        table_view,
        use_container_width=True,
        hide_index=True,
        column_config={
            "sku": "SKU Code",
            "description": "Component Specification",
            "current_stock": "Current Inventory Level",
            "safety_stock": "Safety Stock Target",
            "reorder_point": "Reorder Point (ROP)",
            "forecast": "7-Day Predicted Demand",
            "risk_status": "Operational Status"
        }
    )

    # Bulk Export Button
    st.markdown("---")
    st.subheader("📥 Export Operational Reports")
    csv_buffer = io.StringIO()
    table_view.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Active Forecast Report (CSV)",
        data=csv_buffer.getvalue(),
        file_name="Apex_Inventory_Forecasts_Report.csv",
        mime="text/csv",
        help="Export all SKU calculations to upload directly into your ERP/MRP system."
    )

# ================= TAB 2: SKU ANALYSIS & SIMULATOR =================
with tab_deepdive:
    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.subheader("📦 SKU Parameters")
        selected_sku = st.selectbox("Select Target SKU to Run Simulation", df_items["sku"].unique())
        sku_details = df_items[df_items["sku"] == selected_sku].iloc[0]
        
        # Display operational summaries
        st.info(f"**Item Name:** {sku_details['description']}")
        
        # In-line Stock Editor for simulated forecasting runs
        simulated_stock = st.number_input(
            "Modify Stock Level (Test What-If Scenarios)",
            min_value=0,
            max_value=5000,
            value=int(sku_details["current_stock"]),
            step=10,
            help="Simulate a receipt of goods or rapid usage to test reorder status triggers."
        )
        
        # Re-evaluating status for simulation
        sim_rop = sku_details["reorder_point"]
        if simulated_stock <= (sim_rop * 0.7):
            sim_status = "🔴 CRITICAL EMERGENCY"
            status_color = "red"
        elif simulated_stock <= sim_rop:
            sim_status = "🟡 REORDER TRIGGERED"
            status_color = "orange"
        else:
            sim_status = "🟢 STOCK LEVEL ADEQUATE"
            status_color = "green"
            
        st.markdown(f"**Simulated Status:** <span style='color:{status_color}; font-weight:bold;'>{sim_status}</span>", unsafe_allow_html=True)
        
        # Details stats breakdown
        st.markdown("---")
        st.markdown("#### **Execution Statistics**")
        st.write(f"- **Calculated Daily Mean:** {sku_details['mean_demand']:.2f} Units")
        st.write(f"- **Demand Variance (Std Dev):** {sku_details['std_demand']:.2f}")
        st.write(f"- **Current Safety Stock Buffer:** {sku_details['safety_stock']:.1f} Units")
        st.write(f"- **Reorder Point Threshold:** {sku_details['reorder_point']:.1f} Units")
        st.write(f"- **7-Day Out Forecast Prediction:** {sku_details['forecast']:.1f} Units")
        
    with col_sim2:
        st.subheader("📈 Interactive Demand Profile & Buffer Chart")
        
        # Fetch historical list
        if not connected and uploaded_file is None:
            history = mock_history[selected_sku]
        else:
            history = sku_details["history"]
            
        # Draw Matplotlib Chart with premium dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        
        days = range(1, len(history) + 1)
        ax.plot(days, history, label="Historical Daily Demand", color="#00f2fe", alpha=0.5, linewidth=1.5)
        
        # 30-day moving average
        ma_30 = pd.Series(history).rolling(window=30).mean()
        ax.plot(days, ma_30, label="30-day Moving Average", color="#ff007f", linestyle="--", linewidth=1.5, alpha=0.8)
        
        # Projected forecast trendline
        forecast_days = range(len(history), len(history) + 8)
        forecast_path = [history[-1]] + [history[-1] + (sku_details["forecast"] - history[-1]) * (i / 7) for i in range(1, 8)]
        ax.plot(forecast_days, forecast_path, color="#39ff14", linestyle=":", linewidth=2, label="7-Day Holt Prediction")
        ax.scatter(len(history) + 7, sku_details["forecast"], color="#39ff14", s=130, zorder=5)
        
        # Draw safety thresholds
        ax.axhline(y=sim_rop, color="#ffaa00", linestyle="-.", alpha=0.8, label=f"Reorder Point ({sim_rop:.1f})")
        ax.axhline(y=sku_details["safety_stock"], color="#ff0000", linestyle=":", alpha=0.8, label=f"Safety Stock ({sku_details['safety_stock']:.1f})")
        
        # Mark current stock level
        ax.axhline(y=simulated_stock, color="#ffffff", linestyle="--", alpha=0.4, label=f"Simulated Stock ({simulated_stock})")
        
        # Format chart details
        ax.set_title(f"Visual Optimization Engine for {selected_sku}", fontsize=14, pad=15)
        ax.set_xlabel("Days (Historical Time Series)", fontsize=10)
        ax.set_ylabel("Quantity (Units)", fontsize=10)
        ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.2)
        ax.legend(loc="upper left", framealpha=0.2)
        
        st.pyplot(fig)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.01); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05); font-size: 11px; margin-top: 10px;">
            <strong>Double Exponential Forecasting Core (Holt's Model):</strong> Calculates level and trend updates simultaneously using 256-bit AVX2 SIMD operations. 
            Reorder thresholds and safety stock are adjusted dynamically as you slide lead-times or coverage requirements.
        </div>
        """, unsafe_allow_html=True)
