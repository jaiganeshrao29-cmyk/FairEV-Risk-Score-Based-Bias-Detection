import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os

# --- Configuration & Initialization ---
st.set_page_config(page_title="FairEV: Bias Detection System", page_icon="⚡", layout="wide")

# Custom CSS for Dark Theme Professional look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Modern Metric Cards - Dark Mode */
    div[data-testid="metric-container"] {
        background-color: #1e293b; /* Secondary BG */
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    
    /* Custom Alerts */
    .alert-success {
        padding: 15px;
        border-radius: 8px;
        background-color: #064e3b;
        border-left: 5px solid #10b981;
        color: #d1fae5;
    }
    .alert-error {
        padding: 15px;
        border-radius: 8px;
        background-color: #450a0a;
        border-left: 5px solid #ef4444;
        color: #fee2e2;
    }
    .alert-info {
        padding: 30px;
        border-radius: 12px;
        background-color: #0f172a;
        border-left: 5px solid #3b82f6;
        border: 1px solid #334155;
        color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# Setup Gemini API
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAaNDfTH738HASgYn5FPU6uJa424buJqCA")
if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

# --- Helper Functions ---
@st.cache_data
def generate_dataset(n=30):
    np.random.seed(42)
    data = {
        "Applicant_ID": [f"APP{str(i).zfill(3)}" for i in range(1, n+1)],
        "Income": np.random.randint(20000, 150000, n),
        "Location": np.random.choice(["Urban", "Rural"], n, p=[0.6, 0.4]),
        "Vehicle_Price": np.random.randint(25000, 80000, n),
        "Family_Size": np.random.randint(1, 6, n),
        "Previous_Subsidy": np.random.choice([0, 1], n, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

def calculate_risk_score(df, w_income, w_price, w_location, w_subsidy):
    income_norm = (df['Income'] - df['Income'].min()) / (df['Income'].max() - df['Income'].min())
    price_norm = (df['Vehicle_Price'] - df['Vehicle_Price'].min()) / (df['Vehicle_Price'].max() - df['Vehicle_Price'].min())
    location_norm = df['Location'].apply(lambda x: 1 if x == 'Urban' else 0)
    subsidy_norm = df['Previous_Subsidy']
    
    score = (w_income * income_norm + w_price * price_norm + w_location * location_norm + w_subsidy * subsidy_norm)
    
    max_possible = w_income + w_price + w_location + w_subsidy
    if max_possible == 0:
        score_100 = score * 0
    else:
        score_100 = (score / max_possible) * 100
    
    return score_100.round(2)

def calculate_metrics(df):
    urban_df = df[df['Location'] == 'Urban']
    rural_df = df[df['Location'] == 'Rural']
    
    urban_approval_rate = (urban_df['Decision'] == 'Approved').mean() if not urban_df.empty else 0
    rural_approval_rate = (rural_df['Decision'] == 'Approved').mean() if not rural_df.empty else 0
    
    bias_gap = abs(urban_approval_rate - rural_approval_rate)
    fairness_score = max(0, (1 - bias_gap) * 100)
    
    return urban_approval_rate, rural_approval_rate, bias_gap, fairness_score

# --- Main App ---
def main():
    st.sidebar.markdown("## ⚡ FairEV Platform")
    st.sidebar.caption("Bias Detection & Mitigation System")
    st.sidebar.divider()
    pages = ["Home", "Dataset & Risk Score", "Bias Dashboard", "AI Insights & Chat", "Bias Simulation", "Final Report"]
    selection = st.sidebar.radio("Navigation", pages)
    
    st.sidebar.divider()
    st.sidebar.info("💡 **Tip:** Use the Bias Simulation to adjust algorithm weights and see how it impacts fairness metrics in real-time.")

    # Initialize session state for dataset
    if 'df' not in st.session_state:
        st.session_state.df = generate_dataset()

    # Default weights
    if 'weights' not in st.session_state:
        st.session_state.weights = {'w_income': 0.4, 'w_price': 0.3, 'w_location': 0.2, 'w_subsidy': 0.1}
        
    df = st.session_state.df.copy()
    
    df['Risk_Score'] = calculate_risk_score(df, **st.session_state.weights)
    df['Decision'] = df['Risk_Score'].apply(lambda x: 'Approved' if x < 50 else 'Rejected')

    if selection == "Home":
        st.title("⚡ FairEV: Automated Decision Fairness")
        st.markdown("""
        <div class="alert-info">
            <h3 style="margin-top: 0; color: #60a5fa;">Welcome to FairEV</h3>
            <p style="font-size: 1.1em; color: #cbd5e1;">
            Automated systems are increasingly used to make critical decisions, such as EV subsidy allocation. 
            However, algorithms can inadvertently penalize demographic groups if left unchecked.
            </p>
            <p style="font-size: 1.1em; color: #cbd5e1;">
            This platform serves as an interactive auditor, helping you <b>detect</b>, <b>explain</b>, and <b>mitigate</b> algorithmic bias.
            </p>
        </div>
        <br>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 📊 Analyze")
            st.write("Generate synthetic applicant data and apply weighted risk scoring.")
        with col2:
            st.markdown("### ⚖️ Detect")
            st.write("Automatically calculate the Bias Gap between Urban and Rural populations.")
        with col3:
            st.markdown("### 🔄 Mitigate")
            st.write("Simulate weight changes and consult AI for fairness recommendations.")

    elif selection == "Dataset & Risk Score":
        st.title("📊 Dataset & Risk Score")
        
        tab1, tab2 = st.tabs(["Applicant Dataset", "Risk Scoring System"])
        
        with tab1:
            st.write("Preview of the synthetic applicant data pool.")
            st.dataframe(df.drop(columns=['Risk_Score', 'Decision'], errors='ignore'), use_container_width=True)
            
        with tab2:
            st.write("Risk Score is based on normalized inputs and configurable weights. **Score < 50 = Approved.**")
            
            cols = st.columns(4)
            weight_keys = list(st.session_state.weights.keys())
            labels = ["Income", "Vehicle Price", "Location", "Prev Subsidy"]
            for i, col in enumerate(cols):
                col.metric(f"{labels[i]} Weight", st.session_state.weights[weight_keys[i]])
                
            st.divider()
            
            def highlight_decision(val):
                return 'background-color: #064e3b; color: #a7f3d0; font-weight: bold;' if val == 'Approved' else 'background-color: #450a0a; color: #fecaca; font-weight: bold;'
                
            st.write("### Scored Results")
            st.dataframe(df.style.map(highlight_decision, subset=['Decision']), use_container_width=True)

    elif selection == "Bias Dashboard":
        st.title("⚖️ Fairness Analytics Dashboard")
        
        u_rate, r_rate, bias_gap, fair_score = calculate_metrics(df)
        
        if bias_gap > 0.20:
            st.markdown(f"""
            <div class="alert-error">
                <span style="font-weight: 600; font-size: 1.1em;">⚠️ Alert: System Bias Detected</span><br>
                The gap between demographic groups exceeds the acceptable threshold (0.20). Mitigation required.
            </div>
            <br>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-success">
                <span style="font-weight: 600; font-size: 1.1em;">✅ System is Fair</span><br>
                The approval rates are balanced across demographic groups.
            </div>
            <br>
            """, unsafe_allow_html=True)
            
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fairness Score", f"{fair_score:.1f}/100")
        m2.metric("Bias Gap", f"{bias_gap:.2f}")
        m3.metric("Urban Approval", f"{u_rate*100:.1f}%")
        m4.metric("Rural Approval", f"{r_rate*100:.1f}%")
        
        st.write("### Demographic Fairness Indicator")
        st.progress(fair_score / 100.0)
        
        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Risk Score Distribution")
            fig = px.histogram(df, x="Risk_Score", nbins=10, 
                               color_discrete_sequence=['#3b82f6'],
                               opacity=0.8,
                               labels={'Risk_Score':'Risk Score'})
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_title="Count",
                font_color="#f8fafc"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("#### Approval Rates by Group")
            chart_df = pd.DataFrame({
                'Location': ['Urban', 'Rural'],
                'Approval Rate (%)': [u_rate*100, r_rate*100]
            })
            fig2 = px.bar(chart_df, x='Location', y='Approval Rate (%)', 
                          color='Location',
                          color_discrete_map={'Urban': '#8b5cf6', 'Rural': '#10b981'},
                          text='Approval Rate (%)')
            fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_range=[0, 100],
                showlegend=False,
                font_color="#f8fafc"
            )
            st.plotly_chart(fig2, use_container_width=True)

    elif selection == "AI Insights & Chat":
        st.title("🤖 AI-Powered Fairness Assistant")
        st.write("Leverage Google Gemini to analyze metrics and ask complex questions regarding AI ethics.")
        
        u_rate, r_rate, bias_gap, fair_score = calculate_metrics(df)
        
        tab1, tab2 = st.tabs(["AI Audit Report", "Interactive Chatbot"])
        
        with tab1:
            st.markdown("### Automated Fairness Audit")
            st.write("Generate a comprehensive explanation of the current bias metrics.")
            
            if st.button("Generate Audit Report", type="primary"):
                with st.spinner("Analyzing metrics and generating report..."):
                    prompt = f"""
                    You are an expert AI Auditor assessing an EV subsidy allocation system.
                    Current Metrics:
                    - Urban Approval Rate: {u_rate*100:.1f}%
                    - Rural Approval Rate: {r_rate*100:.1f}%
                    - Bias Gap: {bias_gap:.2f} (Acceptable Threshold: <0.20)
                    - Overall Fairness Score: {fair_score:.1f}/100
                    
                    Please provide a professional, structured report covering:
                    1. **Executive Summary**: Is the system currently fair?
                    2. **Root Cause Analysis**: Why the location feature (where Urban carries a risk penalty) might be skewing results.
                    3. **Actionable Recommendations**: 3 concise steps to mitigate this bias.
                    Format with markdown headers and bullet points.
                    """
                    try:
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Failed to generate report: {e}")
                        
        with tab2:
            st.markdown("### Fairness Q&A")
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Welcome. I am the FairEV AI Assistant. How can I help you understand algorithmic fairness today?"}]
                
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
            if prompt := st.chat_input("Ask about AI bias, the current dataset, or mitigation strategies..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Processing..."):
                        try:
                            history = "You are an AI assistant specialized in algorithmic fairness.\n"
                            for msg in st.session_state.messages[:-1]:
                                role = "User: " if msg["role"] == "user" else "Assistant: "
                                history += f"{role}{msg['content']}\n"
                            history += f"User: {prompt}\nAssistant: "
                            
                            response = model.generate_content(history)
                            st.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except Exception as e:
                            st.error(f"Error: {e}")

    elif selection == "Bias Simulation":
        st.title("🔄 Algorithmic Mitigation Sandbox")
        st.write("Tune the weights of the risk scoring algorithm to observe the real-time impact on demographic fairness.")
        
        u_rate, r_rate, bias_gap, fair_score = calculate_metrics(df)
        
        col_sim1, col_sim2 = st.columns([1, 2])
        
        with col_sim1:
            st.markdown("### Adjust Model Weights")
            with st.container(border=True):
                new_w_inc = st.slider("Income Weight", 0.0, 1.0, st.session_state.weights['w_income'], 0.05)
                new_w_pri = st.slider("Vehicle Price Weight", 0.0, 1.0, st.session_state.weights['w_price'], 0.05)
                new_w_loc = st.slider("Location Weight (Sensitive)", 0.0, 1.0, st.session_state.weights['w_location'], 0.05)
                new_w_sub = st.slider("Previous Subsidy Weight", 0.0, 1.0, st.session_state.weights['w_subsidy'], 0.05)
                
                if st.button("Deploy New Weights", type="primary"):
                    st.session_state.weights = {
                        'w_income': new_w_inc,
                        'w_price': new_w_pri,
                        'w_location': new_w_loc,
                        'w_subsidy': new_w_sub
                    }
                    st.rerun()
                    
        with col_sim2:
            st.markdown("### Simulated Outcomes vs Baseline")
            
            temp_df = df.copy()
            temp_df['Risk_Score'] = calculate_risk_score(temp_df, new_w_inc, new_w_pri, new_w_loc, new_w_sub)
            temp_df['Decision'] = temp_df['Risk_Score'].apply(lambda x: 'Approved' if x < 50 else 'Rejected')
            temp_u, temp_r, temp_gap, temp_fair = calculate_metrics(temp_df)
            
            mc1, mc2 = st.columns(2)
            mc1.metric("Baseline Fairness", f"{fair_score:.1f}/100")
            mc2.metric("Simulated Fairness", f"{temp_fair:.1f}/100", delta=f"{temp_fair - fair_score:.1f}")
            
            mc3, mc4 = st.columns(2)
            mc3.metric("Baseline Bias Gap", f"{bias_gap:.2f}")
            mc4.metric("Simulated Bias Gap", f"{temp_gap:.2f}", delta=f"{temp_gap - bias_gap:.2f}", delta_color="inverse")
            
            st.divider()
            st.markdown("#### Recommendations for Mitigation")
            st.info("💡 **Tip**: Sensitive attributes like 'Location' can often be proxies for wealth or race. Reducing its weight usually bridges the Bias Gap. Notice how dropping the Location Weight toward 0 impacts the Simulated Fairness score.")

    elif selection == "Final Report":
        st.title("📄 Executive Final Report")
        
        with st.container(border=True):
            st.markdown("""
            ### 🎯 Project Objective
            To build a resilient, interactive prototype that demonstrates the lifecycle of automated decision-making—focusing specifically on the **detection, measurement, and mitigation** of algorithmic bias in Electric Vehicle subsidy allocation.
            
            ### 🔬 Methodology & Architecture
            - **Data Generation:** Synthetic dataset modeling demographic features (Income, Location, Vehicle Price).
            - **Algorithmic Engine:** A weighted risk-scoring mechanism mapping applicant data to continuous risk probabilities.
            - **Fairness Auditing:** Calculation of Approval Rates across distinct populations to surface the *Bias Gap* and compute a normalized *Fairness Score*.
            - **AI Integration (Gemini):** Embedded Large Language Models to translate complex metrics into actionable insights and provide a conversational interface for auditing.
            - **Mitigation Sandbox:** Real-time simulation environment enabling policymakers to adjust algorithmic weights and preview the demographic impact before deployment.
            
            ### 📈 Results & Findings
            The baseline algorithm demonstrated that including demographic proxies (like penalizing Urban locations) rapidly creates a Bias Gap exceeding acceptable thresholds (>0.20). Through the Mitigation Sandbox, it was proven that re-calibrating the weights (e.g., heavily reducing the influence of Location) successfully restores parity and boosts the Fairness Score toward 100/100.
            
            ### 🏁 Conclusion
            Transparency alone is insufficient for ethical AI. Automated systems require **active monitoring dashboards** and **dynamic mitigation tools** built directly into the application layer. This prototype successfully models how human-in-the-loop oversight can collaborate with AI auditing tools to ensure equitable resource distribution.
            """)

if __name__ == "__main__":
    main()
