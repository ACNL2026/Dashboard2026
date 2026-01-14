import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from google import genai
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# 1. CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Eindhoven Operations Dashboard", page_icon="✈️", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff; padding: 15px; border-radius: 10px;
        border: 1px solid #e2e8f0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .summary-box {
        background-color: #ffffff; padding: 18px; border-radius: 10px;
        border: 1px solid #e2e8f0; height: 100%;
    }
    .ai-box {
        background-color: #f0fdf4; padding: 20px; border-radius: 10px;
        border: 1px solid #bbf7d0; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

if 'run_main' not in st.session_state: st.session_state['run_main'] = False

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def parse_acnl_date(date_str):
    if pd.isna(date_str): return None
    try:
        clean_str = str(date_str).strip()
        return datetime.strptime(clean_str, "%d%b%Y").date()
    except:
        try: return pd.to_datetime(clean_str).date()
        except: return None

def format_time(dt_obj):
    return dt_obj.strftime("%H:%M") if pd.notnull(dt_obj) else "-"

def fetch_history_data_raw(api_key, ident, start_dt, end_dt):
    target_ident = str(ident).replace(" ", "").strip().upper()
    url = f"https://aeroapi.flightaware.com/aeroapi/history/flights/{target_ident}"
    headers = {'x-apikey': api_key}
    start_ts = (datetime.combine(start_dt, datetime.min.time()) - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_ts = (datetime.combine(end_dt, datetime.max.time()) + timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    params = {'start': start_ts, 'end': end_ts, 'max_pages': 2}
    try:
        response = requests.get(url, headers=headers, params=params)
        return [f for f in response.json().get('flights', []) if f.get('scheduled_out')] if response.status_code == 200 else []
    except: return []

# ==========================================
# 3. MAIN DASHBOARD
# ==========================================

with st.sidebar:
    st.header("⚙️ Settings")
    fa_api_key = st.text_input("FlightAware API Key", type="password")
    google_api_key = st.text_input("Google Gemini API Key", type="password")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

st.title("🛫 Eindhoven Operations & Intelligence Dashboard")

df = pd.read_csv(uploaded_file) if uploaded_file else None

if df is not None:
    df_clean = df.dropna(subset=['Date', 'Registration', 'Operated Flight#']).copy()
    df_clean['Label'] = df_clean['Operated Flight#'].astype(str) + " | " + df_clean['Date'].astype(str)
    selected_option = st.selectbox("Search flight rotation:", df_clean['Label'].unique())
    row = df_clean[df_clean['Label'] == selected_option].iloc[0]
    search_date = parse_acnl_date(row['Date'])
    target_flight_num = str(row['Operated Flight#']).strip()

    c_t1, c_t2, c_t3 = st.columns(3)
    c_t1.metric("Selected Flight", target_flight_num)
    c_t2.metric("Aircraft", row['Registration'])
    c_t3.metric("Date", search_date.strftime("%d-%m-%Y") if search_date else "Unknown")

    if st.button("🚀 Start Analysis", use_container_width=True):
        st.session_state['run_main'] = True

    if st.session_state.get('run_main') and fa_api_key:
        with st.spinner("Processing operational data..."):
            flights = fetch_history_data_raw(fa_api_key, row['Registration'], search_date, search_date)
        
        if flights:
            rows, prev_sta, prev_ata, prev_arr_delay = [], None, None, 0
            sorted_f = sorted(flights, key=lambda x: x.get('scheduled_out'))
            
            for f in sorted_f:
                std = pd.to_datetime(f.get('scheduled_out'))
                if std.date() != search_date: continue
                sta, atd, ata = pd.to_datetime(f.get('scheduled_in')), pd.to_datetime(f.get('actual_out')), pd.to_datetime(f.get('actual_in'))
                off, on = pd.to_datetime(f.get('actual_off')), pd.to_datetime(f.get('actual_on'))
                atd_val = atd if atd else (off - timedelta(minutes=10) if off else None)
                ata_val = ata if ata else (on + timedelta(minutes=5) if on else None)
                dep_delay = int((atd_val - std).total_seconds() / 60) if atd_val and std else 0
                arr_delay = int((ata_val - sta).total_seconds() / 60) if ata_val and sta else 0
                turn_s = int((std - prev_sta).total_seconds() / 60) if prev_sta else None
                turn_a = int((atd_val - prev_ata).total_seconds() / 60) if prev_ata and atd_val else None
                sched_block = int((sta - std).total_seconds() / 60) if sta and std else 0
                act_block = int((ata_val - atd_val).total_seconds() / 60) if ata_val and atd_val else 0

                rows.append({
                    'Flight': f.get('ident'), 'Origin': f.get('origin', {}).get('code_iata'), 'Dest': f.get('destination', {}).get('code_iata'),
                    'STD': format_time(std), 'ATD': format_time(atd_val), 'Dep_Delay': dep_delay, 'STA': format_time(sta), 'ATA': format_time(ata_val), 'Arr_Delay': arr_delay, 
                    'Sched_Block': sched_block, 'Act_Block': act_block, 'Block_Buffer': sched_block - act_block,
                    'Sched_Turn': turn_s, 'Act_Turn': turn_a, 'p_STD': std, 'p_STA': sta, 'p_ATD': atd_val, 'p_ATA': ata_val, 'Node': f.get('ident'), 'Inbound_Arr_Delay': prev_arr_delay
                })
                prev_sta, prev_ata, prev_arr_delay = sta, ata_val, arr_delay
            
            df_res = pd.DataFrame(rows)

            # --- 1. DATA LOG ---
            col_h1, col_i1 = st.columns([0.92, 0.08])
            with col_h1: st.subheader("📋 Flight Rotation Detailed Log")
            with col_i1:
                with st.popover("ℹ️"): st.write("View all flight legs for the tail. Compares scheduled vs actual minutes.")
            st.dataframe(df_res.drop(columns=['p_STD', 'p_STA', 'p_ATD', 'p_ATA', 'Inbound_Arr_Delay']), use_container_width=True)

            # --- 2. DELAY EROSION ---
            col_h2, col_i2 = st.columns([0.92, 0.08])
            with col_h2: st.subheader("📉 Delay Propagation Variance")
            with col_i2:
                with st.popover("ℹ️"): st.write("Visualizes delay recovery across legs. Red line shows outbound delay vs inbound (dashed).")
            fig_erosion = go.Figure()
            fig_erosion.add_trace(go.Scatter(x=df_res['Node'], y=df_res['Inbound_Arr_Delay'], mode='lines+markers', name='Inbound Delay', line=dict(color='#94a3b8', dash='dash')))
            fig_erosion.add_trace(go.Scatter(x=df_res['Node'], y=df_res['Dep_Delay'], mode='lines+markers+text', name='Outbound Delay', line=dict(color='#ef4444', width=3), text=df_res['Dep_Delay'], textposition="top center"))
            st.plotly_chart(fig_erosion.update_layout(height=250, margin=dict(t=10, b=10)), use_container_width=True)

            # --- 3. TIMELINE VARIANCE (GANTT) ---
            col_h3, col_i3 = st.columns([0.92, 0.08])
            with col_h3: st.subheader("🕒 Execution vs. Schedule Timeline Variance")
            with col_i3:
                with st.popover("ℹ️"): st.write("Gantt view: Grey = planned, colored = actual. Red indicates delays >15m.")
            fig_g = go.Figure()
            f_map = {f: i for i, f in enumerate(df_res['Flight'].unique())}
            for _, r in df_res.iterrows():
                y_idx = f_map[r['Flight']]
                fig_g.add_trace(go.Scatter(x=[r['p_STD'], r['p_STA']], y=[y_idx + 0.15, y_idx + 0.15], mode='lines', line=dict(color='#CBD5E1', width=6), name='Planned', showlegend=False))
                color = '#EF4444' if r['Arr_Delay'] > 15 else '#22C55E'
                fig_g.add_trace(go.Scatter(x=[r['p_ATD'], r['p_ATA']], y=[y_idx - 0.15, y_idx - 0.15], mode='lines+markers', line=dict(color=color, width=6), marker=dict(symbol='triangle-right', size=8), name='Actual', showlegend=False))
            fig_g.update_layout(height=350, yaxis=dict(tickvals=list(range(len(f_map))), ticktext=list(f_map.keys())))
            st.plotly_chart(fig_g, use_container_width=True)

            # --- 4. GROUND & BLOCK VARIANCE ---
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                sub1, info1 = st.columns([0.85, 0.15])
                with sub1: st.subheader("⌛ Ground Stop Efficiency Variance")
                with info1:
                    with st.popover("ℹ️"): st.write("Compare planned vs actual ground time. Blue higher than grey = overshoot.")
                st.plotly_chart(go.Figure([go.Bar(x=df_res['Flight'], y=df_res['Sched_Turn'], name='Plan', marker_color='lightgrey'), go.Bar(x=df_res['Flight'], y=df_res['Act_Turn'], name='Actual', marker_color='#3498db')]).update_layout(barmode='group', height=250), use_container_width=True)
            with c2:
                sub2, info2 = st.columns([0.85, 0.15])
                with sub2: st.subheader("🚀 Airborne Performance Buffer Variance")
                with info2:
                    with st.popover("ℹ️"): st.write("Compare planned vs actual flight time. Green higher than grey = slower than plan.")
                st.plotly_chart(go.Figure([go.Bar(x=df_res['Flight'], y=df_res['Sched_Block'], name='Plan', marker_color='lightgrey'), go.Bar(x=df_res['Flight'], y=df_res['Act_Block'], name='Actual', marker_color='#2ecc71')]).update_layout(barmode='group', height=250), use_container_width=True)

            # --- 5. SEASONAL ANALYSIS ---
            st.divider()
            col_h5, col_i5 = st.columns([0.92, 0.08])
            with col_h5: st.subheader("🎯 Seasonal Pattern Bottleneck Variance (S25)")
            with col_i5:
                with st.popover("ℹ️"): st.write("Analyzes the rotation UP TO Eindhoven arrival. Identifies bottlenecks in the sequence leading to EIN.")
            
            if st.button("🚀 Run Seasonal Pattern Search", use_container_width=True):
                with st.spinner(f"Scanning season for {target_flight_num} up to EIN arrival..."):
                    history_rows = []
                    start_s25, end_s25 = datetime(2025, 3, 30).date(), datetime(2025, 10, 31).date()
                    current_dt = start_s25
                    while current_dt <= end_s25:
                        if current_dt.weekday() == search_date.weekday():
                            day_instances = fetch_history_data_raw(fa_api_key, target_flight_num, current_dt, current_dt)
                            valid_f = next((f for f in day_instances if pd.to_datetime(f.get('scheduled_out')).date() == current_dt and f.get('destination', {}).get('code_iata') in ["EIN", "EHEH"]), None)
                            if valid_f:
                                rotation = fetch_history_data_raw(fa_api_key, valid_f.get('registration'), current_dt, current_dt)
                                if rotation:
                                    d_rows, prev_s, prev_a = [], None, None
                                    for h in sorted(rotation, key=lambda x: x.get('scheduled_out')):
                                        h_std, h_sta = pd.to_datetime(h.get('scheduled_out')), pd.to_datetime(h.get('scheduled_in'))
                                        h_atd, h_ata = pd.to_datetime(h.get('actual_out')), pd.to_datetime(h.get('actual_in'))
                                        
                                        # Only process legs UP TO and INCLUDING the Eindhoven arrival
                                        if h_std.date() != current_dt: continue
                                        
                                        s_t, a_t = int((h_std - prev_s).total_seconds()/60) if prev_s else 0, int((h_atd - prev_a).total_seconds()/60) if prev_a and h_atd else 0
                                        s_b, a_b = int((h_sta - h_std).total_seconds()/60) if h_sta and h_std else 0, int((h_ata - h_atd).total_seconds()/60) if h_ata and h_atd else 0
                                        
                                        d_rows.append({'Origin': h.get('origin', {}).get('code_iata'), 'Flight': h.get('ident'), 'S_T': s_t, 'A_T': a_t, 'T_Var': a_t - s_t, 'S_B': s_b, 'A_B': a_b, 'F_Var': a_b - s_b})
                                        
                                        prev_s, prev_a = h_sta, h_ata
                                        
                                        # STOP logic: If this was the flight to EIN, stop analyzing the rest of the day
                                        if h.get('destination', {}).get('code_iata') in ["EIN", "EHEH"]:
                                            break
                                            
                                    if d_rows:
                                        df_h = pd.DataFrame(d_rows)
                                        w_t, w_f = df_h.loc[df_h['T_Var'].idxmax()], df_h.loc[df_h['F_Var'].idxmax()]
                                        history_rows.append({
                                            'Date': current_dt.strftime('%d-%b'), 'Reg': valid_f.get('registration'),
                                            'Total Sched Turn': f"{df_h['S_T'].sum()}m", 'Total Act Turn': f"{df_h['A_T'].sum()}m",
                                            'Max Turn Loss (Airport)': f"{w_t['Origin']} (+{w_t['T_Var']}m)",
                                            'Total Sched Block': f"{df_h['S_B'].sum()}m", 'Total Act Block': f"{df_h['A_B'].sum()}m",
                                            'Max Flight Loss (Leg)': f"{w_f['Flight']} (+{w_f['F_Var']}m)"
                                        })
                        current_dt += timedelta(days=1)
                    if history_rows: st.table(pd.DataFrame(history_rows))

            # --- 6. SMART AI INSIGHTS ---
            st.divider()
            col_h6, col_i6 = st.columns([0.92, 0.08])
            with col_h6: st.subheader("🤖 Smart Operational AI Insights")
            with col_i6:
                with st.popover("ℹ️"): st.write("AI analysis of network recovery and EIN slot protection.")
            
            if google_api_key:
                try:
                    ai_client = genai.Client(api_key=google_api_key)
                    ai_summary = df_res[['Flight', 'Origin', 'Dest', 'Dep_Delay', 'Arr_Delay', 'Sched_Turn', 'Act_Turn']].to_string(index=False)
                    ai_response = ai_client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=f"Analyze this aircraft rotation for flight pattern {target_flight_num}. Identify risks for Eindhoven slots. Data: {ai_summary}"
                    )
                    st.markdown(f"""<div class="ai-box">{ai_response.text}</div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"AI Quota reached or Error: {str(e)}")
            else:
                st.warning("Please provide a Google Gemini API Key.")

st.sidebar.info("S25 Analysis: Mar 30 - Oct 31")