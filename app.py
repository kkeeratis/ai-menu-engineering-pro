import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import google.generativeai as genai
import time

# ==============================
# 1. CONFIG & STYLING
# ==============================
st.set_page_config(
    page_title="AI Restaurant Strategy Pro (Menu + RFM)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_custom_css():
    st.markdown("""
        <style>
        .main { background-color: #f8f9fa; }
        h1, h2, h3 { color: #1e3a8a; font-family: 'Inter', sans-serif; }
        .stButton>button {
            background-color: #1e40af; color: white; border-radius: 8px; font-weight: bold; width: 100%; transition: 0.3s;
        }
        .stButton>button:hover { background-color: #1e3a8a; transform: translateY(-2px); }
        .strategy-card {
            background: white; padding: 25px; border-radius: 12px; border-left: 6px solid #10b981;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==============================
# 2. CORE LOGIC: MENU ENGINEERING
# ==============================
@st.cache_data(show_spinner=False)
def calculate_menu_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    df['Profit Margin (THB)'] = df['Price'] - df['Cost']
    df['Total Profit'] = df['Profit Margin (THB)'] * df['Sold Qty']
    
    # ปรับสมการให้เสถียรขึ้น ป้องกัน ValueError เมื่อตารางว่างหรือค่าเป็น 0
    total_items_sold = df['Sold Qty'].sum()
    if total_items_sold > 0:
        df['Mix %'] = (df['Sold Qty'] / total_items_sold) * 100
    else:
        df['Mix %'] = 0.0
    
    avg_margin = df['Total Profit'].sum() / total_items_sold if total_items_sold > 0 else 0
    avg_mix = (1 / len(df)) * 100 * 0.7 if len(df) > 0 else 0
    
    conditions = [
        (df['Profit Margin (THB)'] >= avg_margin) & (df['Mix %'] >= avg_mix),
        (df['Profit Margin (THB)'] < avg_margin) & (df['Mix %'] >= avg_mix),
        (df['Profit Margin (THB)'] >= avg_margin) & (df['Mix %'] < avg_mix),
        (df['Profit Margin (THB)'] < avg_margin) & (df['Mix %'] < avg_mix)
    ]
    choices = ['🌟 Star (ดาวเด่น)', '🐴 Plowhorse (ม้างาน)', '🧩 Puzzle (ปริศนา)', '🐶 Dog (หมาน้อย)']
    df['Category'] = np.select(conditions, choices, default='Unknown')
    return df, avg_margin, avg_mix

# ==============================
# 3. CORE LOGIC: RFM ANALYSIS
# ==============================
@st.cache_data(show_spinner=False)
def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """คำนวณและจัดกลุ่มลูกค้าตามหลักการ RFM"""
    # กำหนดเกณฑ์คะแนน (1-5) อย่างง่ายสำหรับ Demo
    df['R_Score'] = np.select(
        [df['Recency (Days)'] <= 7, df['Recency (Days)'] <= 15, df['Recency (Days)'] <= 30, df['Recency (Days)'] <= 60], 
        [5, 4, 3, 2], default=1
    )
    df['F_Score'] = np.select(
        [df['Frequency (Visits)'] >= 10, df['Frequency (Visits)'] >= 7, df['Frequency (Visits)'] >= 4, df['Frequency (Visits)'] >= 2], 
        [5, 4, 3, 2], default=1
    )
    
    # จัดกลุ่ม Segments
    conditions = [
        (df['R_Score'] >= 4) & (df['F_Score'] >= 4), # มาบ่อย มาล่าสุด
        (df['R_Score'] >= 3) & (df['F_Score'] >= 3), # ขาประจำ
        (df['R_Score'] <= 2) & (df['F_Score'] >= 3), # เคยมาบ่อย แต่หายไปนาน
        (df['R_Score'] >= 4) & (df['F_Score'] <= 2), # เพิ่งมาครั้งแรกๆ
        (df['R_Score'] <= 2) & (df['F_Score'] <= 2)  # ขาจรที่หายไปนาน
    ]
    choices = ['👑 Champions', '🤝 Loyal Customers', '⚠️ At Risk (กำลังจะหายไป)', '👋 New/Promising', '💤 Hibernating']
    df['Segment'] = np.select(conditions, choices, default='Others')
    return df

# ==============================
# 4. AI HOLISTIC STRATEGY (MENU + RFM)
# ==============================
def generate_holistic_strategy(api_key: str, menu_df: pd.DataFrame, rfm_df: pd.DataFrame) -> str:
    """ส่งข้อมูลทั้ง 2 ด้านให้ AI คิดกลยุทธ์เชื่อมโยง (Cross-Analysis)"""
    if not api_key: return "⚠️ กรุณาใส่ Gemini API Key ที่แถบด้านข้างก่อนครับ"
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # สรุปฝั่งเมนู
        menu_grouped = menu_df.groupby('Category')['Menu Item'].apply(list).to_dict()
        
        # สรุปฝั่งลูกค้า
        rfm_summary = rfm_df['Segment'].value_counts().to_dict()
        total_customers = len(rfm_df)
        
        prompt = f"""
        คุณคือ 'Chief Strategy Officer (CSO)' ระดับองค์กรธุรกิจ F&B 
        เรามีข้อมูล Data Analytics 2 แกนหลักของร้านอาหาร ดังนี้:
        
        📊 1. Product Analytics (Menu Engineering BCG Matrix):
        - 🌟 ดาวเด่น (กำไรสูง ขายดี): {', '.join(menu_grouped.get('🌟 Star (ดาวเด่น)', ['ไม่มี']))}
        - 🐴 ม้างาน (กำไรต่ำ ขายดี): {', '.join(menu_grouped.get('🐴 Plowhorse (ม้างาน)', ['ไม่มี']))}
        - 🧩 ปริศนา (กำไรสูง ขายไม่ดี): {', '.join(menu_grouped.get('🧩 Puzzle (ปริศนา)', ['ไม่มี']))}
        - 🐶 หมาน้อย (กำไรต่ำ ขายไม่ดี): {', '.join(menu_grouped.get('🐶 Dog (หมาน้อย)', ['ไม่มี']))}
        
        👥 2. Customer Analytics (RFM Segmentation จาก {total_customers} ตัวอย่างลูกค้า):
        - 👑 Champions (ลูกค้ารักเราสุดๆ): {rfm_summary.get('👑 Champions', 0)} คน
        - ⚠️ At Risk (ลูกค้าประจำที่เริ่มหายไป): {rfm_summary.get('⚠️ At Risk (กำลังจะหายไป)', 0)} คน
        - 👋 New/Promising (ลูกค้าหน้าใหม่): {rfm_summary.get('👋 New/Promising', 0)} คน
        
        📌 คำสั่ง: 
        ให้ออกแบบกลยุทธ์ "การตลาดและการขาย (Sales & Marketing Tactics)" แบบเจาะลึก 3 ข้อ 
        โดยให้ "จับคู่กลุ่มเมนู (Product)" เข้ากับ "กลุ่มลูกค้า (Customer Segment)" เพื่อแก้ปัญหาหรือเพิ่มยอดขาย 
        ตัวอย่างเช่น: "ดึงลูกค้า At Risk กลับมาด้วยโปรโมชันเมนู Plowhorse" หรือ "ทำ Upsell เมนู Puzzle ให้กลุ่ม Champions"
        
        ตอบเป็นภาษาไทย รูปแบบมืออาชีพสำหรับพรีเซนต์ผู้บริหาร (Actionable Insights) ห้ามใช้แท็ก HTML
        """
        response = model.generate_content(prompt)
        return f"*(Powered by: `gemini-2.5-flash`)*\n\n" + response.text
    except Exception as e:
        return f"❌ AI Error: ระบบเชื่อมต่อขัดข้อง ({str(e)})"

# ==============================
# 5. UI RENDERING FUNCTIONS
# ==============================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>🧠</h1>", unsafe_allow_html=True)
        st.header("⚙️ ตั้งค่าระบบ (Settings)")
        api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else st.text_input("Gemini API Key:", type="password")
        st.caption("รับ Key ฟรีได้ที่ Google AI Studio")
        st.divider()
        st.markdown("### 📊 Module 1: Menu Engineering")
        st.caption("วิเคราะห์ความคุ้มค่าของเมนู (Product-Centric)")
        st.markdown("### 👥 Module 2: RFM Analysis")
        st.caption("วิเคราะห์พฤติกรรมลูกค้า (Customer-Centric)")
        return api_key

def init_mock_data():
    if 'menu_data' not in st.session_state:
        st.session_state.menu_data = pd.DataFrame({
            'Menu Item': ['อเมริกาโน่เย็น', 'ลาเต้เย็น', 'คาราเมลมัคคิอาโต้', 'ครัวซองต์เนยสด', 'เค้กช็อกโกแลต', 'มัทฉะพรีเมียม', 'แซนด์วิชแฮมชีส', 'น้ำเปล่า'],
            'Cost': [15, 25, 30, 30, 45, 40, 40, 5],
            'Price': [60, 75, 95, 80, 120, 110, 85, 15],
            'Sold Qty': [500, 450, 150, 300, 80, 120, 100, 200]
        })
    if 'rfm_data' not in st.session_state:
        # Mock ข้อมูลลูกค้า 15 คน
        st.session_state.rfm_data = pd.DataFrame({
            'Customer ID': [f"C{str(i).zfill(3)}" for i in range(1, 16)],
            'Recency (Days)': [2, 15, 45, 2, 5, 60, 10, 30, 90, 4, 1, 35, 8, 12, 50],
            'Frequency (Visits)': [12, 5, 2, 15, 8, 1, 6, 3, 1, 10, 20, 4, 3, 7, 2],
            'Monetary (THB)': [4500, 1200, 300, 6000, 2500, 150, 1800, 900, 200, 3800, 8000, 1500, 750, 2100, 400]
        })

# ==============================
# 6. MAIN APP EXECUTION
# ==============================
def main():
    apply_custom_css()
    
    st.markdown("<h1 style='text-align: center;'>🧠 AI Restaurant Strategy Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Holistic Analytics: <b>Product (Menu Engineering)</b> x <b>Customer (RFM)</b></p>", unsafe_allow_html=True)
    st.divider()

    api_key_input = render_sidebar()
    init_mock_data()

    # ใช้ Tabs ในการจัดระเบียบหน้าจอ
    tab1, tab2, tab3 = st.tabs(["🍽️ 1. Menu Engineering (เมนู)", "👥 2. RFM Analysis (ลูกค้า)", "🚀 3. AI Holistic Strategy (เชื่อมโยงกลยุทธ์)"])

    # ---------------- TAB 1: MENU ----------------
    with tab1:
        st.subheader("วิเคราะห์โครงสร้างเมนูและกำไร (Product Centric)")
        edited_menu_df = st.data_editor(st.session_state.menu_data, num_rows="dynamic", use_container_width=True)
        
        if st.button("📈 คำนวณ BCG Matrix", key="btn_menu"):
            with st.spinner("กำลังประมวลผล..."):
                result_df, avg_margin, avg_mix = calculate_menu_engineering(edited_menu_df.copy())
                st.session_state['analyzed_menu'] = result_df
                
        if 'analyzed_menu' in st.session_state:
            df = st.session_state['analyzed_menu']
            st.markdown("#### 📊 BCG Menu Matrix")
            
            base_chart = alt.Chart(df).encode(
                x=alt.X('Mix %:Q', title='ความนิยม (%)'),
                y=alt.Y('Profit Margin (THB):Q', title='กำไรต่อจาน (฿)')
            )
            points = base_chart.mark_circle(size=200, opacity=0.8).encode(
                color=alt.Color('Category:N', scale=alt.Scale(
                    domain=['🌟 Star (ดาวเด่น)', '🐴 Plowhorse (ม้างาน)', '🧩 Puzzle (ปริศนา)', '🐶 Dog (หมาน้อย)'],
                    range=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
                )), tooltip=['Menu Item', 'Category']
            )
            text_labels = base_chart.mark_text(align='left', dx=10, fontSize=11, fontWeight='bold').encode(text='Menu Item:N')
            
            st.altair_chart((points + text_labels).interactive(), use_container_width=True)

    # ---------------- TAB 2: RFM ----------------
    with tab2:
        st.subheader("วิเคราะห์พฤติกรรมลูกค้า (Customer Centric)")
        st.caption("R = วันล่าสุดที่มาซื้อ, F = ความถี่ที่มาซื้อ, M = ยอดใช้จ่ายรวม")
        edited_rfm_df = st.data_editor(st.session_state.rfm_data, num_rows="dynamic", use_container_width=True)
        
        if st.button("🔍 แบ่งกลุ่มลูกค้า (Segment Customers)", key="btn_rfm"):
            with st.spinner("กำลังประมวลผล..."):
                result_rfm = calculate_rfm(edited_rfm_df.copy())
                st.session_state['analyzed_rfm'] = result_rfm
                
        if 'analyzed_rfm' in st.session_state:
            rfm_df = st.session_state['analyzed_rfm']
            st.markdown("#### 👥 RFM Customer Segments")
            
            # กราฟลูกค้า R vs F ขนาดวงกลมตาม M
            rfm_chart = alt.Chart(rfm_df).mark_circle(opacity=0.8).encode(
                x=alt.X('Recency (Days):Q', title='วันล่าสุดที่ซื้อ (น้อย = ดี)', sort='descending'),
                y=alt.Y('Frequency (Visits):Q', title='ความถี่ในการซื้อ (มาก = ดี)'),
                size=alt.Size('Monetary (THB):Q', title='ยอดใช้จ่ายรวม'),
                color=alt.Color('Segment:N', scale=alt.Scale(scheme='set2')),
                tooltip=['Customer ID', 'Segment', 'Recency (Days)', 'Frequency (Visits)', 'Monetary (THB)']
            ).properties(height=400)
            
            st.altair_chart(rfm_chart.interactive(), use_container_width=True)
            
            # สรุปจำนวนลูกค้า (แก้บั๊ก DataFrame Duplicate Columns เรียบร้อย)
            st.markdown("##### 📌 สรุปจำนวนลูกค้าแต่ละกลุ่ม")
            summary_df = rfm_df['Segment'].value_counts().reset_index()
            summary_df.columns = ['Segment', 'จำนวนคน']
            st.dataframe(summary_df, hide_index=True)

    # ---------------- TAB 3: AI HOLISTIC ----------------
    with tab3:
        st.subheader("🧠 กลยุทธ์ผสานข้อมูล (Menu x RFM)")
        st.caption("กดปุ่มด้านล่างเพื่อให้ AI นำข้อมูล 'ความคุ้มค่าของเมนู' มาจับคู่กับ 'พฤติกรรมลูกค้า' เพื่อสร้างแผนการตลาดที่ตรงจุด")
        
        if 'analyzed_menu' not in st.session_state or 'analyzed_rfm' not in st.session_state:
            st.warning("⚠️ กรุณากดคำนวณข้อมูลใน Tab 1 และ Tab 2 ให้ครบก่อนครับ")
        else:
            if st.button("✨ สร้างแผนกลยุทธ์แบบองค์รวม (Generate Holistic Strategy)", key="btn_ai"):
                with st.spinner("AI กำลังวิเคราะห์และจับคู่ข้อมูล Product กับ Customer..."):
                    strategy_text = generate_holistic_strategy(api_key_input, st.session_state['analyzed_menu'], st.session_state['analyzed_rfm'])
                    st.markdown(f'<div class="strategy-card">{strategy_text}</div>', unsafe_allow_html=True)

    st.markdown("<br><hr><center><small>Built with Streamlit, Pandas, Altair & Gemini AI | Multi-Dimensional Analytics Portfolio</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
