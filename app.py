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
    page_title="AI Menu Engineering Pro",
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
# 2. CORE BUSINESS LOGIC (Optimized)
# ==============================
@st.cache_data(show_spinner=False)
def calculate_menu_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """คำนวณ Menu Engineering ด้วย NumPy Vectorization (ประสิทธิภาพสูง)"""
    # 1. คำนวณกำไร และ กำไรรวม
    df['Profit Margin (THB)'] = df['Price'] - df['Cost']
    df['Total Profit'] = df['Profit Margin (THB)'] * df['Sold Qty']
    
    # 2. คำนวณสัดส่วนยอดขาย
    total_items_sold = df['Sold Qty'].sum()
    df['Mix %'] = np.where(total_items_sold > 0, (df['Sold Qty'] / total_items_sold) * 100, 0)
    
    # 3. คำนวณค่าเฉลี่ย (Benchmarks)
    avg_margin = df['Total Profit'].sum() / total_items_sold if total_items_sold > 0 else 0
    avg_mix = (1 / len(df)) * 100 * 0.7 if len(df) > 0 else 0
    
    # 4. จัดกลุ่มเมนู (Vectorized Conditions - เร็วกว่า df.apply หลายเท่า)
    conditions = [
        (df['Profit Margin (THB)'] >= avg_margin) & (df['Mix %'] >= avg_mix),
        (df['Profit Margin (THB)'] < avg_margin) & (df['Mix %'] >= avg_mix),
        (df['Profit Margin (THB)'] >= avg_margin) & (df['Mix %'] < avg_mix),
        (df['Profit Margin (THB)'] < avg_margin) & (df['Mix %'] < avg_mix)
    ]
    choices = ['🌟 Star (ดาวเด่น)', '🐴 Plowhorse (ม้างาน)', '🧩 Puzzle (ปริศนา)', '🐶 Dog (หมาน้อย)']
    
    df['Category'] = np.select(conditions, choices, default='Unknown')
    
    return df, avg_margin, avg_mix

def generate_ai_strategy(api_key: str, df: pd.DataFrame) -> str:
    """เรียกใช้ Gemini AI เพื่อวางแผนกลยุทธ์จาก Dataframe"""
    if not api_key:
        return "⚠️ กรุณาใส่ Gemini API Key ที่แถบด้านข้างก่อนครับ"
        
    try:
        genai.configure(api_key=api_key)
        # ใช้ 2.5-flash เป็นโมเดลหลัก
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        # จัดกลุ่มข้อมูลส่งให้ AI
        grouped = df.groupby('Category')['Menu Item'].apply(list).to_dict()
        
        prompt = f"""
        ในฐานะ F&B Consultant ระดับองค์กร นี่คือโครงสร้างเมนู (Menu Engineering) ของร้านอาหาร:
        
        - 🌟 ดาวเด่น (กำไรสูง ขายดี): {', '.join(grouped.get('🌟 Star (ดาวเด่น)', ['ไม่มี']))}
        - 🐴 ม้างาน (กำไรต่ำ ขายดี): {', '.join(grouped.get('🐴 Plowhorse (ม้างาน)', ['ไม่มี']))}
        - 🧩 ปริศนา (กำไรสูง ขายไม่ดี): {', '.join(grouped.get('🧩 Puzzle (ปริศนา)', ['ไม่มี']))}
        - 🐶 หมาน้อย (กำไรต่ำ ขายไม่ดี): {', '.join(grouped.get('🐶 Dog (หมาน้อย)', ['ไม่มี']))}
        
        กรุณาวิเคราะห์และแนะนำ Actionable Strategy สำหรับเมนูแต่ละกลุ่มเพื่อเพิ่ม 'กำไรรวม' 
        ตอบเป็นภาษาไทย รูปแบบมืออาชีพสำหรับผู้บริหาร ห้ามใช้แท็ก HTML
        """
        response = model.generate_content(prompt)
        return f"*(Powered by: `gemini-2.5-flash`)*\n\n" + response.text
    except Exception as e:
        return f"❌ AI Error: ระบบเชื่อมต่อขัดข้อง ({str(e)})"

# ==============================
# 3. UI RENDERING FUNCTIONS
# ==============================
def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>🍽️</h1>", unsafe_allow_html=True)
        st.header("⚙️ ตั้งค่าระบบ (Settings)")
        api_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else st.text_input("Gemini API Key:", type="password")
        st.caption("รับ Key ฟรีได้ที่ Google AI Studio")
        st.divider()
        st.markdown("""
        **โครงสร้าง 4 กลุ่มเมนู:**
        - 🌟 **Stars:** กำไรสูง ขายน้อย (โปรโมทเพิ่ม)
        - 🐴 **Plowhorses:** กำไรน้อย ขายดี (ขึ้นราคาเนียนๆ/ลดต้นทุน)
        - 🧩 **Puzzles:** กำไรสูง ขายไม่ดี (ปรับหน้าตา/ทำโปร)
        - 🐶 **Dogs:** กำไรน้อย ขายไม่ดี (พิจารณาตัดทิ้ง)
        """)
        return api_key

def init_mock_data():
    if 'menu_data' not in st.session_state:
        st.session_state.menu_data = pd.DataFrame({
            'Menu Item': ['อเมริกาโน่เย็น', 'ลาเต้เย็น', 'คาราเมลมัคคิอาโต้', 'ครัวซองต์เนยสด', 'เค้กช็อกโกแลต', 'มัทฉะพรีเมียม', 'แซนด์วิชแฮมชีส', 'น้ำเปล่า'],
            'Cost': [15, 25, 30, 30, 45, 40, 40, 5],
            'Price': [60, 75, 95, 80, 120, 110, 85, 15],
            'Sold Qty': [500, 450, 150, 300, 80, 120, 100, 200]
        })

def render_dashboard(df: pd.DataFrame, avg_margin: float, avg_mix: float):
    st.divider()
    st.subheader("2. 🎯 ผลการวิเคราะห์ BCG Matrix (Menu Categorization)")
    
    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ยอดขายรวม (Total Items)", f"{df['Sold Qty'].sum():,} ชิ้น")
    m2.metric("กำไรรวม (Total Profit)", f"฿ {df['Total Profit'].sum():,.2f}")
    m3.metric("ค่าเฉลี่ยกำไรต่อจาน", f"฿ {avg_margin:.2f}")
    m4.metric("เกณฑ์ความนิยมเฉลี่ย", f"{avg_mix:.2f} %")
    
    st.write("")
    c1, c2 = st.columns([1.6, 1])
    
    with c1:
        # --- Optimized Altair Chart (Added Text Labels) ---
        base_chart = alt.Chart(df).encode(
            x=alt.X('Mix %:Q', title='ความนิยม (Popularity / Mix %)', scale=alt.Scale(zero=False)),
            y=alt.Y('Profit Margin (THB):Q', title='กำไรต่อจาน (Profitability)', scale=alt.Scale(zero=False))
        )
        
        # จุด Scatter Plot
        points = base_chart.mark_circle(size=200, opacity=0.8).encode(
            color=alt.Color('Category:N', scale=alt.Scale(
                domain=['🌟 Star (ดาวเด่น)', '🐴 Plowhorse (ม้างาน)', '🧩 Puzzle (ปริศนา)', '🐶 Dog (หมาน้อย)'],
                range=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            ), legend=alt.Legend(title="กลุ่มเมนู", orient='bottom')),
            tooltip=['Menu Item', 'Profit Margin (THB)', 'Mix %', 'Sold Qty', 'Category']
        )
        
        # ใส่ชื่อเมนูข้างๆ จุด (ทำให้ดูง่ายขึ้นมากเวลาพรีเซนต์)
        text_labels = base_chart.mark_text(
            align='left', baseline='middle', dx=10, fontSize=11, fontWeight='bold', color='#333333'
        ).encode(text='Menu Item:N')
        
        # เส้นตัดแกนค่าเฉลี่ย
        x_rule = alt.Chart(pd.DataFrame({'x': [avg_mix]})).mark_rule(color='#475569', strokeDash=[5,5]).encode(x='x:Q')
        y_rule = alt.Chart(pd.DataFrame({'y': [avg_margin]})).mark_rule(color='#475569', strokeDash=[5,5]).encode(y='y:Q')
        
        st.altair_chart((points + text_labels + x_rule + y_rule).interactive(), use_container_width=True)

    with c2:
        st.markdown("##### 📌 สรุปเมนูตามกลุ่ม")
        display_df = df[['Menu Item', 'Profit Margin (THB)', 'Mix %', 'Category']].sort_values(by='Profit Margin (THB)', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ==============================
# 4. MAIN APP EXECUTION
# ==============================
def main():
    apply_custom_css()
    
    st.markdown("<h1 style='text-align: center;'>📊 AI Menu Engineering & Profit Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>ระบบวิเคราะห์และเพิ่มผลกำไรโครงสร้างเมนูร้านอาหารระดับองค์กร</p>", unsafe_allow_html=True)
    st.divider()

    api_key_input = render_sidebar()
    init_mock_data()

    # --- Section 1: Data Input ---
    st.subheader("1. 📝 ป้อนข้อมูลเมนูอาหาร (Data Input)")
    st.caption("ทดลองแก้ไขตัวเลขในตารางด้านล่าง ระบบจะคำนวณกำไรและสัดส่วนให้แบบ Real-time")

    edited_df = st.data_editor(
        st.session_state.menu_data, 
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Menu Item": st.column_config.TextColumn("ชื่อเมนู (Menu Item)", required=True),
            "Cost": st.column_config.NumberColumn("ต้นทุน (Cost)", min_value=0, format="%d ฿"),
            "Price": st.column_config.NumberColumn("ราคาขาย (Price)", min_value=0, format="%d ฿"),
            "Sold Qty": st.column_config.NumberColumn("ยอดขายต่อเดือน (Qty)", min_value=0)
        }
    )

    if st.button("📈 1. วิเคราะห์โครงสร้างเมนู (Analyze Matrix)"):
        with st.spinner("กำลังคำนวณตัวเลขทางบัญชีและจัดกลุ่ม BCG Matrix..."):
            result_df, avg_margin, avg_mix = calculate_menu_engineering(edited_df.copy())
            st.session_state['analyzed_df'] = result_df
            st.session_state['avg_margin'] = avg_margin
            st.session_state['avg_mix'] = avg_mix
            time.sleep(0.3)

    # --- Section 2: Dashboard ---
    if 'analyzed_df' in st.session_state:
        df = st.session_state['analyzed_df']
        render_dashboard(df, st.session_state['avg_margin'], st.session_state['avg_mix'])
            
        # --- Section 3: AI Strategy ---
        st.divider()
        st.subheader("3. 🧠 ที่ปรึกษากลยุทธ์ AI (AI Strategy Consultant)")
        st.caption("ส่งข้อมูลเมนูที่ถูกจัดกลุ่มเสร็จแล้วให้ AI วางแผนการตลาดและปรับปรุงต้นทุนเพื่อเพิ่มกำไรสูงสุด")
        
        if st.button("✨ 2. สร้างแผนกลยุทธ์ระดับผู้บริหาร (Generate AI Strategy)"):
            with st.spinner("AI กำลังวิเคราะห์และเรียบเรียงกลยุทธ์..."):
                strategy_text = generate_ai_strategy(api_key_input, df)
                st.markdown(f'<div class="strategy-card">{strategy_text}</div>', unsafe_allow_html=True)
                
    st.markdown("<br><hr><center><small>Built with Streamlit, Pandas, Altair & Gemini AI | Optimized Professional Portfolio</small></center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
