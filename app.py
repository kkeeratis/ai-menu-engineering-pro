import streamlit as st
import pandas as pd
import altair as alt
import google.generativeai as genai
import time

# ==============================
# CONFIG & SETUP
# ==============================
st.set_page_config(
    page_title="AI Menu Engineering Pro",
    page_icon="📊",
    layout="wide"
)

# --- Custom CSS เพื่อความสวยงามระดับ Enterprise ---
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
    .metric-box {
        background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# UTILITIES & MATH LOGIC
# ==============================
def calculate_menu_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """คำนวณกำไร สัดส่วนการขาย และจัดกลุ่มตามทฤษฎี Menu Engineering"""
    # 1. คำนวณกำไรต่อจาน (Contribution Margin)
    df['Profit Margin (THB)'] = df['Price'] - df['Cost']
    
    # 2. คำนวณกำไรรวมของแต่ละเมนู
    df['Total Profit'] = df['Profit Margin (THB)'] * df['Sold Qty']
    
    # 3. คำนวณสัดส่วนการขาย (Menu Mix %)
    total_items_sold = df['Sold Qty'].sum()
    df['Mix %'] = (df['Sold Qty'] / total_items_sold) * 100 if total_items_sold > 0 else 0
    
    # 4. หาค่าเฉลี่ยของร้าน (Benchmark)
    avg_margin = df['Total Profit'].sum() / total_items_sold if total_items_sold > 0 else 0
    avg_mix = (1 / len(df)) * 100 * 0.7 if len(df) > 0 else 0  # ทฤษฎีมักใช้ 70% ของค่าเฉลี่ยปกติเป็นเกณฑ์
    
    # 5. จัดกลุ่มเมนู (Categorization)
    def categorize(row):
        high_margin = row['Profit Margin (THB)'] >= avg_margin
        high_popularity = row['Mix %'] >= avg_mix
        
        if high_margin and high_popularity:
            return '🌟 Star (ดาวเด่น)'
        elif not high_margin and high_popularity:
            return '🐴 Plowhorse (ม้างาน)'
        elif high_margin and not high_popularity:
            return '🧩 Puzzle (ปริศนา)'
        else:
            return '🐶 Dog (หมาน้อย)'
            
    df['Category'] = df.apply(categorize, axis=1)
    
    return df, avg_margin, avg_mix

def generate_ai_strategy(api_key: str, df: pd.DataFrame) -> str:
    """ส่งข้อมูลที่วิเคราะห์แล้วให้ AI วางแผนกลยุทธ์"""
    if not api_key:
        return "⚠️ กรุณาใส่ Gemini API Key ที่แถบด้านข้างก่อนครับ"
        
    try:
        genai.configure(api_key=api_key)
        
        # ค้นหา Model ที่รองรับ
        available_models = [m.name.replace('models/', '') for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        preferred = ['gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-1.5-pro']
        model_name = next((m for m in preferred if m in available_models), available_models[0] if available_models else "gemini-1.5-flash")
        
        model = genai.GenerativeModel(model_name)
        
        # สรุปข้อมูลเมนูแต่ละกลุ่มเพื่อส่งให้ AI
        stars = df[df['Category'] == '🌟 Star (ดาวเด่น)']['Menu Item'].tolist()
        plowhorses = df[df['Category'] == '🐴 Plowhorse (ม้างาน)']['Menu Item'].tolist()
        puzzles = df[df['Category'] == '🧩 Puzzle (ปริศนา)']['Menu Item'].tolist()
        dogs = df[df['Category'] == '🐶 Dog (หมาน้อย)']['Menu Item'].tolist()
        
        prompt = f"""
        ในฐานะผู้เชี่ยวชาญด้านกลยุทธ์ร้านอาหารและ F&B Consultant ระดับองค์กร
        นี่คือผลการวิเคราะห์โครงสร้างเมนู (Menu Engineering) ของร้านอาหารแห่งหนึ่ง:
        
        - 🌟 ดาวเด่น (กำไรสูง ขายดี): {', '.join(stars) if stars else 'ไม่มี'}
        - 🐴 ม้างาน (กำไรต่ำ แต่ขายดีมาก ดึงคนเข้าร้าน): {', '.join(plowhorses) if plowhorses else 'ไม่มี'}
        - 🧩 ปริศนา (กำไรสูงมาก แต่คนไม่ค่อยสั่ง): {', '.join(puzzles) if puzzles else 'ไม่มี'}
        - 🐶 หมาน้อย (กำไรต่ำ และคนไม่สั่ง): {', '.join(dogs) if dogs else 'ไม่มี'}
        
        คำสั่ง:
        กรุณาวิเคราะห์และให้คำแนะนำเชิงกลยุทธ์แบบเจาะจง (Actionable Strategy) สำหรับเมนูแต่ละกลุ่ม 
        เพื่อเพิ่ม 'กำไรรวม (Total Profit)' ให้กับร้านนี้ 
        ตอบเป็นภาษาไทย รูปแบบมืออาชีพแบบที่ผู้บริหารอ่าน และห้ามใช้แท็ก HTML
        """
        
        response = model.generate_content(prompt)
        return f"*(AI Model: `{model_name}`)*\n\n" + response.text
    except Exception as e:
        return f"❌ AI Error: ระบบ AI ขัดข้อง กรุณาตรวจสอบ API Key หรือการเชื่อมต่อ ({str(e)})"

# ==============================
# UI DESIGN
# ==============================
st.markdown("<h1 style='text-align: center;'>📊 AI Menu Engineering & Profit Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b;'>ระบบวิเคราะห์และเพิ่มผลกำไรโครงสร้างเมนูร้านอาหารระดับองค์กร</p>", unsafe_allow_html=True)
st.divider()

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>🍽️</h1>", unsafe_allow_html=True)
    st.header("⚙️ ตั้งค่าระบบ (Settings)")
    api_key_input = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else st.text_input("Gemini API Key:", type="password")
    st.caption("รับ Key ฟรีได้ที่ Google AI Studio")
    st.divider()
    st.markdown("""
    **คำแนะนำโครงสร้าง 4 กลุ่มเมนู:**
    - 🌟 **Stars:** กำไรสูง ขายน้อย (โปรโมทเพิ่ม)
    - 🐴 **Plowhorses:** กำไรน้อย ขายดี (ขึ้นราคาเนียนๆ/ลดต้นทุน)
    - 🧩 **Puzzles:** กำไรสูง ขายไม่ดี (ปรับหน้าตา/ทำโปร)
    - 🐶 **Dogs:** กำไรน้อย ขายไม่ดี (พิจารณาตัดทิ้ง)
    """)

# --- Main App ---
st.subheader("1. 📝 ป้อนข้อมูลเมนูอาหาร (Data Input)")
st.caption("ทดลองแก้ไขตัวเลขในตารางด้านล่าง ระบบจะคำนวณหากำไรและสัดส่วนให้แบบ Real-time")

# Mock Data เริ่มต้นเพื่อให้ User เห็นภาพการทำงานทันที
if 'menu_data' not in st.session_state:
    st.session_state.menu_data = pd.DataFrame({
        'Menu Item': ['อเมริกาโน่เย็น', 'ลาเต้เย็น', 'คาราเมลมัคคิอาโต้', 'ครัวซองต์เนยสด', 'เค้กช็อกโกแลต', 'มัทฉะพรีเมียม', 'แซนด์วิชแฮมชีส', 'น้ำเปล่า'],
        'Cost': [15, 25, 30, 30, 45, 40, 40, 5],
        'Price': [60, 75, 95, 80, 120, 110, 85, 15],
        'Sold Qty': [500, 450, 150, 300, 80, 120, 100, 200]
    })

# ให้ผู้ใช้แก้ไขตารางได้ (Data Editor)
edited_df = st.data_editor(
    st.session_state.menu_data, 
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Menu Item": st.column_config.TextColumn("ชื่อเมนู (Menu Item)"),
        "Cost": st.column_config.NumberColumn("ต้นทุน (Cost)", min_value=0, format="%d ฿"),
        "Price": st.column_config.NumberColumn("ราคาขาย (Price)", min_value=0, format="%d ฿"),
        "Sold Qty": st.column_config.NumberColumn("ยอดขายต่อเดือน (Qty)", min_value=0)
    }
)

if st.button("📈 1. วิเคราะห์โครงสร้างเมนู (Analyze Matrix)"):
    with st.spinner("กำลังคำนวณตัวเลขทางบัญชีและจัดกลุ่ม BCG Matrix..."):
        # คำนวณ Logic
        result_df, avg_margin, avg_mix = calculate_menu_engineering(edited_df)
        st.session_state['analyzed_df'] = result_df
        st.session_state['avg_margin'] = avg_margin
        st.session_state['avg_mix'] = avg_mix
        time.sleep(0.5) # Fake loading for UX

# หากมีการกดวิเคราะห์แล้ว ให้แสดงผลลัพธ์
if 'analyzed_df' in st.session_state:
    df = st.session_state['analyzed_df']
    avg_margin = st.session_state['avg_margin']
    avg_mix = st.session_state['avg_mix']
    
    st.divider()
    st.subheader("2. 🎯 ผลการวิเคราะห์ BCG Matrix (Menu Categorization)")
    
    # สรุปตัวเลข (Metrics)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ยอดขายรวม (Total Items)", f"{df['Sold Qty'].sum():,} ชิ้น")
    m2.metric("กำไรรวม (Total Profit)", f"฿ {df['Total Profit'].sum():,.2f}")
    m3.metric("ค่าเฉลี่ยกำไรต่อจาน", f"฿ {avg_margin:.2f}")
    m4.metric("เกณฑ์ความนิยมเฉลี่ย", f"{avg_mix:.2f} %")
    
    st.write("")
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        # สร้าง Scatter Plot ด้วย Altair เพื่อทำ Matrix 4 ช่อง
        scatter = alt.Chart(df).mark_circle(size=250, opacity=0.8).encode(
            x=alt.X('Mix %:Q', title='ความนิยม (Popularity / Mix %)', scale=alt.Scale(zero=False)),
            y=alt.Y('Profit Margin (THB):Q', title='กำไรต่อจาน (Profitability)', scale=alt.Scale(zero=False)),
            color=alt.Color('Category:N', scale=alt.Scale(
                domain=['🌟 Star (ดาวเด่น)', '🐴 Plowhorse (ม้างาน)', '🧩 Puzzle (ปริศนา)', '🐶 Dog (หมาน้อย)'],
                range=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
            ), legend=alt.Legend(title="กลุ่มเมนู", orient='bottom')),
            tooltip=['Menu Item', 'Profit Margin (THB)', 'Mix %', 'Sold Qty', 'Category']
        ).properties(height=400)
        
        # เส้นตัดเพื่อแบ่ง 4 Quadrants
        x_rule = alt.Chart(pd.DataFrame({'x': [avg_mix]})).mark_rule(color='black', strokeDash=[5,5]).encode(x='x:Q')
        y_rule = alt.Chart(pd.DataFrame({'y': [avg_margin]})).mark_rule(color='black', strokeDash=[5,5]).encode(y='y:Q')
        
        st.altair_chart((scatter + x_rule + y_rule).interactive(), use_container_width=True)

    with c2:
        # แสดงตารางผลลัพธ์ย่อ
        st.markdown("##### 📌 สรุปเมนูตามกลุ่ม")
        display_df = df[['Menu Item', 'Profit Margin (THB)', 'Mix %', 'Category']].sort_values(by='Profit Margin (THB)', ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
    st.divider()
    st.subheader("3. 🧠 ที่ปรึกษากลยุทธ์ AI (AI Strategy Consultant)")
    st.caption("ส่งข้อมูลเมนูที่ถูกจัดกลุ่มเสร็จแล้วให้ AI วางแผนการตลาดและปรับปรุงต้นทุนเพื่อเพิ่มกำไรสูงสุด")
    
    if st.button("✨ 2. สร้างแผนกลยุทธ์ระดับผู้บริหาร (Generate AI Strategy)"):
        with st.spinner("AI กำลังวิเคราะห์และเรียบเรียงกลยุทธ์..."):
            strategy_text = generate_ai_strategy(api_key_input, df)
            st.markdown(f'<div class="strategy-card">{strategy_text}</div>', unsafe_allow_html=True)
            
st.markdown("<br><hr><center><small>Built with Streamlit, Pandas, Altair & Gemini AI | Professional Portfolio Project</small></center>", unsafe_allow_html=True)