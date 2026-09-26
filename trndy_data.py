import streamlit as st
import feedparser
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="AI Market Radar & Confluence Engine", layout="wide")

# VADER Lexicon డౌన్‌లోడ్ (కేవలం ఒక్కసారే రన్ అవుతుంది)
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_vader()

# స్కానింగ్ కోసం ప్రధాన నిఫ్టీ 50 హెవీవెయిట్ & ట్రెండింగ్ స్టాక్స్
SCANNER_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
    "BHARTIARTL", "ITC", "LT", "TATAMOTORS", "SUNPHARMA", "BAJFINANCE",
    "MARUTI", "AXISBANK", "KOTAKBANK", "TATASTEEL", "ADANIENT", "ZOMATO"
]

def calculate_rsi(series, period=14):
    """ZeroDivision నివారించి సురక్షితంగా లెక్కించే 14-Period RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    
    loss_val = loss.iloc[-1]
    gain_val = gain.iloc[-1]
    
    if pd.isna(gain_val) or pd.isna(loss_val):
        return 50.0
    if loss_val == 0:
        return 100.0 if gain_val > 0 else 50.0
    
    rs = gain_val / loss_val
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(float(rsi), 2)

@st.cache_data(ttl=300)
def get_stock_technical_data(ticker_symbol):
    """లైవ్ CMP, % మార్పు, ATR, RSI మరియు పివట్ లెవెల్స్ లెక్కింపు"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="1mo")
        if data.empty or len(data) < 15:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        cmp = round(float(latest['Close']), 2)
        prev_close = float(prev['Close'])
        day_change = round(((cmp - prev_close) / prev_close) * 100, 2)
        
        data['H-L'] = data['High'] - data['Low']
        atr = round(float(data['H-L'].tail(14).mean()), 2)
        rsi = calculate_rsi(data['Close'], period=14)
        
        high = float(prev['High'])
        low = float(prev['Low'])
        close = float(prev['Close'])
        
        pivot = (high + low + close) / 3
        r1 = round((2 * pivot) - low, 2)
        s1 = round((2 * pivot) - high, 2)
        
        return {
            "CMP": cmp,
            "Change%": day_change,
            "ATR": atr,
            "RSI": rsi,
            "MaxUp": round(cmp + atr, 2),
            "MaxDown": round(cmp - atr, 2),
            "Pivot": round(pivot, 2),
            "R1": r1,
            "S1": s1
        }
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_stock_news_buzz(search_keyword, limit=4):
    """వార్తలు, సెంటిమెంట్ స్కోర్ మరియు న్యూస్ బజ్ స్కోర్ (News Buzz Score) లెక్కింపు"""
    query = f"{search_keyword} share price news India"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    total_score = 0.0
    impact_magnitude = 0.0
    articles = []
    
    for entry in feed.entries[:limit]:
        title = entry.title
        link = entry.link
        
        published_parsed = getattr(entry, 'published_parsed', None)
        pub_date = datetime(*published_parsed[:6]).strftime("%d-%b %I:%M %p") if published_parsed else "N/A"
        
        score = sia.polarity_scores(title)['compound']
        total_score += score
        impact_magnitude += abs(score)
        
        articles.append({
            "Title": title,
            "Score": score,
            "Link": link,
            "Date": pub_date
        })
        
    num_articles = len(articles)
    avg_score = round(total_score / num_articles, 2) if num_articles > 0 else 0.0
    # బజ్ స్కోర్ (0 - 100): వ్యాసాల సంఖ్య + వార్త పదాల తీవ్రత
    buzz_score = round(min((impact_magnitude * 30) + (num_articles * 10), 100.0), 1)
    
    return avg_score, buzz_score, articles

def compute_stock_confluence(rsi, day_change, news_score):
    """కాన్‌ఫ్లూయెన్స్ స్కోర్ (0-100)"""
    norm_mom = np.clip((day_change + 3.0) / 6.0 * 100, 0, 100)
    norm_rsi = np.clip(rsi, 0, 100)
    norm_news = np.clip((news_score + 1.0) / 2.0 * 100, 0, 100)
    return round((0.40 * norm_mom) + (0.30 * norm_rsi) + (0.30 * norm_news), 1)

# --- UI ప్రారంభం ---
st.title("⚡ AI Market Radar & News Buzz Intelligence Engine")
st.markdown("**లైవ్ ప్రైస్ మూమెంటం, RSI, సపోర్ట్/రెసిస్టెన్స్ జోన్స్ మరియు మీడియా బజ్ విశ్లేషణ (తెలుగు & English)**")

st.sidebar.header("🔍 నావిగేషన్ & సెట్టింగ్స్")
view_mode = st.sidebar.radio(
    "మోడ్ ఎంచుకోండి (Select Mode):",
    [
        "📊 Trending Stocks Radar (దిశా నిర్ధారణ స్కానర్)",
        "🔥 Hot News Buzz Stocks (వార్తల్లో బాగా ట్రెండ్ అవుతున్నవి)",
        "🔎 Single Stock Deep-Dive (సింగిల్ స్టాక్ విశ్లేషణ)"
    ]
)

exchange_suffix = st.sidebar.selectbox("ఎక్స్చేంజ్ (Exchange):", ["NSE (.NS)", "BSE (.BO)"])
suffix = ".NS" if "NSE" in exchange_suffix else ".BO"

if st.sidebar.button("🔄 రీ-స్కాన్ చేయండి (Clear Cache & Refresh)"):
    st.cache_data.clear()
    st.rerun()

# ----------------- 1. TRENDING STOCKS RADAR SCANNER -----------------
if view_mode == "📊 Trending Stocks Radar (దిశా నిర్ధారణ స్కానర్)":
    st.subheader("📋 Top Stocks Directional Radar (ఏ స్టాక్స్ పైకి/కిందికి వెళ్ళే అవకాశం ఉంది?)")
    
    scanned_data = []
    progress_bar = st.progress(0)

    for idx, sym in enumerate(SCANNER_UNIVERSE):
        ticker = f"{sym}{suffix}"
        tech = get_stock_technical_data(ticker)
        news_score, buzz_score, articles = fetch_stock_news_buzz(sym, limit=2)
        top_headline = articles[0]['Title'] if articles else "No Recent News"
        
        if tech:
            score = compute_stock_confluence(tech['RSI'], tech['Change%'], news_score)
            
            if score >= 65 and tech['RSI'] <= 72:
                bias = "🟢 UP (బుల్లిష్)"
                target = f"₹{tech['R1']} - ₹{tech['MaxUp']}"
                risk = f"₹{tech['Pivot']}"
            elif score <= 35 and tech['RSI'] >= 28:
                bias = "🔴 DOWN (బేరిష్)"
                target = f"₹{tech['S1']} - ₹{tech['MaxDown']}"
                risk = f"₹{tech['Pivot']}"
            elif tech['RSI'] > 72:
                bias = "⚠️ FOMO TRAP (ఎగ్జిట్ జోన్)"
                target = "లాభాల స్వీకరణ"
                risk = f"₹{tech['R1']}"
            elif tech['RSI'] < 28:
                bias = "⚠️ BOUNCE RISK (షార్ట్ ట్రాప్)"
                target = "బౌన్స్ అవకాశం"
                risk = f"₹{tech['S1']}"
            else:
                bias = "⚪ RANGEBOUND (తటస్థం)"
                target = f"₹{tech['R1']}"
                risk = f"₹{tech['S1']}"
                
            scanned_data.append({
                "Stock (స్టాక్)": sym,
                "CMP (ధర)": tech['CMP'],
                "Change%": tech['Change%'],
                "RSI": tech['RSI'],
                "News Score": news_score,
                "Prediction Score": score,
                "Expected Direction (దిశ)": bias,
                "Target Zone (లక్ష్యం)": target,
                "Support / Stop": risk,
                "Top Headline": top_headline
            })
            
        progress_bar.progress((idx + 1) / len(SCANNER_UNIVERSE))

    progress_bar.empty()

    if scanned_data:
        df_radar = pd.DataFrame(scanned_data)
        
        up_stocks = df_radar[df_radar['Expected Direction (దిశ)'].str.contains("🟢 UP")]
        down_stocks = df_radar[df_radar['Expected Direction (దిశ)'].str.contains("🔴 DOWN")]
        trap_stocks = df_radar[df_radar['Expected Direction (దిశ)'].str.contains("⚠️")]

        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 పెరిగే అవకాశం ఉన్న స్టాక్స్ (Bullish)", len(up_stocks))
        c2.metric("🔴 పడే అవకాశం ఉన్న స్టాక్స్ (Bearish)", len(down_stocks))
        c3.metric("⚠️ ట్రాప్ / రివర్సల్ రిస్క్ స్టాక్స్", len(trap_stocks))
        
        st.divider()

        col_up, col_down = st.columns(2)
        with col_up:
            st.success("### 🟢 Top Bullish Contenders (పైకి వెళ్లే అవకాశం ఉన్నవి)")
            if not up_stocks.empty:
                for _, row in up_stocks.iterrows():
                    st.markdown(f"**{row['Stock (స్టాక్)']}** (CMP: ₹{row['CMP (ధర)']}) | స్కోర్: `{row['Prediction Score']}` | న్యూస్ స్కోర్: `{row['News Score']}`")
                    st.caption(f"🎯 **టార్గెట్ రేంజ్:** {row['Target Zone (లక్ష్యం)']} | స్టాప్‌లాస్: {row['Support / Stop']}")
                    st.caption(f"📰 *{row['Top Headline']}*")
                    st.write("---")
            else:
                st.info("ప్రస్తుతం బలమైన బుల్లిష్ కాన్‌ఫ్లూయెన్స్ ఉన్న స్టాక్స్ లేవు.")

        with col_down:
            st.error("### 🔴 Top Bearish Contenders (కిందికి పడే అవకాశం ఉన్నవి)")
            if not down_stocks.empty:
                for _, row in down_stocks.iterrows():
                    st.markdown(f"**{row['Stock (స్టాక్)']}** (CMP: ₹{row['CMP (ధర)']}) | స్కోర్: `{row['Prediction Score']}` | న్యూస్ స్కోర్: `{row['News Score']}`")
                    st.caption(f"🎯 **డౌన్‌సైడ్ టార్గెట్:** {row['Target Zone (లక్ష్యం)']} | రెసిస్టెన్స్: {row['Support / Stop']}")
                    st.caption(f"📰 *{row['Top Headline']}*")
                    st.write("---")
            else:
                st.info("ప్రస్తుతం బలమైన బేరిష్ ప్రెజర్ ఉన్న స్టాక్స్ లేవు.")

        st.divider()
        st.subheader("📊 పూర్తి వాచ్‌లిస్ట్ టేబుల్ (Complete Pre-Market Matrix)")
        st.dataframe(
            df_radar.style.map(
                lambda v: 'background-color: rgba(40, 167, 69, 0.2); font-weight: bold;' if "🟢" in str(v) else (
                    'background-color: rgba(220, 53, 69, 0.2); font-weight: bold;' if "🔴" in str(v) else ''
                ),
                subset=['Expected Direction (దిశ)']
            ),
            use_container_width=True
        )

# ----------------- 2. HOT NEWS BUZZ STOCKS -----------------
elif view_mode == "🔥 Hot News Buzz Stocks (వార్తల్లో బాగా ట్రెండ్ అవుతున్నవి)":
    st.subheader("🔥 High Media Buzz Stocks (వార్తల్లో అత్యధికంగా ట్రెండ్ అవుతున్న స్టాక్స్)")
    st.markdown("తాజా వార్తల సంఖ్య, తేదీలు మరియు వాటి సెంటిమెంట్ తీవ్రత ఆధారంగా విశ్లేషణ.")

    buzz_list = []
    prog = st.progress(0)

    for i, sym in enumerate(SCANNER_UNIVERSE):
        ticker = f"{sym}{suffix}"
        tech = get_stock_technical_data(ticker)
        avg_score, buzz_score, articles = fetch_stock_news_buzz(sym, limit=4)

        if tech and buzz_score >= 35:
            if avg_score >= 0.15:
                news_bias = "🟢 Positive Catalyst (పాజిటివ్ వార్త)"
                action_text = f"బై టార్గెట్: ₹{tech['R1']} నుండి ₹{tech['MaxUp']}"
            elif avg_score <= -0.15:
                news_bias = "🔴 Negative Risk (నెగెటివ్ వార్త)"
                action_text = f"డౌన్‌సైడ్ రిస్క్: ₹{tech['S1']} నుండి ₹{tech['MaxDown']}"
            else:
                news_bias = "⚡ High Buzz - Neutral (చర్చల్లో ఉంది/స్పష్టత లేదు)"
                action_text = f"రేంజ్ బౌండ్ అస్థిరత: ₹{tech['S1']} - ₹{tech['R1']}"

            # అత్యంత తాజా వార్త తేదీని తీసుకోవడం
            latest_date = articles[0]['Date'] if articles else "N/A"

            buzz_list.append({
                "Stock": sym,
                "CMP": tech['CMP'],
                "Change%": tech['Change%'],
                "Buzz Score": buzz_score,
                "News Bias": news_bias,
                "Action": action_text,
                "Latest Date": latest_date,
                "Articles": articles
            })
        prog.progress((i + 1) / len(SCANNER_UNIVERSE))

    prog.empty()

    if buzz_list:
        df_buzz = pd.DataFrame(buzz_list).sort_values(by="Buzz Score", ascending=False)
        st.info(f"💡 ప్రస్తుతం **{len(df_buzz)}** స్టాక్స్ బలమైన వార్తలతో మీడియా మరియు మార్కెట్‌లో ట్రెండ్ అవుతున్నాయి.")

        for _, item in df_buzz.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 4])
                
                # స్టాక్ వివరాలు
                c1.markdown(f"### **{item['Stock']}**")
                c1.write(f"ధర (CMP): **₹{item['CMP']}** ({item['Change%']}%)")
                c1.caption(f"🕒 తాజా వార్త సమయం: **{item['Latest Date']}**")
                
                # స్కోర్ & సెంటిమెంట్
                c2.markdown(f"**బజ్ స్కోర్: `{item['Buzz Score']} / 100`**")
                c2.write(f"స్టేటస్: **{item['News Bias']}**")
                c2.caption(f"🎯 {item['Action']}")
                
                # తేదీ వారీ వార్తలు (Date-wise Expandable List)
                with c3:
                    st.markdown("**📅 తేదీల వారీగా వార్తలు (Date-wise Articles):**")
                    for art in item['Articles']:
                        art_score = art['Score']
                        tag = "🟢" if art_score >= 0.05 else ("🔴" if art_score <= -0.05 else "⚪")
                        st.markdown(
                            f"- {tag} **[{art['Date']}]** [{art['Title']}]({art['Link']}) `(Score: {art_score})`"
                        )
                st.divider()
    else:
        st.warning("ప్రస్తుతం చెప్పుకోదగ్గ వార్తల తీవ్రత (High News Buzz) ఉన్న స్టాక్స్ లేవు.")

# ----------------- 3. SINGLE STOCK DEEP-DIVE -----------------
else:
    st.subheader("🔎 సింగిల్ స్టాక్ డీప్-డైవ్ అనలిసిస్")
    sym_input = st.text_input("స్టాక్ సింబల్ ఎంటర్ చేయండి (ఉదా: TATAMOTORS, INFY, ZOMATO, SBIN):", "RELIANCE").strip().upper()
    ticker = f"{sym_input}{suffix}"
    
    with st.spinner(f"{sym_input} డేటా లోడ్ అవుతోంది..."):
        tech = get_stock_technical_data(ticker)
        news_score, buzz_score, articles = fetch_stock_news_buzz(sym_input, limit=4)
        
    if tech:
        score = compute_stock_confluence(tech['RSI'], tech['Change%'], news_score)
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ధర (CMP)", f"₹{tech['CMP']}", f"{tech['Change%']}%")
        m2.metric("కాన్‌ఫ్లూయెన్స్ స్కోర్", f"{score} / 100")
        m3.metric("14-Day RSI", f"{tech['RSI']}")
        m4.metric("న్యూస్ బజ్ స్కోర్", f"{buzz_score} / 100")
        
        st.progress(score / 100.0)
        
        if score >= 65 and tech['RSI'] <= 72:
            st.success(f"🟢 **బుల్లిష్ సిగ్నల్:** ప్రైస్ లక్ష్యం **₹{tech['R1']}** నుండి గరిష్టంగా **₹{tech['MaxUp']}** వరకు చేరే అవకాశం ఉంది.")
        elif score <= 35 and tech['RSI'] >= 28:
            st.error(f"🔴 **బేరిష్ సిగ్నల్:** ప్రైస్ లక్ష్యం **₹{tech['S1']}** నుండి కనిష్టంగా **₹{tech['MaxDown']}** వరకు పడే అవకాశం ఉంది.")
        elif tech['RSI'] > 72:
            st.warning(f"⚠️ **FOMO TRAP:** RSI అధికంగా ఉంది ({tech['RSI']}). రెసిస్టెన్స్ ₹{tech['R1']} వద్ద రివర్సల్ రిస్క్ ఎక్కువ.")
        elif tech['RSI'] < 28:
            st.warning(f"⚠️ **BOUNCE RISK:** స్టాక్ ఓవర్‌సోల్డ్ జోన్‌లో ఉంది. సపోర్ట్ ₹{tech['S1']} నుండి బౌన్స్ అవ్వచ్చు.")
        else:
            st.info(f"⚪ **రేంజ్ బౌండ్:** స్పష్టమైన డైరెక్షన్ లేదు. Pivot ₹{tech['Pivot']} మరియు లెవెల్స్ మధ్య ట్రేడ్ అవుతుంది.")
            
        st.markdown(f"""
        | సపోర్ట్ జోన్స్ (Support) | పివట్ (Pivot) | రెసిస్టెన్స్ జోన్స్ (Resistance) | రోజువారీ పరిధి (Expected Range) |
        | :---: | :---: | :---: | :---: |
        | **S1:** ₹{tech['S1']} | **Pivot:** ₹{tech['Pivot']} | **R1:** ₹{tech['R1']} | **Max Up:** ₹{tech['MaxUp']} <br> **Max Down:** ₹{tech['MaxDown']} |
        """)
        
        st.divider()
        st.subheader("📰 తాజా వార్తలు & లింకులు:")
        for art in articles:
            st.markdown(f"- [{art['Title']}]({art['Link']}) &nbsp;&nbsp; `(Score: {art['Score']})`")
    else:
        st.error("డేటా లభించలేదు. దయచేసి సరైన NSE/BSE సింబల్ ఎంటర్ చేశారో లేదో సరిచూడండి.")