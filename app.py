import streamlit as st
import feedparser
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="Universal Stock & Index Confluence Engine", layout="wide")

# VADER Lexicon డౌన్‌లోడ్
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ప్రీసెట్ ఇండెక్స్ డేటా
INDEX_DATA = {
    "NIFTY 50": {"ticker": "^NSEI", "search_term": "Nifty 50 share market India"},
    "BANK NIFTY": {"ticker": "^NSEBANK", "search_term": "Bank Nifty share market India"},
    "SENSEX (BSE)": {"ticker": "^BSESN", "search_term": "BSE Sensex share market India"}
}

# ప్రముఖ స్టాక్స్ సూచనల కోసం (Suggestions List)
POPULAR_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", 
    "BHARTIARTL", "ITC", "LT", "TATAMOTORS", "SUNPHARMA", "BAJFINANCE",
    "MARUTI", "AXISBANK", "KOTAKBANK", "TATASTEEL", "ADANIENT", "ZOMATO"
]

def calculate_rsi(series, period=14):
    """14-Period RSI లెక్కింపు"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50.0

def get_stock_technical_data(ticker_symbol):
    """లైవ్ ప్రైస్, మార్పు %, ATR రేంజ్, RSI మరియు జోన్ లెవెల్స్ లెక్కిస్తుంది"""
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo")
        if data.empty or len(data) < 14:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        cmp = round(latest['Close'], 2)
        day_change = round(((cmp - prev['Close']) / prev['Close']) * 100, 2)
        
        # ATR (14 Days)
        data['H-L'] = data['High'] - data['Low']
        atr = round(data['H-L'].tail(14).mean(), 2)
        rsi = calculate_rsi(data['Close'], period=14)
        
        # Pivot Levels
        high, low, close = prev['High'], prev['Low'], prev['Close']
        pivot = (high + low + close) / 3
        r1 = round((2 * pivot) - low, 2)
        r2 = round(pivot + (high - low), 2)
        s1 = round((2 * pivot) - high, 2)
        s2 = round(pivot - (high - low), 2)
        
        return {
            "CMP": cmp,
            "Change%": day_change,
            "ATR": atr,
            "RSI": rsi,
            "MaxUp": round(cmp + atr, 2),
            "MaxDown": round(cmp - atr, 2),
            "Pivot": round(pivot, 2),
            "R1": r1, "R2": r2,
            "S1": s1, "S2": s2
        }
    except Exception:
        return None

def fetch_stock_news(search_keyword, limit=4):
    """Google News RSS ద్వారా తాజా వార్తలు & సెంటిమెంట్ స్కోరింగ్"""
    query = f"{search_keyword} share price news India"
    encoded_query = urllib.parse.quote(query)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(rss_url)
    news_list = []
    total_score = 0.0
    
    for entry in feed.entries[:limit]:
        title = entry.title
        link = entry.link
        
        published_parsed = getattr(entry, 'published_parsed', None)
        pub_date = datetime(*published_parsed[:6]).strftime("%d-%b %I:%M %p") if published_parsed else "N/A"
            
        score = sia.polarity_scores(title)['compound']
        total_score += score
        
        if abs(score) >= 0.40:
            impact_text = "🔥 High Impact (తీవ్ర ప్రభావం)"
        elif abs(score) >= 0.15:
            impact_text = "⚡ Moderate Impact (మధ్యస్థ ప్రభావం)"
        else:
            impact_text = "💧 Low Impact (స్వల్ప ప్రభావం)"

        if score >= 0.05:
            sentiment_text = "🟢 Bullish (పాజిటివ్)"
        elif score <= -0.05:
            sentiment_text = "🔴 Bearish (నెగెటివ్)"
        else:
            sentiment_text = "⚪ Neutral (తటస్థం)"

        news_list.append({
            "Date": pub_date,
            "Title": title,
            "Score": score,
            "Impact": impact_text,
            "Sentiment": sentiment_text,
            "URL": link
        })
        
    avg_score = round(total_score / len(news_list), 3) if news_list else 0.0
    return news_list, avg_score

def compute_stock_confluence(rsi, day_change, news_score):
    """
    స్టాక్ కాన్‌ఫ్లూయెన్స్ స్కోర్ (0-100):
    - 40% ప్రైస్ మూమెంటం (Day Change %)
    - 30% RSI
    - 30% న్యూస్ సెంటిమెంట్
    """
    norm_mom = np.clip((day_change + 3.0) / 6.0 * 100, 0, 100)
    norm_rsi = np.clip(rsi, 0, 100)
    norm_news = np.clip((news_score + 1.0) / 2.0 * 100, 0, 100)
    return round((0.40 * norm_mom) + (0.30 * norm_rsi) + (0.30 * norm_news), 1)

# --- UI ప్రారంభం ---
st.title("🔎 Live Stock & Index Confluence Predictor")
st.markdown("**ఏదైనా NSE/BSE స్టాక్ సింబల్ ఎంటర్ చేయగానే లైవ్ ప్రైస్, జోన్స్, వార్తల ప్రభావం & ప్రిడిక్షన్ స్కోర్ లభిస్తుంది.**")

# సైడ్‌బార్ మోడ్ ఎంపిక
mode = st.sidebar.radio("విశ్లేషణ మోడ్ (Select Mode):", ["🔍 Search Any Stock (వ్యక్తిగత స్టాక్)", "🏛️ Major Indices (ప్రధాన ఇండెక్స్‌లు)"])

if mode == "🔍 Search Any Stock (వ్యక్తిగత స్టాక్)":
    st.sidebar.subheader("స్టాక్ ఎంపిక (Select / Type Stock)")
    
    # యూజర్ స్వయంగా టైప్ చేసే ఆప్షన్
    user_symbol_input = st.sidebar.text_input("స్టాక్ సింబల్ టైప్ చేయండి (ఉదా: RELIANCE, ZOMATO, SBIN):", "").strip().upper()
    
    # డ్రాప్‌డౌన్ నుండి ఎంచుకునే ఆప్షన్
    preset_choice = st.sidebar.selectbox("లేదా పాపులర్ స్టాక్స్ నుండి ఎంచుకోండి:", ["-- సెలెక్ట్ చేయండి --"] + POPULAR_STOCKS)
    
    if user_symbol_input:
        target_name = user_symbol_input
    elif preset_choice != "-- సెలెక్ట్ చేయండి --":
        target_name = preset_choice
    else:
        target_name = "RELIANCE"  # డిఫాల్ట్ స్టాక్

    # ఎక్స్చేంజ్ సెలెక్టర్
    exchange = st.sidebar.selectbox("ఎక్స్చేంజ్ (Exchange):", ["NSE (.NS)", "BSE (.BO)"])
    suffix = ".NS" if "NSE" in exchange else ".BO"
    target_ticker = f"{target_name}{suffix}"
    search_keyword = f"{target_name} share"

else:
    target_name = st.sidebar.selectbox("ఇండెక్స్ ఎంచుకోండి (Select Index):", list(INDEX_DATA.keys()))
    target_ticker = INDEX_DATA[target_name]["ticker"]
    search_keyword = INDEX_DATA[target_name]["search_term"]

news_count = st.sidebar.slider("వార్తల సంఖ్య (News Count):", 2, 6, 3)

if st.sidebar.button("🔄 డేటా రీఫ్రెష్ చేయండి"):
    st.rerun()

# డేటా విశ్లేషణ ప్రారంభం
with st.spinner(f"{target_name} టెక్నికల్ డేటా మరియు తాజా వార్తలను లోడ్ చేస్తోంది..."):
    tech_data = get_stock_technical_data(target_ticker)
    news_items, avg_news_score = fetch_stock_news(search_keyword, limit=news_count)

if tech_data:
    confluence = compute_stock_confluence(tech_data['RSI'], tech_data['Change%'], avg_news_score)

    # న్యూస్ ఇంపాక్ట్ స్టేటస్
    if abs(avg_news_score) >= 0.20:
        news_verdict = "🔥 News Impact is High & Active (వార్తల ప్రభావం బలంగా ఉంది)"
    elif abs(avg_news_score) >= 0.05:
        news_verdict = "⚡ News Impact is Moderate (వార్తల ప్రభావం మధ్యస్థంగా ఉంది)"
    else:
        news_verdict = "⚪ News Impact is Neutral (వార్తల ప్రభావం తక్కువ/తటస్థంగా ఉంది)"

    # ఎథికల్ ట్రాప్ ఫిల్టర్లు & సిగ్నల్స్
    is_fomo_risk = tech_data['RSI'] > 72 and tech_data['CMP'] >= tech_data['R1']
    is_oversold_trap = tech_data['RSI'] < 28 and tech_data['CMP'] <= tech_data['S1']

    if is_fomo_risk:
        signal_state = "⚠️ FOMO TRAP WARNING (కొనవద్దు - బుల్ ట్రాప్ ప్రమాదం)"
        action_note = f"English: RSI is Overbought ({tech_data['RSI']}) near Resistance R1 ({tech_data['R1']}). High risk of sharp reversal.\n\nతెలుగు: స్టాక్ ఇప్పటికే బాగా పెరిగింది. రెసిస్టెన్స్ వద్ద ఉంది కాబట్టి ఇప్పుడు కొత్తగా బై చేయడం చాలా రిస్క్."
        box_style = st.warning
    elif is_oversold_trap:
        signal_state = "⚠️ BREAKDOWN TRAP WARNING (అమ్మవద్దు - షార్ట్ ట్రాప్ ప్రమాదం)"
        action_note = f"English: RSI is Oversold ({tech_data['RSI']}) at Support S1 ({tech_data['S1']}). Bounce back likely.\n\nతెలుగు: స్టాక్ సపోర్ట్ లెవెల్ వద్ద ఓవర్‌సోల్డ్ స్థితిలో ఉంది. ఇక్కడ అమ్మితే తిరిగి బౌన్స్ అయ్యే అవకాశం ఉంది."
        box_style = st.warning
    elif confluence >= 65:
        signal_state = "🟢 HIGH-PROBABILITY BULLISH (బలమైన కొనుగోలు సంకేతం)"
        action_note = f"English: Momentum, RSI and News are Positive. Target: ₹{tech_data['R1']} to ₹{tech_data['MaxUp']}. Stop Loss: ₹{tech_data['Pivot']}.\n\nతెలుగు: సాంకేతిక సూచికలు మరియు వార్తలు అనుకూలంగా ఉన్నాయి. టార్గెట్ లెవెల్: ₹{tech_data['R1']} నుండి గరిష్టంగా ₹{tech_data['MaxUp']}."
        box_style = st.success
    elif confluence <= 35:
        signal_state = "🔴 HIGH-PROBABILITY BEARISH (బలమైన అమ్మకాల సంకేతం)"
        action_note = f"English: Technical weakness and negative sentiment. Downside Target: ₹{tech_data['S1']} to ₹{tech_data['MaxDown']}. Stop Loss: ₹{tech_data['Pivot']}.\n\nతెలుగు: స్టాక్‌లో అమ్మకాల ఒత్తిడి మరియు నెగెటివ్ వార్తలు ఉన్నాయి. సపోర్ట్ టార్గెట్: ₹{tech_data['S1']} నుండి కనిష్టంగా ₹{tech_data['MaxDown']}."
        box_style = st.error
    else:
        signal_state = "🟡 CONSOLIDATION / NO-TRADE ZONE (రేంజ్ బౌండ్ - వేచి చూడండి)"
        action_note = f"English: Confluence is neutral ({confluence}/100). No clear trend.\n\nతెలుగు: స్టాక్‌లో స్పష్టమైన దిశ లేదు. Pivot మరియు S1/R1 మధ్య కన్సాలిడేట్ అవుతుంది."
        box_style = st.info

    # హెడర్ మరియు మెట్రిక్స్
    st.subheader(f"📌 {target_name} ({target_ticker}) అనలిటిక్స్ సారాంశం")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ప్రస్తుత ధర (CMP)", f"₹{tech_data['CMP']}", f"{tech_data['Change%']}%")
    c2.metric("కాన్‌ఫ్లూయెన్స్ స్కోర్ (0-100)", f"{confluence} / 100")
    c3.metric("14-Day RSI", f"{tech_data['RSI']}")
    c4.metric("న్యూస్ సెంటిమెంట్ స్కోర్", f"{avg_news_score}")

    st.progress(confluence / 100.0)
    st.caption(f"📢 **న్యూస్ ఇంపాక్ట్ స్థితి:** {news_verdict}")

    # ప్రిడిక్షన్ బాక్స్
    box_style(f"### {signal_state}\n{action_note}")

    # సపోర్ట్ & రెసిస్టెన్స్ జోన్స్
    st.subheader("📍 కీలక మార్కెట్ లెవెల్స్ & ఊహాత్మక పరిధి (Key Support & Resistance Zones)")
    st.markdown(f"""
    | సపోర్ట్ జోన్స్ (Support Levels) | పివట్ (Pivot Mean) | రెసిస్టెన్స్ జోన్స్ (Resistance Levels) | రోజువారీ పరిధి (Expected Day Range) |
    | :---: | :---: | :---: | :---: |
    | **S2:** ₹{tech_data['S2']} <br> **S1:** ₹{tech_data['S1']} | **Pivot:** ₹{tech_data['Pivot']} | **R1:** ₹{tech_data['R1']} <br> **R2:** ₹{tech_data['R2']} | **గరిష్ట పెరుగుదల (Max Up):** ₹{tech_data['MaxUp']} <br> **గరిష్ట పతనం (Max Down):** ₹{tech_data['MaxDown']} |
    """)

    st.divider()

    # వార్తల విభాగం
    st.subheader(f"📰 {target_name} పై తాజా వార్తలు & ప్రభావ విశ్లేషణ (Live News & Impact)")
    if news_items:
        for item in news_items:
            with st.expander(f"🕒 {item['Date']} | {item['Sentiment']} | {item['Impact']}"):
                st.markdown(f"**హెడ్‌లైన్:** [{item['Title']}]({item['URL']})")
                st.write(f"- **సెంటిమెంట్ స్కోర్:** `{item['Score']}`")
                st.write(f"- **ప్రభావ స్థాయి:** {item['Impact']}")
    else:
        st.info("ఈ స్టాక్‌పై ఇటీవల ఎటువంటి వార్తలు నమోదు కాలేదు.")

else:
    st.error(f"❌ '{target_name}' కోసం డేటా లభించలేదు. దయచేసి సరైన NSE/BSE సింబల్ ఎంటర్ చేశారో లేదో సరిచూడండి (ఉదాహరణకు: TATASTEEL, INFY, HDFCBANK).")
