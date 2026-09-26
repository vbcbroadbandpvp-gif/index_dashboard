import streamlit as st
import feedparser
import urllib.parse
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

st.set_page_config(page_title="Market Analytics (Telugu & English)", layout="wide")

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

INDEX_DATA = {
    "NIFTY 50": {
        "ticker": "^NSEI",
        "search_term": "Nifty 50 share market India",
        "components": {
            "HDFCBANK": {"symbol": "HDFCBANK.NS", "weight": 11.5},
            "RELIANCE": {"symbol": "RELIANCE.NS", "weight": 9.2},
            "ICICIBANK": {"symbol": "ICICIBANK.NS", "weight": 7.8},
            "INFY": {"symbol": "INFY.NS", "weight": 5.8},
            "TCS": {"symbol": "TCS.NS", "weight": 3.9},
            "ITC": {"symbol": "ITC.NS", "weight": 3.8},
            "LT": {"symbol": "LT.NS", "weight": 3.7},
            "AXISBANK": {"symbol": "AXISBANK.NS", "weight": 3.3},
            "SBIN": {"symbol": "SBIN.NS", "weight": 2.9},
            "BHARTIARTL": {"symbol": "BHARTIARTL.NS", "weight": 2.8}
        }
    },
    "BANK NIFTY": {
        "ticker": "^NSEBANK",
        "search_term": "Bank Nifty share market India",
        "components": {
            "HDFCBANK": {"symbol": "HDFCBANK.NS", "weight": 28.0},
            "ICICIBANK": {"symbol": "ICICIBANK.NS", "weight": 23.0},
            "SBIN": {"symbol": "SBIN.NS", "weight": 10.0},
            "AXISBANK": {"symbol": "AXISBANK.NS", "weight": 9.5},
            "KOTAKBANK": {"symbol": "KOTAKBANK.NS", "weight": 9.0},
            "INDUSINDBK": {"symbol": "INDUSINDBK.NS", "weight": 5.5},
            "BANKBARODA": {"symbol": "BANKBARODA.NS", "weight": 3.0},
            "PNB": {"symbol": "PNB.NS", "weight": 2.5},
            "AUBANK": {"symbol": "AUBANK.NS", "weight": 2.0},
            "FEDERALBNK": {"symbol": "FEDERALBNK.NS", "weight": 2.0}
        }
    },
    "SENSEX (BSE)": {
        "ticker": "^BSESN",
        "search_term": "BSE Sensex share market India",
        "components": {
            "HDFCBANK": {"symbol": "HDFCBANK.BO", "weight": 14.5},
            "RELIANCE": {"symbol": "RELIANCE.BO", "weight": 11.2},
            "ICICIBANK": {"symbol": "ICICIBANK.BO", "weight": 9.5},
            "INFY": {"symbol": "INFY.BO", "weight": 7.1},
            "ITC": {"symbol": "ITC.BO", "weight": 5.2},
            "TCS": {"symbol": "TCS.BO", "weight": 4.8},
            "LT": {"symbol": "LT.BO", "weight": 4.6},
            "AXISBANK": {"symbol": "AXISBANK.BO", "weight": 4.0},
            "SBIN": {"symbol": "SBIN.BO", "weight": 3.6},
            "BHARTIARTL": {"symbol": "BHARTIARTL.BO", "weight": 3.4}
        }
    }
}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2) if not pd.isna(rsi.iloc[-1]) else 50.0

def get_technical_data(ticker_symbol):
    try:
        data = yf.Ticker(ticker_symbol).history(period="1mo")
        if data.empty or len(data) < 15:
            return None
        
        latest = data.iloc[-1]
        prev = data.iloc[-2]
        
        cmp = round(latest['Close'], 2)
        day_change = round(((cmp - prev['Close']) / prev['Close']) * 100, 2)
        
        data['H-L'] = data['High'] - data['Low']
        atr = round(data['H-L'].tail(14).mean(), 2)
        rsi = calculate_rsi(data['Close'], period=14)
        
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

def fetch_news_and_sentiment(query_term, limit=3):
    encoded_query = urllib.parse.quote(query_term)
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
        
        # న్యూస్ ఇంపాక్ట్ కేటగిరీ (తెలుగు & ఇంగ్లీష్)
        if abs(score) >= 0.40:
            impact_text = "🔥 High Impact (తీవ్ర ప్రభావం)"
        elif abs(score) >= 0.15:
            impact_text = "⚡ Moderate Impact (మధ్యస్థ ప్రభావం)"
        else:
            impact_text = "💧 Low Impact (స్వల్ప ప్రభావం)"

        if score >= 0.05:
            sentiment_text = "🟢 Bullish (లాభాల సంకేతం)"
        elif score <= -0.05:
            sentiment_text = "🔴 Bearish (నష్టాల సంకేతం)"
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

def compute_confluence_score(rsi, weighted_mom, news_sentiment):
    norm_mom = np.clip((weighted_mom + 2.0) / 4.0 * 100, 0, 100)
    norm_rsi = np.clip(rsi, 0, 100)
    norm_news = np.clip((news_sentiment + 1.0) / 2.0 * 100, 0, 100)
    final_score = round((0.40 * norm_mom) + (0.30 * norm_rsi) + (0.30 * norm_news), 1)
    return final_score

# --- UI ప్రారంభం ---
st.title("🏛️ Multi-Index Analytics Dashboard (తెలుగు & English)")
st.markdown("**Real-Time Index Health, Weighted Momentum, Confluence Probability & Verified News Impact**")

selected_index_name = st.sidebar.selectbox("ఇండెక్స్ ఎంచుకోండి (Select Index):", list(INDEX_DATA.keys()))
news_limit = st.sidebar.slider("వార్తల సంఖ్య (Number of News Items):", 1, 5, 3)

if st.sidebar.button("🔄 రీఫ్రెష్ చేయండి (Refresh Data)"):
    st.rerun()

index_info = INDEX_DATA[selected_index_name]

with st.spinner("టెక్నికల్ డేటా మరియు న్యూస్ ఇంపాక్ట్ విశ్లేషిస్తోంది..."):
    idx_tech = get_technical_data(index_info["ticker"])
    idx_news, idx_avg_news_score = fetch_news_and_sentiment(index_info["search_term"], limit=news_limit)
    
    comp_results = []
    weighted_price_change = 0.0
    total_weight_tracked = 0.0
    
    for name, item in index_info["components"].items():
        tech = get_technical_data(item["symbol"])
        if tech:
            weight = item["weight"]
            weighted_price_change += (tech["Change%"] * weight)
            total_weight_tracked += weight
            comp_results.append({
                "Stock (స్టాక్)": name,
                "Weight% (వెయిట్)": weight,
                "CMP (ధర)": tech["CMP"],
                "Change% (మార్పు)": tech["Change%"],
                "RSI (ఆర్.ఎస్.ఐ)": tech["RSI"],
                "ATR (రోజువారీ రేంజ్)": tech["ATR"]
            })

    composite_momentum_pct = round(weighted_price_change / total_weight_tracked, 2) if total_weight_tracked > 0 else 0.0

if idx_tech:
    confluence = compute_confluence_score(idx_tech['RSI'], composite_momentum_pct, idx_avg_news_score)

    # News Impact వర్గీకరణ
    if abs(idx_avg_news_score) >= 0.20:
        news_verdict = "✅ News Impact is Active & Significant (వార్తల ప్రభావం బలంగా ఉంది)"
    elif abs(idx_avg_news_score) >= 0.05:
        news_verdict = "⚡ News Impact is Moderate (వార్తల ప్రభావం మధ్యస్థంగా ఉంది)"
    else:
        news_verdict = "⚪ News Impact is Neutral / Muted (వార్తల ప్రభావం తక్కువగా ఉంది)"

    # ఎథికల్ ఫిల్టర్లు & సిగ్నల్స్ (తెలుగు & English)
    is_fomo_risk = idx_tech['RSI'] > 72 and idx_tech['CMP'] >= idx_tech['R1']
    is_oversold_trap = idx_tech['RSI'] < 28 and idx_tech['CMP'] <= idx_tech['S1']

    if is_fomo_risk:
        signal_state = "⚠️ FOMO TRAP WARNING (కొనవద్దు - ట్రాప్ అయ్యే ప్రమాదం)"
        action_note = f"English: RSI is Overbought ({idx_tech['RSI']}) at Resistance R1 ({idx_tech['R1']}). High risk of fake breakout.\n\nతెలుగు: మార్కెట్ ఇప్పటికే బాగా పెరిగింది (RSI: {idx_tech['R1']}). ఇక్కడ కొత్తగా బై చేయడం రిస్క్."
        box_style = st.warning
    elif is_oversold_trap:
        signal_state = "⚠️ BREAKDOWN TRAP WARNING (అమ్మవద్దు - షార్ట్ ట్రాప్ ప్రమాదం)"
        action_note = f"English: Market is oversold at Support S1 ({idx_tech['S1']}). Shorting here has poor Risk-to-Reward.\n\nతెలుగు: మార్కెట్ సపోర్ట్ జోన్ వద్ద ఓవర్‌సోల్డ్‌గా ఉంది. ఇక్కడ అమ్మడం సరైనది కాదు."
        box_style = st.warning
    elif confluence >= 65:
        signal_state = "🟢 HIGH-PROBABILITY BULLISH EXPANSION (బలమైన కొనుగోలు సూచన)"
        action_note = f"English: Technicals, Momentum & News are Positive. Target: {idx_tech['R1']} to {idx_tech['MaxUp']}. Stop Loss: {idx_tech['Pivot']}.\n\nతెలుగు: స్టాక్స్ మరియు వార్తలు అనుకూలంగా ఉన్నాయి. టార్గెట్ లెవెల్: {idx_tech['R1']} నుండి గరిష్టంగా {idx_tech['MaxUp']}."
        box_style = st.success
    elif confluence <= 35:
        signal_state = "🔴 HIGH-PROBABILITY BEARISH BREAKDOWN (బలమైన అమ్మకాల సూచన)"
        action_note = f"English: Components and News indicate heavy selling pressure. Target: {idx_tech['S1']} to {idx_tech['MaxDown']}. Stop Loss: {idx_tech['Pivot']}.\n\nతెలుగు: హెవీవెయిట్స్ మరియు వార్తల్లో నెగెటివ్ ఒత్తిడి ఉంది. టార్గెట్ సపోర్ట్: {idx_tech['S1']} నుండి కనిష్టంగా {idx_tech['MaxDown']}."
        box_style = st.error
    else:
        signal_state = "🟡 CHOPPY / NO-TRADE ZONE (రేంజ్ బౌండ్ - వేచి చూడటం మంచిది)"
        action_note = f"English: Score is neutral ({confluence}/100). Technicals are conflicting.\n\nతెలుగు: మార్కెట్‌లో స్పష్టమైన ట్రెండ్ లేదు. కొత్త పొజిషన్లకు దూరంగా ఉండటం మంచిది."
        box_style = st.info

    # కీ మెట్రిక్స్ డిస్‌ప్లే
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ఇండెక్స్ ధర (CMP)", f"{idx_tech['CMP']}", f"{idx_tech['Change%']}%")
    c2.metric("కాన్‌ఫ్లూయెన్స్ స్కోర్ (0-100)", f"{confluence} / 100")
    c3.metric("వార్తల సెంటిమెంట్ (News Score)", f"{idx_avg_news_score}")
    c4.metric("హెవీవెయిట్స్ మూమెంటం", f"{composite_momentum_pct}%")

    st.progress(confluence / 100.0)

    # న్యూస్ ఇంపాక్ట్ స్టేటస్ బ్యానర్
    st.caption(f"📢 **న్యూస్ ఇంపాక్ట్ స్థితి (News Impact Status):** {news_verdict} | Overall Sentiment: `{idx_avg_news_score}`")

    # డెసిషన్ బాక్స్
    box_style(f"### {signal_state}\n{action_note}")

    # జోన్ మ్యాప్ (సపోర్ట్ & రెసిస్టెన్స్ లెవెల్స్)
    st.subheader("📍 మార్కెట్ జోన్ లెవెల్స్ (Market Support & Resistance Zones)")
    st.markdown(f"""
    | సపోర్ట్ జోన్స్ (Supports / Buy Zone) | పివట్ (Pivot Mean) | రెసిస్టెన్స్ జోన్స్ (Resistances / Sell Zone) | రోజువారీ అంచనా పరిధి (Expected Range) |
    | :---: | :---: | :---: | :---: |
    | **S2:** {idx_tech['S2']} <br> **S1:** {idx_tech['S1']} | **Pivot:** {idx_tech['Pivot']} | **R1:** {idx_tech['R1']} <br> **R2:** {idx_tech['R2']} | **గరిష్ట పెరుగుదల (Max Up):** {idx_tech['MaxUp']} <br> **గరిష్ట పతనం (Max Down):** {idx_tech['MaxDown']} |
    """)

st.divider()

# కాంపోనెంట్ స్టాక్స్ పనితీరు
st.subheader(f"🧱 {selected_index_name} హెవీవెయిట్ స్టాక్స్ పనితీరు (Component Breakdown)")
if comp_results:
    df_comp = pd.DataFrame(comp_results)
    st.dataframe(
        df_comp.style.map(
            lambda v: 'color: #28a745; font-weight: bold;' if v > 0 else ('color: #dc3545; font-weight: bold;' if v < 0 else ''),
            subset=['Change% (మార్పు)']
        ),
        use_container_width=True
    )

st.divider()

# వార్తలు & ఇంపాక్ట్ విశ్లేషణ (తెలుగు & ఇంగ్లీష్)
st.subheader(f"📰 తాజా వార్తలు & ప్రభావ విశ్లేషణ (Live News & Verified Impact)")
for item in idx_news:
    with st.expander(f"🕒 {item['Date']} | {item['Sentiment']} | {item['Impact']}"):
        st.markdown(f"**హెడ్‌లైన్ (Headline):** [{item['Title']}]({item['URL']})")
        st.write(f"- **సెంటిమెంట్ స్కోర్ (Score):** `{item['Score']}`")
        st.write(f"- **ప్రభావ స్థాయి (Impact Level):** {item['Impact']}")