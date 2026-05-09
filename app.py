"""
TrendPulse v5 — Fixed Backend
- Platform filter no longer blocks keyword search
- Each platform has its OWN trending topics
- Live update endpoint for real-time data
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from collections import Counter
import re, os, random
from datetime import datetime

app = Flask(__name__)
app.secret_key = "trendpulse_v5_2024"
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "posts.csv")

# ── Sentiment words ────────────────────────────────────────────────────────────
POS = {"amazing","great","excellent","fantastic","wonderful","love","awesome","incredible",
       "brilliant","outstanding","superb","excited","happy","thrilled","beautiful","perfect",
       "best","masterpiece","stunning","spectacular","breathtaking","unstoppable","legendary",
       "proud","champion","genius","electric","unbelievable","win","winning","record","gold",
       "victory","dream","earned","revolutionary","innovative","magical","viral","milestone",
       "epic","inspirational","brave","powerful","worth","helpful","creative","strong","success",
       "life-changing","saved","generous","transformative","transcendent","gorgeous","glorious"}

NEG = {"terrible","awful","horrible","bad","worst","hate","disgusting","disappointing",
       "frustrating","annoying","angry","sad","depressed","devastated","crashing","failing",
       "problem","issue","burnout","ruining","unacceptable","harmful","unbearable","wrong",
       "exposed","dropped","shadowban","fake","heartbreaking","injury","broken","ban","lose",
       "losing","disaster","sacked","toxic","dangerous","unfair","demonetized","cringe",
       "chaotic","snubbed","trouble","failed","scam","misleading","addiction","brutal",
       "overrated","slow","boring","waste","damaged","concerning","terrible","controversial"}

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from",
    "is","it","this","that","are","was","be","have","has","had","do","does","did","will",
    "would","can","could","should","may","might","not","no","so","as","we","i","my","me",
    "our","you","your","he","she","they","their","its","about","more","just","get","got",
    "still","than","how","what","which","who","when","where","why","all","every","like",
    "much","very","too","also","up","out","into","over","after","before","now","again",
    "back","even","new","time","year","day","make","look","know","want","come","see",
    "well","really","going","been","here","there","some","most","only","absolutely",
    "completely","literally","always","never","ever","once","film","movie","show","series",
    "platform","social","media","post","content","video","channel","page","account"
}

def load_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    for col in ['category','platform','dataset']:
        if col not in df.columns: df[col] = 'General'
    df['rating'] = pd.to_numeric(df.get('rating', 5), errors='coerce').fillna(5.0)
    df['helpful_votes'] = pd.to_numeric(df.get('helpful_votes', 0), errors='coerce').fillna(0)
    return df

def sentiment(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    p = sum(1 for w in words if w in POS)
    n = sum(1 for w in words if w in NEG)
    t = p + n
    s = 0.0 if t == 0 else (p - n) / t
    if s > 0.1:    return "positive", round(s, 2)
    elif s < -0.1: return "negative", round(s, 2)
    else:          return "neutral",  round(s, 2)

def keywords(texts, n=15):
    words = []
    for text in texts:
        clean = re.sub(r'http\S+|@\w+|#\w+|[^a-zA-Z\s]', ' ', str(text))
        words += [w for w in clean.lower().split() if len(w) > 3 and w not in STOPWORDS]
    return Counter(words).most_common(n)

def hashtags(col, n=12):
    tags = []
    for s in col:
        if pd.notna(s): tags += [t.lower() for t in re.findall(r'#\w+', str(s))]
    return Counter(tags).most_common(n)

def trending_for(df, n=10):
    """Get platform-specific trending topics from a dataframe slice."""
    topics = []
    for _, row in df.iterrows():
        tags = re.findall(r'#\w+', str(row.get('hashtags', '')))
        eng  = int(row.get('likes', 0)) + int(row.get('retweets', 0))
        for tag in tags:
            topics.append({"tag": tag.lower(), "eng": eng,
                           "cat": str(row.get('category', 'General')),
                           "plat": str(row.get('platform', 'Twitter'))})
    if not topics: return []
    tdf = pd.DataFrame(topics)
    agg = tdf.groupby('tag').agg(
        score=('eng','sum'), count=('tag','count'),
        category=('cat','first'), platform=('plat','first')
    ).reset_index()
    agg['avg_eng'] = (agg['score'] / agg['count']).round(0)
    return agg.nlargest(n, 'score').to_dict(orient='records')

def over_time(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.strftime('%H:00')
    df['sent'] = df['text'].apply(lambda t: sentiment(t)[0])
    g = df.groupby(['hour','sent']).size().unstack(fill_value=0)
    for c in ['positive','negative','neutral']:
        if c not in g.columns: g[c] = 0
    return {"labels":g.index.tolist(),"positive":g['positive'].tolist(),
            "negative":g['negative'].tolist(),"neutral":g['neutral'].tolist()}

def platform_stats(full):
    PLATS = ["Twitter","Instagram","TikTok","YouTube","LinkedIn","Snapchat","Pinterest","Reddit","WhatsApp"]
    res = {}
    for p in PLATS:
        sub = full[full['platform']==p]
        if sub.empty: continue
        sents = sub['text'].apply(lambda t: sentiment(t)[0])
        c = sents.value_counts().to_dict(); n = len(sub)
        res[p] = {
            "total":n,
            "positive": round(c.get("positive",0)/n*100,1),
            "negative": round(c.get("negative",0)/n*100,1),
            "neutral":  round(c.get("neutral",0)/n*100,1),
            "avg_likes": round(sub['likes'].mean(),0),
            "trending": trending_for(sub, 6)   # ← EACH PLATFORM OWN TRENDING
        }
    return res

def dataset_stats(full):
    res = {}
    for d in ["IMDb","Amazon","Twitter"]:
        sub = full[full['dataset']==d]
        if sub.empty: continue
        sents = sub['text'].apply(lambda t: sentiment(t)[0])
        c = sents.value_counts().to_dict(); n = len(sub)
        res[d] = {
            "total":n,
            "positive":round(c.get("positive",0)/n*100,1),
            "negative":round(c.get("negative",0)/n*100,1),
            "neutral": round(c.get("neutral",0)/n*100,1),
            "avg_rating": round(sub['rating'].mean(),1)
        }
    return res

def rating_dist(full):
    def bucket(sub):
        b = {'1-3':0,'4-6':0,'7-8':0,'9-10':0}
        for r in sub['rating']:
            if r<=3: b['1-3']+=1
            elif r<=6: b['4-6']+=1
            elif r<=8: b['7-8']+=1
            else: b['9-10']+=1
        return b
    return {"imdb":bucket(full[full['dataset']=='IMDb']),
            "amazon":bucket(full[full['dataset']=='Amazon'])}

def top_posts(df, field, value, n=5):
    sub = df[df[field]==value].copy()
    if sub.empty: return []
    sub['sentiment'] = sub['text'].apply(lambda t: sentiment(t)[0])
    cols = [c for c in ['text','sentiment','likes','retweets','hashtags','category','platform','dataset','rating'] if c in sub.columns]
    return sub.nlargest(n,'likes')[cols].fillna('').to_dict(orient='records')

# ── Live update data (simulated real-time) ─────────────────────────────────────
LIVE_TEMPLATES = {
    "YouTube":  ["#MrBeast hits 300M subscribers! 🎉","#YouTube Shorts paying creators $1 per 1000 views now!","New #YouTube feature: AI auto-chapters for all videos!","#Pewdiepie announces surprise comeback stream tonight!","#YouTube Premium now includes 4K downloads!"],
    "Instagram": ["#Instagram rolls out vertical feed for all users!","#Reels now supports 10-minute videos!","#Instagram Threads crosses 200M daily users!","New #Instagram feature: AI background removal in stories!","#Instagram Collab posts now show on both profiles' grids!"],
    "TikTok":   ["#TikTok launches in-app AI music generator!","#TikTok shop reaches $10B GMV this quarter!","New #TikTok LIVE feature: multi-guest streams up to 10!","#TikTok ban reversed in US courts! Creators celebrate!","#TikTokMadeMeBuyIt trend reaches 50B views!"],
    "Twitter":  ["#Twitter X launches audio-only spaces with recording!","Elon Musk announces #Twitter premium at $1/month!","#CommunityNotes adds image fact-checking feature!","#Twitter launches job board for tech creators!","New #X feature: edit tweets up to 24 hours after posting!"],
    "Sports":   ["BREAKING: #Messi signs extension with Inter Miami!","#Cricket World Cup 2026 schedule announced!","#NBA announces new In-Season Tournament format!","#F1 adds 3 new race locations for 2027 season!","#Olympics 2028 LA ticket sales open tomorrow!"],
    "Entertainment": ["#TaylorSwift announces 2026 Eras Tour Vol.2!","#Marvel drops first look at Avengers: Secret Wars!","#BTS announces reunion concert after military service!","#Netflix acquires rights to new Tolkien series!","#Oscars 2027 moves to February for first time!"],
}

def get_live_updates():
    updates = []
    for platform, msgs in LIVE_TEMPLATES.items():
        msg = random.choice(msgs)
        updates.append({
            "platform": platform,
            "text": msg,
            "time": datetime.now().strftime("%H:%M"),
            "likes": random.randint(1000, 50000),
            "sentiment": "positive" if any(w in msg.lower() for w in ["hit","launch","announce","reaches","celebrates"]) else "neutral"
        })
    return updates

# ── Main analysis ──────────────────────────────────────────────────────────────
def analyze(keyword="", category="All", platform="All", dataset="All"):
    full = load_data()

    # Search across FULL dataset first (keyword never blocked by platform filter)
    if keyword:
        kw = keyword.strip().lstrip('#')
        mask = (full['text'].str.contains(kw, case=False, na=False) |
                full['hashtags'].str.contains(kw, case=False, na=False))
        filtered = full[mask].copy()
    else:
        filtered = full.copy()

    # Apply category/platform/dataset filters AFTER keyword search
    if category != "All": filtered = filtered[filtered['category']==category]
    if platform  != "All": filtered = filtered[filtered['platform']==platform]
    if dataset   != "All": filtered = filtered[filtered['dataset']==dataset]

    if filtered.empty:
        return {"total":0,"error":"No results found"}

    sents = filtered['text'].apply(sentiment)
    filtered = filtered.copy()
    filtered['sentiment'] = sents.apply(lambda x: x[0])
    filtered['score']     = sents.apply(lambda x: x[1])

    counts = filtered['sentiment'].value_counts().to_dict()
    total  = len(filtered)
    sc = {"positive":counts.get("positive",0),"negative":counts.get("negative",0),"neutral":counts.get("neutral",0)}
    sp = {k: round(v/total*100,1) for k,v in sc.items()}

    # Breakdowns (always from full data)
    cat_order = ["Movies","Sports","Entertainment","Instagram","TikTok","YouTube","Twitter","LinkedIn","Reddit","Snapchat","Pinterest","WhatsApp","Electronics","Fashion","Health","Books","General"]
    cat_bd = {c:int((full['category']==c).sum()) for c in cat_order if (full['category']==c).sum()>0}
    plat_bd = {p:int((full['platform']==p).sum()) for p in ["Twitter","Instagram","TikTok","YouTube","LinkedIn","Snapchat","Pinterest","Reddit","WhatsApp"] if (full['platform']==p).sum()>0}

    # Platform stats (each with own trending topics)
    plat_stats = platform_stats(full)

    return {
        "total":total,"keyword":keyword,"category":category,"platform":platform,"dataset":dataset,
        "sentiment_counts":sc,"sentiment_pct":sp,
        "top_keywords": [{"word":w,"count":c} for w,c in keywords(filtered['text'].tolist())],
        "top_hashtags": [{"tag":t,"count":c}  for t,c in hashtags(filtered['hashtags'].tolist())],

        # OVERALL trending (from filtered results)
        "trending_all":           trending_for(filtered, 10),

        # PLATFORM-SPECIFIC trending (each platform independent)
        "trending_youtube":       plat_stats.get("YouTube",{}).get("trending",[]),
        "trending_instagram":     plat_stats.get("Instagram",{}).get("trending",[]),
        "trending_tiktok":        plat_stats.get("TikTok",{}).get("trending",[]),
        "trending_twitter":       plat_stats.get("Twitter",{}).get("trending",[]),
        "trending_linkedin":      plat_stats.get("LinkedIn",{}).get("trending",[]),
        "trending_reddit":        plat_stats.get("Reddit",{}).get("trending",[]),

        # CATEGORY trending
        "trending_sports":        trending_for(full[full['category']=='Sports'],8),
        "trending_entertainment": trending_for(full[full['category']=='Entertainment'],8),
        "trending_movies":        trending_for(full[full['dataset']=='IMDb'],8),
        "trending_amazon":        trending_for(full[full['dataset']=='Amazon'],8),

        "trend_over_time": over_time(filtered),
        "sample_posts":    filtered.nlargest(6,'likes')[['text','sentiment','likes','retweets','hashtags','category','platform','dataset']].fillna('').to_dict(orient='records'),
        "avg_engagement":  round((filtered['likes']+filtered['retweets']).mean(),1),
        "avg_likes":       round(filtered['likes'].mean(),1),
        "avg_retweets":    round(filtered['retweets'].mean(),1),
        "cat_breakdown":   cat_bd,
        "platform_breakdown": plat_bd,
        "platform_stats":  plat_stats,
        "dataset_stats":   dataset_stats(full),
        "rating_dist":     rating_dist(full),
        "sports_posts":        top_posts(full,'category','Sports',5),
        "entertainment_posts": top_posts(full,'category','Entertainment',5),
        "movies_posts":        top_posts(full,'dataset','IMDb',5),
        "amazon_posts":        top_posts(full,'dataset','Amazon',5),
        "instagram_posts":     top_posts(full,'platform','Instagram',4),
        "tiktok_posts":        top_posts(full,'platform','TikTok',4),
        "youtube_posts":       top_posts(full,'platform','YouTube',4),
        "twitter_posts":       top_posts(full,'platform','Twitter',4),
        "linkedin_posts":      top_posts(full,'platform','LinkedIn',4),
        "reddit_posts":        top_posts(full,'platform','Reddit',4),
    }

# ── Routes ─────────────────────────────────────────────────────────────────────
USERS = {"admin":"admin123","demo":"demo123","user":"pass123"}

@app.route('/')
def index(): return redirect(url_for('dashboard') if session.get('logged_in') else url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        u,p = request.form.get('username','').strip(), request.form.get('password','').strip()
        if USERS.get(u)==p:
            session['logged_in'],session['username']=True,u
            return redirect(url_for('dashboard'))
        error="Invalid credentials. Try: admin / admin123"
    return render_template('login.html', error=error)

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'): return redirect(url_for('login'))
    return render_template('dashboard.html', username=session.get('username','User'))

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not session.get('logged_in'): return jsonify({"error":"Not authenticated"}),401
    d = request.get_json() or {}
    try:
        return jsonify(analyze(d.get('keyword',''), d.get('category','All'),
                               d.get('platform','All'), d.get('dataset','All')))
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route('/api/live')
def api_live():
    """Live updates endpoint — called every 15 seconds by frontend."""
    if not session.get('logged_in'): return jsonify({"error":"Not authenticated"}),401
    return jsonify({"updates": get_live_updates(), "time": datetime.now().strftime("%H:%M:%S")})

if __name__ == '__main__':
    print("="*55)
    print("  TrendPulse v5  |  http://127.0.0.1:5000")
    print("  Login: admin / admin123")
    print("="*55)
    app.run(debug=True, port=5000)
