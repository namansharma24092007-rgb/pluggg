import os, json, re, sqlite3, random, difflib, urllib.request
from datetime import datetime, timedelta
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

D = os.path.dirname(os.path.abspath(__file__)); DB = os.path.join(D, "upi_guard.db")
if os.path.exists(os.path.join(D, ".env")):
    for l in open(os.path.join(D, ".env")):
        if "=" in l and not l.startswith("#"):
            k, v = l.strip().split("=", 1); os.environ.setdefault(k, v.strip("\"'"))
KEY = os.getenv("XAI_API_KEY"); GM = os.getenv("GROK_MODEL", "grok-3")

FEATS = ["amount_log", "night", "amount_ratio", "new_payee", "payee_age", "complaints", "fan_in", "collect", "qr",
         "link", "refund_word", "kyc_word", "urgency_word", "name_mismatch", "screenshare", "remote_app"]
FL = {"amount_log": "Large amount", "night": "Unusual hour (night)", "amount_ratio": "Amount far above your usual",
      "new_payee": "First payment to this payee", "payee_age": "Very new payee account", "complaints": "Past fraud complaints on payee",
      "fan_in": "Payee receives from many senders (mule pattern)", "collect": "Collect request (money goes OUT)",
      "qr": "Paying via QR", "link": "Paying via link", "refund_word": "'Refund/cashback' wording",
      "kyc_word": "KYC / account-block wording", "urgency_word": "Urgency pressure", "name_mismatch": "Name shown differs from account holder",
      "screenshare": "Screen sharing active", "remote_app": "Remote-access app installed"}
CATS = {"collect_refund": "Collect-request refund scam", "qr_swap": "QR code swap", "remote_access": "Remote-access app scam",
        "fake_kyc": "Fake KYC / bank alert", "mule_account": "Mule-account chain", "fake_job_investment": "Fake job / investment",
        "suspicious_other": "Other suspicious", "none": "Legitimate"}
M = {}
app = Flask(__name__, static_folder=D)

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

def init():
    c = db()
    c.executescript("""create table if not exists txns(id integer primary key autoincrement,ts text,payee_vpa text,payee_name text,
display_name text,amount real,mode text,message text,risk real,model_risk real,decision text,category text,features text,reasons text,
contrib text,ai text,user_action text,analyst_label text,source text);
create table if not exists payees(vpa text primary key,age_days real,complaints real,fan_in real);
create table if not exists patterns(id integer primary key autoincrement,name text unique,description text,keywords text,status text,hits integer default 0,created text);
create table if not exists model_meta(id integer primary key autoincrement,ts text,metrics text);""")
    if not c.execute("select 1 from payees").fetchone():
        c.executemany("insert into payees values(?,?,?,?)", [
            ("ramesh.kirana@oksbi", 900, 0, 6), ("chai.point@icici", 700, 0, 40), ("refund.help4u@ybl", 3, 7, 4),
            ("quickcash.9921@axl", 5, 4, 46), ("bsnl-kyc.update@upi", 1, 12, 9), ("easyearn.task77@ibl", 4, 9, 52)])
    c.commit(); c.close()

# ---------- synthetic data ----------
def synth(n, seed):
    r = np.random.default_rng(seed); X = []; y = []; C = []; U = r.uniform
    P = lambda q: float(r.random() < q)
    for c in r.choice(["legit", "collect_refund", "qr_swap", "remote_access", "fake_kyc", "mule_account"], n, p=[.74, .06, .06, .06, .05, .03]):
        amt = float(np.clip(r.lognormal(6.3, 1.1), 10, 60000)); mode = r.choice(["pay", "qr", "link", "collect"], p=[.45, .38, .08, .09])
        new = P(.2)
        f = dict(night=P(.06), new_payee=new, payee_age=U(1, 60) if new else U(60, 2500), complaints=float(r.poisson(.05)),
                 fan_in=float(r.integers(1, 9)), refund_word=P(.01), kyc_word=P(.01), urgency_word=P(.03),
                 name_mismatch=float(np.clip(abs(r.normal(0, .06)), 0, 1)) if P(.9) else U(.1, .35), screenshare=P(.01), remote_app=P(.03))
        ratio = float(np.clip(r.lognormal(0, .6), .05, 30))
        if c == "collect_refund":
            mode = "collect"; amt = U(300, 20000); ratio = U(.5, 6)
            f.update(refund_word=P(.9), new_payee=P(.9), payee_age=U(1, 40), urgency_word=P(.4), complaints=float(r.poisson(2)))
        elif c == "qr_swap":
            mode = "qr"; f.update(name_mismatch=U(.45, .95) if P(.85) else U(0, .3), new_payee=P(.8), payee_age=U(1, 90), complaints=float(r.poisson(1)))
        elif c == "remote_access":
            amt = U(5000, 100000); ratio = U(2, 25)
            f.update(screenshare=P(.85), remote_app=P(.9), new_payee=P(.8), payee_age=U(1, 60), urgency_word=P(.5))
        elif c == "fake_kyc":
            mode = "link"; amt = U(1, 10) if P(.5) else U(500, 15000)
            f.update(kyc_word=P(.9), urgency_word=P(.8), new_payee=P(.9), payee_age=U(1, 30), name_mismatch=U(.3, .9) if P(.5) else U(0, .2))
        elif c == "mule_account":
            amt = U(2000, 40000); f.update(fan_in=U(15, 80), payee_age=U(1, 14), complaints=float(r.poisson(3) + 1), new_payee=P(.7), night=P(.4))
        f.update(amount_log=float(np.log1p(amt)), amount_ratio=ratio, collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"))
        X.append([float(f[k]) for k in FEATS]); y.append(0 if c == "legit" else 1); C.append(c)
    return np.array(X), np.array(y), np.array(C)

def mets(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return dict(precision=round(tp / max(tp + fp, 1), 4), recall=round(tp / max(tp + fn, 1), 4), fpr=round(fp / max(fp + tn, 1), 4),
                accuracy=round((tp + tn) / len(y), 4), tp=tp, fp=fp, fn=fn, tn=tn)

def train():
    X, y, C = synth(8000, 1)
    Xa, Xt, ya, yt, Ca, Ct = train_test_split(X, y, C, test_size=.25, stratify=y, random_state=7)
    w = np.ones(len(ya)); c = db()
    fb = c.execute("select features,analyst_label from txns where analyst_label is not null").fetchall(); c.close()
    if fb:
        Xa = np.vstack([Xa, [[json.loads(r["features"])[k] for k in FEATS] for r in fb]])
        ya = np.concatenate([ya, [1 if r["analyst_label"] == "fraud" else 0 for r in fb]]); w = np.concatenate([w, np.full(len(fb), 15.)])
    clf = RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced", random_state=7, n_jobs=-1).fit(Xa, ya, sample_weight=w)
    p = clf.predict_proba(Xt)[:, 1]; pred = (p >= .4).astype(int)
    ai = {FEATS[i]: i for i in range(len(FEATS))}
    base_pred = ((np.expm1(Xt[:, ai["amount_log"]]) > 10000) | ((Xt[:, ai["new_payee"]] == 1) & (np.expm1(Xt[:, ai["amount_log"]]) > 5000))).astype(int)
    m = dict(test_size=len(yt), fraud_in_test=int(yt.sum()), threshold=.4, model=mets(yt, pred), static_rule_baseline=mets(yt, base_pred),
             per_category_recall={k: round(float(pred[Ct == k].mean()), 3) for k in CATS if k in set(Ct) and k != "legit"},
             feedback_samples=len(fb), trained_at=datetime.now().isoformat(timespec="seconds"))
    M.update(clf=clf, base=np.median(Xa[ya == 0], axis=0), metrics=m)
    c = db(); c.execute("insert into model_meta(ts,metrics) values(?,?)", (m["trained_at"], json.dumps(m))); c.commit(); c.close()

def score(f):
    x = np.array([[float(f[k]) for k in FEATS]]); Z = [x[0]]
    for i in range(len(FEATS)):
        z = x[0].copy(); z[i] = M["base"][i]; Z.append(z)
    pr = M["clf"].predict_proba(np.array(Z))[:, 1]; rows = []
    for i, k in enumerate(FEATS):
        d = float(pr[0] - pr[i + 1])
        if d > .01: rows.append(dict(feature=k, label=FL[k], impact=round(d, 3), value=round(float(x[0, i]), 2)))
    return float(pr[0]), sorted(rows, key=lambda r: -r["impact"])[:6]

# ---------- context analysis ----------
REF = re.compile(r"refund|cashback|reversal|money back|reward|prize|lottery", re.I)
KYC = re.compile(r"kyc|account.{0,20}(block|suspend|freez|deactiv)|update.{0,15}(pan|aadhaar|aadhar)|verification fee|sim.{0,10}block", re.I)
URG = re.compile(r"urgent|immediately|today|tonight|last chance|expire|within \d+|final notice|disconnect", re.I)

def featurize(d):
    c = db(); vpa = d["payee_vpa"].strip().lower(); txt = d.get("message") or ""
    pr = c.execute("select * from payees where vpa=?", (vpa,)).fetchone()
    h = c.execute("select count(*) n from txns where payee_vpa=? and user_action='paid' and source in('history','live')", (vpa,)).fetchone()["n"]
    av = c.execute("select avg(amount) a from txns where user_action='paid' and source in('history','live')").fetchone()["a"] or 1500
    c.close(); amt = float(d["amount"]); mode = d.get("mode", "pay")
    reg, disp = (d.get("registered_name") or "").lower(), (d.get("display_name") or d.get("registered_name") or "").lower()
    mism = 1 - difflib.SequenceMatcher(None, reg, disp).ratio() if reg and disp else 0
    hr = int(d["hour"]) if d.get("hour") not in (None, "") else datetime.now().hour
    f = dict(amount_log=float(np.log1p(amt)), night=float(hr < 5 or hr >= 23), amount_ratio=min(amt / max(av, 1), 30), new_payee=float(h == 0),
             payee_age=float(pr["age_days"]) if pr else 3., complaints=float(pr["complaints"]) if pr else 0., fan_in=float(pr["fan_in"]) if pr else 1.,
             collect=float(mode == "collect"), qr=float(mode == "qr"), link=float(mode == "link"), refund_word=float(bool(REF.search(txt))),
             kyc_word=float(bool(KYC.search(txt))), urgency_word=float(bool(URG.search(txt))), name_mismatch=float(mism),
             screenshare=float(bool(d.get("screenshare"))), remote_app=float(bool(d.get("remote_app"))))
    return f

def indicators(f, d):
    I = []; a = lambda t, s: I.append(dict(text=t, severity=s)); amt = d["amount"]
    if f["collect"]: a(f"COLLECT request: entering your PIN sends ₹{amt:,.0f} OUT of your account.", "high" if f["refund_word"] else "medium")
    if f["collect"] and f["refund_word"]: a("'Refund' wording on a debit request: real refunds never need your PIN.", "high")
    if f["name_mismatch"] > .4: a(f"Name shown ('{d.get('display_name')}') does not match account holder ('{d.get('registered_name')}').", "high")
    if f["screenshare"] or f["remote_app"]: a("Screen-sharing / remote-access app detected during payment.", "high")
    if f["kyc_word"]: a("Message uses KYC / account-block wording typical of fake bank alerts.", "high")
    if f["urgency_word"]: a("Message pressures you to act urgently.", "medium")
    if f["fan_in"] >= 15 and f["payee_age"] < 30: a(f"Payee is {f['payee_age']:.0f} days old but received from {f['fan_in']:.0f} senders: mule-account pattern.", "high")
    if f["complaints"] >= 2: a(f"Payee has {f['complaints']:.0f} fraud complaints on record.", "high")
    if f["new_payee"] and f["amount_ratio"] > 4: a("First payment to this payee and well above your usual amount.", "medium")
    if f["night"]: a("Payment at an unusual hour.", "low")
    return I

def grok(ctx):
    if not KEY: return None
    sysm = ("You are a UPI fraud analyst for India. Given a payment context, an ML risk score and rule indicators, decide whether it is a scam. "
            "Known categories: collect_refund, qr_swap, remote_access, fake_kyc, mule_account, fake_job_investment, none. If it is a scam fitting none of these, "
            "use category 'novel' with new_pattern_name (snake_case) and 3-6 lowercase keywords from the message that characterise it. "
            'Reply ONLY with JSON: {"fraud_confidence":0-1,"category":str,"new_pattern_name":str|null,"keywords":[str],'
            '"user_warning":"one short sentence to the payer starting with the key fact","explanation":"max 2 plain sentences citing concrete indicators"}')
    body = {"model": GM, "temperature": .1, "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": json.dumps(ctx)}]}
    rq = urllib.request.Request("https://api.x.ai/v1/chat/completions", json.dumps(body).encode(),
                                {"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "User-Agent": "upi-guard/1.0"})
    try:
        out = json.load(urllib.request.urlopen(rq, timeout=25))["choices"][0]["message"]["content"]
        return json.loads(re.search(r"\{.*\}", out, re.S).group())
    except Exception as e:
        return {"error": str(e)[:160]}

def decide(d, f, source="live", use_ai=True):
    p, contrib = score(f); ind = indicators(f, d); c = db(); txt = (d.get("message") or "").lower()
    learned = None
    for r in c.execute("select * from patterns where status='approved'").fetchall():
        kws = json.loads(r["keywords"]); need = 1 if len(kws) < 2 else 2
        if sum(k in txt for k in kws) >= need:
            learned = r; c.execute("update patterns set hits=hits+1 where id=?", (r["id"],))
            ind.insert(0, dict(text=f"Matches analyst-approved pattern '{r['name']}' learned from earlier cases.", severity="high"))
    g = None
    if use_ai and (p >= .2 or txt.strip() or ind):
        g = grok(dict(amount_inr=d["amount"], mode=d.get("mode"), payee_vpa=d["payee_vpa"], registered_name=d.get("registered_name"),
                      display_name=d.get("display_name"), message=d.get("message"), screen_sharing=bool(d.get("screenshare")),
                      remote_app=bool(d.get("remote_app")), ml_risk=round(p, 3), indicators=[i["text"] for i in ind]))
    gc = float(g["fraud_confidence"]) if g and "fraud_confidence" in g else None
    risk = p if gc is None else .6 * p + .4 * gc
    if learned: risk = max(risk, .75)
    dec = "BLOCK" if risk >= .7 else "WARN" if risk >= .4 else "ALLOW"
    cat = "none"
    if dec != "ALLOW":
        if learned: cat = learned["name"]
        elif f["screenshare"] or (f["remote_app"] and d["amount"] > 5000): cat = "remote_access"
        elif f["collect"] and (f["refund_word"] or f["complaints"] >= 2): cat = "collect_refund"
        elif f["qr"] and f["name_mismatch"] > .4: cat = "qr_swap"
        elif f["kyc_word"] and (f["link"] or f["urgency_word"]): cat = "fake_kyc"
        elif f["fan_in"] >= 15 and f["payee_age"] < 30: cat = "mule_account"
        elif g and g.get("category") not in (None, "none", "novel") and g.get("category") != "none": cat = g["category"]
        elif g and g.get("category") == "novel" and g.get("new_pattern_name"): cat = g["new_pattern_name"]
        else: cat = "suspicious_other"
    newp = None
    if g and g.get("category") == "novel" and gc and gc >= .6 and g.get("new_pattern_name"):
        try:
            c.execute("insert into patterns(name,description,keywords,status,created) values(?,?,?,?,?)",
                      (g["new_pattern_name"], g.get("explanation", ""), json.dumps([k.lower() for k in g.get("keywords", [])]), "pending", datetime.now().isoformat(timespec="seconds")))
            newp = g["new_pattern_name"]
        except sqlite3.IntegrityError: pass
    amt = d["amount"]
    head = (f"You are about to PAY, not receive, ₹{amt:,.0f}" if f["collect"] and dec != "ALLOW" else
            "Screen sharing detected while paying" if cat == "remote_access" else
            f"QR shows '{d.get('display_name')}' but pays '{d.get('registered_name')}'" if cat == "qr_swap" else
            "This looks like a scam" if dec != "ALLOW" else "No fraud signals found")
    warn = (g or {}).get("user_warning") or (ind[0]["text"] if ind and dec != "ALLOW" else "Payment looks consistent with your normal behaviour.")
    expl = (g or {}).get("explanation") or ("; ".join(i["text"] for i in ind[:3]) if ind else "Amount, payee and context are all normal.")
    out = dict(risk=round(risk, 3), model_risk=round(p, 3), ai_confidence=gc, decision=dec, category=cat, category_label=CATS.get(cat, cat.replace("_", " ").title()),
               headline=head, warning=warn, explanation=expl, indicators=ind, contributions=contrib, ai_used=bool(g and "error" not in g),
               ai_error=(g or {}).get("error") if g else ("No XAI_API_KEY set: running ML + rules only" if use_ai else None), new_pattern=newp)
    cur = c.execute("insert into txns(ts,payee_vpa,payee_name,display_name,amount,mode,message,risk,model_risk,decision,category,features,reasons,contrib,ai,source) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (datetime.now().isoformat(timespec="seconds"), d["payee_vpa"], d.get("registered_name"), d.get("display_name"), amt, d.get("mode"), d.get("message"),
                     risk, p, dec, cat, json.dumps(f), json.dumps(ind), json.dumps(contrib), json.dumps(dict(warning=warn, explanation=expl, ai_used=out["ai_used"])), source))
    out["id"] = cur.lastrowid; c.commit(); c.close(); return out

def seed():
    c = db()
    if c.execute("select 1 from txns").fetchone(): c.close(); return
    now = datetime.now()
    for a in (320, 450, 2100, 600, 380):
        c.execute("insert into txns(ts,payee_vpa,amount,mode,decision,category,user_action,source,features,reasons,contrib,ai,risk) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ((now - timedelta(days=random.randint(1, 20))).isoformat(timespec="seconds"), "ramesh.kirana@oksbi", a, "pay", "ALLOW", "none", "paid", "history", "{}", "[]", "[]", "{}", .02))
    X, y, C = synth(160, 99); random.seed(5)
    for i, row in enumerate(X):
        f = dict(zip(FEATS, map(float, row))); p, _ = score(f); dec = "BLOCK" if p >= .7 else "WARN" if p >= .4 else "ALLOW"
        cat = "none" if dec == "ALLOW" else (C[i] if C[i] != "legit" else "suspicious_other")
        ts = now - timedelta(days=random.randint(0, 6), hours=random.randint(0, 23), minutes=random.randint(0, 59))
        c.execute("insert into txns(ts,payee_vpa,payee_name,amount,mode,risk,model_risk,decision,category,features,reasons,contrib,ai,user_action,source) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (ts.isoformat(timespec="seconds"), f"user{i}@ok{random.choice(['sbi', 'hdfc', 'axis'])}", "Unknown", float(np.expm1(f["amount_log"])),
                   "collect" if f["collect"] else "qr" if f["qr"] else "link" if f["link"] else "pay", p, p, dec, cat, json.dumps(f),
                   json.dumps(indicators(f, dict(amount=float(np.expm1(f["amount_log"])), display_name="", registered_name=""))), "[]", "{}",
                   "cancelled" if dec != "ALLOW" and random.random() < .6 else "paid", "seed"))
    c.commit(); c.close()

# ---------- API ----------
@app.get("/")
def home(): return send_from_directory(D, "index.html")

@app.post("/api/analyze")
def analyze():
    d = request.get_json(force=True)
    try: d["amount"] = float(d["amount"]); assert d["amount"] > 0 and d.get("payee_vpa")
    except Exception: return jsonify(error="amount (>0) and payee_vpa are required"), 400
    return jsonify(decide(d, featurize(d)))

@app.post("/api/txn/<int:i>/action")
def action(i):
    c = db(); c.execute("update txns set user_action=? where id=?", (request.json.get("action"), i)); c.commit(); c.close(); return jsonify(ok=True)

@app.post("/api/txn/<int:i>/label")
def label(i):
    c = db(); c.execute("update txns set analyst_label=? where id=?", (request.json.get("label"), i)); c.commit(); c.close(); return jsonify(ok=True)

@app.get("/api/transactions")
def txns():
    q = "select * from txns where source!='history'" + (" and decision!='ALLOW'" if request.args.get("flagged") == "1" else "") + " order by ts desc limit 150"
    c = db(); rows = []
    for r in c.execute(q).fetchall():
        r = dict(r); r["reasons"] = json.loads(r["reasons"] or "[]"); r["contrib"] = json.loads(r["contrib"] or "[]"); r["ai"] = json.loads(r["ai"] or "{}")
        r["category_label"] = CATS.get(r["category"], (r["category"] or "").replace("_", " ").title()); r.pop("features", None); rows.append(r)
    c.close(); return jsonify(rows)

@app.get("/api/stats")
def stats():
    c = db(); w = "where source!='history'"; q = lambda s: c.execute(s).fetchall()
    t = q(f"select count(*) n,sum(decision!='ALLOW') fl,sum(decision='BLOCK') bl,sum(case when decision!='ALLOW' then amount else 0 end) val from txns {w}")[0]
    cats = [dict(category=r["category"], label=CATS.get(r["category"], r["category"].replace("_", " ").title()), n=r["n"]) for r in
            q(f"select category,count(*) n from txns {w} and decision!='ALLOW' group by category order by n desc")]
    tr = [dict(day=r["d"], total=r["n"], flagged=r["fl"]) for r in q(f"select substr(ts,1,10) d,count(*) n,sum(decision!='ALLOW') fl from txns {w} group by d order by d desc limit 7")][::-1]
    pend = q("select count(*) n from patterns where status='pending'")[0]["n"]; c.close()
    return jsonify(total=t["n"], flagged=t["fl"] or 0, blocked=t["bl"] or 0, value_at_risk=t["val"] or 0, categories=cats, trend=tr, pending_patterns=pend, ai_enabled=bool(KEY), ai_model=GM)

@app.get("/api/metrics")
def metrics(): return jsonify(M["metrics"])

@app.post("/api/retrain")
def retrain(): train(); return jsonify(M["metrics"])

@app.get("/api/patterns")
def patterns():
    c = db(); r = [dict(x, keywords=json.loads(x["keywords"])) for x in c.execute("select * from patterns order by id desc").fetchall()]; c.close(); return jsonify(r)

@app.post("/api/patterns/<int:i>")
def pattern_set(i):
    c = db(); c.execute("update patterns set status=? where id=?", (request.json.get("status"), i)); c.commit(); c.close(); return jsonify(ok=True)

init(); train(); seed()
if __name__ == "__main__":
    print("AI:", f"Grok ({GM})" if KEY else "OFF (set XAI_API_KEY)"); app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
