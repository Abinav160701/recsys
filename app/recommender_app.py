# ────────────────────────────────────────────────────────────────
# 0.  LOAD STATIC & LIVE DATA ON START-UP
# ────────────────────────────────────────────────────────────────
url="redis://default:caGsThrOKxjMqBtfAJsJIsDDnbZDQWxf@shortline.proxy.rlwy.net:43464"
#rdb = redis.from_url("redis://localhost:6379/0", decode_responses=False)
rdb = redis.from_url(url, decode_responses=False)
_unpickle = lambda k: pickle.loads(rdb.get(k))

print("⏳  loading artefacts from Redis …")
user_item   = _unpickle("cf:user_item")
u2i, s2j    = _unpickle("cf:u2i"), _unpickle("cf:s2j")
j2s         = {v: k for k, v in s2j.items()}

user_pprice = _unpickle("cf:price_pref")
u_brand     = _unpickle("cf:brand_pref")
u_cat       = _unpickle("cf:cat_pref")

user_norm   = np.frombuffer(rdb.get("cf:user_norm"), dtype="float32")
R           = sp.load_npz(io.BytesIO(rdb.get("cf:R")))
item_co     = sp.load_npz(io.BytesIO(rdb.get("cf:item_cooc")))

META = (pd.read_csv("./skus_metadata.csv",
                    usecols=["sku", "sale_price", "size_availability",
                             "brand", "l1", "l2", "color"])
        .set_index("sku"))
PRICE = META.sale_price.to_dict()
BRAND = META.brand.fillna("unknown").to_dict()
CAT   = (META.l1.fillna("unknown") + "▸" + META.l2.fillna("unknown")).to_dict()

# optional ad-hoc boost table ↓
try:
    adhoc = (pd.read_csv("boost_table.csv",
                         dtype={"boost_type": str, "key": str, "pct": float})
               .dropna(subset=["boost_type", "key", "pct"]))
    EXTRA_BOOST = {
        "brand": dict(adhoc.query("boost_type=='brand'")[["key", "pct"]].values),
        "cat":   dict(adhoc.query("boost_type=='cat'")  [["key", "pct"]].values),
    }
    print(f"✓  loaded ad-hoc boosts ({len(EXTRA_BOOST['brand'])} brands, "
          f"{len(EXTRA_BOOST['cat'])} cats)")
except FileNotFoundError:
    EXTRA_BOOST = {"brand": {}, "cat": {}}
    print("⚠️  boost_table.csv not found – skipping extra boosts")

print("✓  artefact load complete")

# ────────────────────────────────────────────────────────────────
# 1.  HELPERS & BUSINESS-LOGIC FUNCTIONS  (unchanged)
# ────────────────────────────────────────────────────────────────
def valid_sku(s: str) -> bool:
    return META.at[s, "size_availability"] >= .7 if s in META.index else False

def valid_user(u: str) -> bool:
    return len(user_item.get(u, {})) >= 1            # plug your own rule

price_of = PRICE.get
brand_of = BRAND.get
cat_of   = CAT.get

def _cosine_row(i: int) -> np.ndarray:
    sims = (R.getrow(i) @ R.T).A.ravel()
    sims[i] = 0.
    return sims / (user_norm[i] * user_norm + 1e-9)

def _minmax(d: dict) -> dict:
    if not d: return {}
    m = max(d.values())
    return {k: v/m for k, v in d.items()} if m else d

def _apply_profile_boost(user: str, scores: dict) -> None:
    bp, cp = set(u_brand.get(user, [])), set(u_cat.get(user, []))
    for s in scores:
        if brand_of(s) in bp: scores[s] *= 1.15
        if cat_of(s)   in cp: scores[s] *= 1.10

def _apply_adhoc_boost(scores: dict) -> None:
    for s in scores:
        b, c = brand_of(s), cat_of(s)
        if b in EXTRA_BOOST["brand"]:
            scores[s] *= 1 + EXTRA_BOOST["brand"][b] / 100.0
        if c in EXTRA_BOOST["cat"]:
            scores[s] *= 1 + EXTRA_BOOST["cat"][c]   / 100.0

def _user_side(user, target_sku, price_band):
    if user not in u2i or not valid_user(user): return {}
    sims = _cosine_row(u2i[user])
    scores = defaultdict(float)
    for n in sims.argsort()[-200:][::-1]:
        if sims[n] <= 0: break
        for j in R[n, :].indices:
            s = j2s[j]
            if s == target_sku or not valid_sku(s):  continue
            if price_band and not price_band(s):     continue
            scores[s] += sims[n]
    return scores

def _item_side(sku, price_band):
    if sku not in s2j: return {}
    row = item_co.getrow(s2j[sku])
    out = {}
    for idx, cnt in zip(row.indices, row.data):
        s = j2s[idx]
        if not valid_sku(s):          continue
        if price_band and not price_band(s): continue
        out[s] = float(cnt)
    return out

def recommend(user: str, sku: str,
              top_n=20, pct_band=.30, w_user=.7):
    if not valid_sku(sku):
        raise ValueError("SKU not live")
    p_ref = price_of(sku)
    if p_ref is None: return []

    if user in user_pprice and user_pprice[user]["p10"] <= p_ref <= user_pprice[user]["p90"]:
        lo = user_pprice[user]["p10"] * (1-pct_band)
        hi = user_pprice[user]["p90"] * (1+pct_band)
    else:
        lo, hi = p_ref*(1-pct_band), p_ref*(1+pct_band)
    in_band = lambda s: lo <= price_of(s) <= hi if price_of(s) else False

    u_scores = _minmax(_user_side(user, sku, in_band))
    i_scores = _minmax(_item_side(sku,        in_band))

    _apply_profile_boost(user, u_scores)
    _apply_profile_boost(user, i_scores)
    _apply_adhoc_boost(u_scores)
    _apply_adhoc_boost(i_scores)

    combined = {s: (w_user*u_scores.get(s, 0) +
                    (1-w_user)*i_scores.get(s, 0))
                for s in set(u_scores) | set(i_scores)
                if s != sku}

    if len(combined) < top_n:                          # back-fill
        pool = [s for s in META.index
                if valid_sku(s) and s not in combined and s != sku]
        nearest = heapq.nsmallest(top_n-len(combined),
                                  pool,
                                  key=lambda s: abs(price_of(s)-p_ref)
                                                 if price_of(s) else 9e9)
        for s in nearest:
            combined[s] = (u_scores.get(s, 0)+i_scores.get(s, 0))

    return heapq.nlargest(top_n, combined.items(), key=lambda x: x[1])

# ────────────────────────────────────────────────────────────────
# 2.  FASTAPI LAYER
# ────────────────────────────────────────────────────────────────
app = FastAPI(title="CF-Recommender", version="1.0")


def root():
    return {"msg": "CF-Recommender - see /docs for usage"}


class RecoRequest(BaseModel):
    user   : str
    sku    : str
    top_n  : int   = Field(20, ge=1, le=100)
    pct    : float = Field(.30, ge=0., le=.9)
    w_user : float = Field(.7,  ge=0., le=1.)

class RecoItem(BaseModel):
    sku   : str
    score : float

class RecoResponse(BaseModel):
    user : str
    sku  : str
    recs : List[RecoItem]


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend", response_model=RecoResponse)
def reco(req: RecoRequest):
    try:
        recs = recommend(req.user, req.sku,
                         top_n=req.top_n,
                         pct_band=req.pct,
                         w_user=req.w_user)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "user": req.user,
        "sku":  req.sku,
        "recs": [ {"sku": s, "score": float(sc)} for s, sc in recs ]
    }

# run with:  uvicorn reco_service:app --host 0.0.0.0 --port 8000 --workers 2


# run with:  uvicorn reco_service:app --host 0.0.0.0 --port 8000 --workers 2
