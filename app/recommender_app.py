# app/recommender_app.py
import os, io, pickle, heapq, asyncio, redis
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# FastAPI factory
# ------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="CF-Recommender", version="1.0")

    # ------------------------------------------------------------------
    # 0. globals that will be filled on startup
    # ------------------------------------------------------------------
    global rdb, user_item, u2i, s2j, j2s, user_pprice, u_brand, u_cat
    global user_norm, R, item_co, META, PRICE, BRAND, CAT, EXTRA_BOOST
    rdb = None          # helps mypy / editors; will be a redis.Redis

    # ------------------------------------------------------------------
    # 1. load artefacts *after* the container has started
    # ------------------------------------------------------------------
    @app.on_event("startup")
    async def _load_everything():
        # ----- Redis connection ----------------------------------------
        redis_url = os.getenv("REDIS_URL") or \
                    f"redis://:{os.getenv('REDIS_PASSWORD')}@" \
                    f"{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}"
        if not redis_url:
            raise RuntimeError("Redis connection info not provided")

        # optional: a very small retry loop
        for attempt in range(5):
            try:
                globals()["rdb"] = redis.from_url(redis_url,
                                                  decode_responses=False,
                                                  socket_connect_timeout=5)
                rdb.ping()
                break
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"Cannot connect to Redis: {e}")
                await asyncio.sleep(2)

        print("✓ connected to Redis")

        unp = lambda k: pickle.loads(rdb.get(k) or b"")   # tolerate empty key

        # ----- load pickled blobs --------------------------------------
        try:
            globals()["user_item"]   = unp("cf:user_item")
            globals()["u2i"]         = unp("cf:u2i")
            globals()["s2j"]         = unp("cf:s2j")
            globals()["j2s"]         = {v: k for k, v in s2j.items()}
            globals()["user_pprice"] = unp("cf:price_pref")
            globals()["u_brand"]     = unp("cf:brand_pref")
            globals()["u_cat"]       = unp("cf:cat_pref")
            globals()["user_norm"]   = np.frombuffer(rdb.get("cf:user_norm"),
                                                     dtype="float32")
            globals()["R"]           = sp.load_npz(io.BytesIO(rdb.get("cf:R")))
            globals()["item_co"]     = sp.load_npz(io.BytesIO(rdb.get("cf:item_cooc")))
        except Exception as e:
            raise RuntimeError(f"Missing or bad artefacts in Redis: {e}")

        # ----- small static CSV (bundled into the image) ---------------
        globals()["META"] = (pd.read_csv("/assets/skus_metadata.csv",
                                         usecols=["sku", "sale_price",
                                                  "size_availability",
                                                  "brand", "l1", "l2", "color"])
                               .set_index("sku"))
        globals()["PRICE"] = META.sale_price.to_dict()
        globals()["BRAND"] = META.brand.fillna("unknown").to_dict()
        globals()["CAT"]   = (META.l1.fillna("unknown")+"▸"+
                              META.l2.fillna("unknown")).to_dict()

        # ----- optional adhoc boosts -----------------------------------
        globals()["EXTRA_BOOST"] = {"brand": {}, "cat": {}}
        if os.path.exists("/assets/boost_table.csv"):
            boost = (pd.read_csv("/assets/boost_table.csv")
                       .dropna(subset=["boost_type", "key", "pct"]))
            EXTRA_BOOST["brand"] = dict(boost.query("boost_type=='brand'")[["key","pct"]].values)
            EXTRA_BOOST["cat"]   = dict(boost.query("boost_type=='cat'")  [["key","pct"]].values)

        print("✓  artefacts loaded – service ready")

    # ------------------------------------------------------------------
    # 2. helpers (same code as before – .A fixed)
    # ------------------------------------------------------------------
    def valid_sku(s: str) -> bool:
        return META.at[s, "size_availability"] >= .7 if s in META.index else False

    def valid_user(u: str) -> bool:
        return len(user_item.get(u, {})) >= 1

    price_of = lambda s: PRICE.get(s)
    brand_of = lambda s: BRAND.get(s)
    cat_of   = lambda s: CAT.get(s)

    def _cosine_row(i: int) -> np.ndarray:
        sims = (R.getrow(i) @ R.T).toarray().ravel()   # << .toarray() portable
        sims[i] = 0.0
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
        sc   = defaultdict(float)
        for n in sims.argsort()[-200:][::-1]:
            if sims[n] <= 0: break
            for j in R[n, :].indices:
                s = j2s[j]
                if s == target_sku or not valid_sku(s): continue
                if price_band and not price_band(s):    continue
                sc[s] += sims[n]
        return sc

    def _item_side(sku, price_band):
        if sku not in s2j: return {}
        row, out = item_co.getrow(s2j[sku]), {}
        for idx, cnt in zip(row.indices, row.data):
            s = j2s[idx]
            if not valid_sku(s): continue
            if price_band and not price_band(s): continue
            out[s] = float(cnt)
        return out

    def _recommend(user, sku, *, top_n=20, pct_band=.30, w_user=.7):
        if not valid_sku(sku):
            raise ValueError("SKU not live")
        p_ref = price_of(sku)
        if p_ref is None: return []

        if (user in user_pprice and
            user_pprice[user]["p10"] <= p_ref <= user_pprice[user]["p90"]):
            lo = user_pprice[user]["p10"]*(1-pct_band)
            hi = user_pprice[user]["p90"]*(1+pct_band)
        else:
            lo, hi = p_ref*(1-pct_band), p_ref*(1+pct_band)
        in_band = lambda s: lo <= price_of(s) <= hi if price_of(s) else False

        u_scores = _minmax(_user_side(user, sku, in_band))
        i_scores = _minmax(_item_side(sku,        in_band))

        _apply_profile_boost(user, u_scores)
        _apply_profile_boost(user, i_scores)
        _apply_adhoc_boost(u_scores)
        _apply_adhoc_boost(i_scores)

        comb = {s: (w_user*u_scores.get(s,0)+(1-w_user)*i_scores.get(s,0))
                for s in set(u_scores)|set(i_scores) if s != sku}

        if len(comb) < top_n:   # back-fill
            pool = [s for s in META.index if valid_sku(s) and s not in comb and s!=sku]
            near = heapq.nsmallest(top_n-len(comb), pool,
                                   key=lambda s: abs(price_of(s)-p_ref) if price_of(s) else 9e9)
            for s in near:
                comb[s] = (u_scores.get(s,0)+i_scores.get(s,0))

        return heapq.nlargest(top_n, comb.items(), key=lambda x:x[1])

    # ------------------------------------------------------------------
    # 3. API routes
    # ------------------------------------------------------------------
    class RecoRequest(BaseModel):
        user: str
        sku:  str
        top_n: int = Field(20, ge=1, le=100)
        pct:   float = Field(.30, ge=0., le=.9)
        w_user: float = Field(.7, ge=0., le=1.)

    class RecoItem(BaseModel):
        sku: str; score: float

    class RecoResponse(BaseModel):
        user: str; sku: str; recs: List[RecoItem]

    @app.get("/health")
    def health(): return {"status": "ok"}

    @app.post("/recommend", response_model=RecoResponse)
    def recommend_ep(req: RecoRequest):
        recs = _recommend(req.user, req.sku,
                          top_n=req.top_n, pct_band=req.pct, w_user=req.w_user)
        return {"user": req.user, "sku": req.sku,
                "recs": [{"sku": s, "score": float(sc)} for s, sc in recs]}

    return app


# ASGI callable for Uvicorn -------------------------------------------------
app = create_app()

# run with:  uvicorn reco_service:app --host 0.0.0.0 --port 8000 --workers 2
