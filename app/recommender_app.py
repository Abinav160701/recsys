# app/recommender_app.py
import os, io, pickle, heapq, redis
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# 0.  APP FACTORY
# ---------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="CF-Recommender", version="1.0")

    # ---------- Lazy-load big artefacts on *startup* ------------------
    @app.on_event("startup")
    def _load():
        nonlocal user_item, u2i, s2j, j2s, user_pprice, u_brand, u_cat
        nonlocal user_norm, R, item_co, META, PRICE, BRAND, CAT, EXTRA_BOOST

        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            raise RuntimeError("REDIS_URL env-var missing")

        rdb = redis.from_url(redis_url, decode_responses=False)
        unp  = lambda k: pickle.loads(rdb.get(k))

        user_item = unp("cf:user_item")
        u2i, s2j  = unp("cf:u2i"), unp("cf:s2j")
        j2s       = {v: k for k, v in s2j.items()}

        user_pprice = unp("cf:price_pref")
        u_brand     = unp("cf:brand_pref")
        u_cat       = unp("cf:cat_pref")

        user_norm = np.frombuffer(rdb.get("cf:user_norm"), dtype="float32")
        R         = sp.load_npz(io.BytesIO(rdb.get("cf:R")))
        item_co   = sp.load_npz(io.BytesIO(rdb.get("cf:item_cooc")))

        # small file – ok to keep in repo
        META = (pd.read_csv("skus_metadata.csv",
                            usecols=["sku","sale_price","size_availability",
                                     "brand","l1","l2","color"])
                  .set_index("sku"))
        PRICE = META.sale_price.to_dict()
        BRAND = META.brand.fillna("unknown").to_dict()
        CAT   = (META.l1.fillna("unknown")+"▸"+META.l2.fillna("unknown")).to_dict()

        # optional boost table (often tiny)
        EXTRA_BOOST = {"brand": {}, "cat": {}}
        if os.path.exists("boost_table.csv"):
            adhoc = (pd.read_csv("boost_table.csv")
                       .dropna(subset=["boost_type","key","pct"]))
            EXTRA_BOOST["brand"] = dict(adhoc.query("boost_type=='brand'")[["key","pct"]].values)
            EXTRA_BOOST["cat"]   = dict(adhoc.query("boost_type=='cat'")  [["key","pct"]].values)

        print("✓ artefacts ready – app is live")

    # ---------- request / response models & endpoint -----------------
    class RecoRequest(BaseModel):
        user   : str
        sku    : str
        top_n  : int   = Field(20, ge=1, le=100)
        pct    : float = Field(.30, ge=0., le=.9)
        w_user : float = Field(.7,  ge=0., le=1.)

    class RecoItem(BaseModel):
        sku: str; score: float

    class RecoResponse(BaseModel):
        user: str; sku: str; recs: List[RecoItem]

    @app.get("/health")
    def health(): return {"status":"ok"}

    @app.post("/recommend", response_model=RecoResponse)
    def recommend_ep(req: RecoRequest):
        try:
            recs = _recommend(req.user, req.sku,
                              top_n=req.top_n,
                              pct_band=req.pct,
                              w_user=req.w_user)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"user":req.user, "sku":req.sku,
                "recs":[{"sku":s,"score":float(sc)} for s,sc in recs]}

    # ---------- business-logic helpers (unchanged) -------------------
    # (place your previous helper functions here – left out for brevity)

    return app

# run with:  uvicorn reco_service:app --host 0.0.0.0 --port 8000 --workers 2
