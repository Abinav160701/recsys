
### Builds the *same* artefacts as before but stores everything in Redis.

#### Key-schema
##### cf:user_item             ->  Pickle  {user -> {sku -> score}}
##### cf:R                     ->  npz bytes  (#user × #sku CSR matrix)
##### cf:user_norm             ->  np.ndarray bytes (float32, len=#user)
##### cf:item_cooc             ->  npz bytes (top-30 co-occurrence CSR)
##### cf:u2i , cf:s2j          ->  Pickle id-maps
##### cf:price_pref            ->  Pickle {user -> {"p10":..,"p90":..}}
##### cf:brand_pref            ->  Pickle {user -> [brand1,..]}
##### cf:cat_pref              ->  Pickle {user -> [cat1,..]}

# Imports

import os, io, pickle, redis, numpy as np, pandas as pd, scipy.sparse as sp
from collections import defaultdict

# Data load and preprocessing
RAW_INTER  = "user-item-dates_sum_price_cats.csv"
RAW_META   = "skus_metadata.csv"
rdb        = redis.from_url("redis://localhost:6379/0", decode_responses=False)

# 1 ▸ Load & clean
ui   = pd.read_csv(RAW_INTER, parse_dates=["max_date"])
meta = pd.read_csv(RAW_META,
                   usecols=["sku","brand","l1","l2","color","sale_price"])
for c in ["brand","l1","l2","color"]:
    meta[c] = meta[c].fillna("unknown")
for c in ["purchased","checkout","add_to_cart","pdp","plp_click"]:
    ui[c] = ui[c].fillna(0).astype("int8")    

### Interaction score

# 2 ▸ Behaviour score
REF = ui.max_date.max();  HL = 30
tier = (
      10*ui.purchased
    +  8*((ui.checkout>0)&(ui.purchased==0))
    +  6*((ui.add_to_cart>0)&(ui.checkout==0))
    +  3*((ui.pdp>0)&(ui.add_to_cart==0))
    +  3*((ui.plp_click>0)&(ui.pdp==0))
).astype("float32")
ui["score"] = tier * np.power(0.5, (REF-ui.max_date).dt.days/HL)
ui = ui[ui.score > 0]

### User-sku dict

# 3 ▸ user→sku dict
u_s_dict = defaultdict(dict)
for r in ui.itertuples(index=False):
    u_s_dict[r.userid][r.sku] = u_s_dict[r.userid].get(r.sku, 0) + r.score

# 4 ▸ Sparse matrix + co-occurrence
users,  inv_u = np.unique(ui.userid, return_inverse=True)
skus,   inv_s = np.unique(ui.sku,    return_inverse=True)
R = sp.csr_matrix((ui.score, (inv_u, inv_s)),
                  shape=(users.size, skus.size), dtype="float32")
user_norm = np.sqrt(R.power(2).sum(axis=1)).A1.astype("float32")
binary = (R > 0).astype("float32")
C = (binary.T @ binary).tocsr()

TOPK = 30
indptr, idx, dat = C.indptr, C.indices, C.data
rows, cols, vals = [], [], []
for j in range(C.shape[0]):
    st, ed = indptr[j], indptr[j+1]
    if st == ed: continue
    k = min(TOPK, ed-st)
    top = np.argpartition(-dat[st:ed], k-1)[:k]
    rows.extend([j]*k);  cols.extend(idx[st:ed][top]);  vals.extend(dat[st:ed][top])
C_top = sp.csr_matrix((vals, (rows, cols)), shape=C.shape, dtype="int32")

### Pricecat and brand preference
price_pref = (ui.groupby("userid")[["sale_price_10","sale_price_90"]]
                 .mean()
                 .rename(columns={"sale_price_10":"p10","sale_price_90":"p90"})
                 .to_dict("index"))

ui["elig"] = ui.groupby("userid").sku.transform("nunique") > 5
ui["cat"] = ui.l1.fillna("unknown")+"▸"+ui.l2.fillna("unknown")
sub = ui[ui.elig]
brand_pref = (sub.groupby(["userid","brand"])["score"].sum()
                .reset_index()
                .sort_values(["userid","score"], ascending=[True,False])
                .groupby("userid").head(3)
                .groupby("userid")["brand"].agg(list).to_dict())

cat_pref   = (sub.groupby(["userid","cat"])["score"].sum()
                .reset_index()
                .sort_values(["userid","score"], ascending=[True,False])
                .groupby("userid").head(3)
                .groupby("userid")["cat"].agg(list).to_dict())

### Push to redis

def _npz_bytes(mat: sp.csr_matrix) -> bytes:
    """helper – returns the npz‐encoding of a sparse matrix as bytes"""
    bio = io.BytesIO()
    sp.save_npz(bio, mat)
    return bio.getvalue()

pipe = rdb.pipeline()

pipe.set("cf:user_item",  pickle.dumps(dict(u_s_dict), protocol=4))
pipe.set("cf:R",          _npz_bytes(R))          # <── fixed
pipe.set("cf:user_norm",  user_norm.tobytes())
pipe.set("cf:item_cooc",  _npz_bytes(C_top))      # <── fixed
pipe.set("cf:u2i",        pickle.dumps({u:i for i,u in enumerate(users)}, 4))
pipe.set("cf:s2j",        pickle.dumps({s:j for j,s in enumerate(skus)}, 4))
pipe.set("cf:price_pref", pickle.dumps(price_pref, 4))
pipe.set("cf:brand_pref", pickle.dumps(brand_pref, 4))
pipe.set("cf:cat_pref",   pickle.dumps(cat_pref,   4))

pipe.execute()
print("✓ artefacts stored in Redis (‘cf:*’ keys)")
