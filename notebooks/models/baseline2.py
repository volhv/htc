
import pandas as pd
import numpy as np
from collections import Counter 
from models.core import CartModel



# из списка кандидатов по совстречаемости удаляем повторяющиеся item_id, сохраняя порядок
def get_unique_recs(recs: list, top_n: int) -> list:
    rec_dict = {}
    counter = 0
    for k, v in recs:
        if k not in rec_dict:
            rec_dict[k] = v
            counter += 1
        if counter == top_n:
            break
    return list(rec_dict.keys())

def rec_by_item(item_id: int, most_freq_dict: dict) -> list:
    return most_freq_dict.get(item_id, None)

# для каждого item_id соберем top_n самых часто встречающихся item_id, отсортируем по частоте и выберем уникальные
def rec_by_basket(basket: list, most_freq_dict: dict, top_n: int = 20) -> list:
    
    res = []
    for item in basket:
        recs = rec_by_item(item, most_freq_dict)
        if recs is not None:
            res += [ r for r in recs if not r in basket] 
    
    res = sorted(res, key=lambda x: x[1], reverse=True)
    
    return get_unique_recs(res, top_n)


class BaselineModel(CartModel):

    def __init__(self, **kwargs):
        CartModel.__init__(self, **kwargs)
        self.top_n = self.param("top_n", 10)
        self.most_freq_dict = None
    
    def train(self, X_train):
        top_n = self.top_n
        pairs = X_train[['item_id', 'pav_order_id']]\
                .sort_values(['item_id', 'pav_order_id'])\
                .merge(X_train[['item_id', 'pav_order_id']], 
                       how='left', on=['pav_order_id'], suffixes=('', '_left'))
        
        q = pairs.item_id != pairs.item_id_left
        
        pairs['cnt'] = 1.
        xy = pairs[q][['item_id', 'item_id_left', 'cnt']]\
                    .groupby(['item_id', 'item_id_left'])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'xy'}, axis=1)\
                    .reset_index()
        
        x = pairs.groupby(["item_id"])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'x'}, axis=1)\
                    .reset_index()
        
        buyers = X_train[["buyer_id", "item_id"]].drop_duplicates()
        buyers["cnt"] = 1
        
        u = buyers.groupby(["buyer_id"])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'u'}, axis=1)\
                    .reset_index()
        
        xu = buyers.groupby(["item_id", "buyer_id"])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'xu'}, axis=1)\
                    .reset_index()
        
        
        #p(y|x) = p(xy) / p(x)
        ptrs1 = xy.merge(x, on="item_id")
        ptrs1["y|x"] = ptrs1["xy"] / ptrs1["x"]
        
        ptrs2 = xu.merge(u, on="buyer_id")
        ptrs2["x|u"] = ptrs2["xu"] / ptrs2["u"]
        
        self.ptrs1 = ptrs1[ ptrs1["y|x"] > 1e-2]
        self.ptrs2 = ptrs2[ ptrs2["x|u"] > 1e-2]
        self.ptrs2 = self.ptrs2.rename({"item_id":"item_id_left"}, axis=1)
        
#         .groupby(['item_id'])['item_id_left'].agg(
#             lambda x: Counter(x).most_common(top_n))
#         self.most_freq_dict = {k: v for (k, v) in top_n_pairs.iteritems()}
    
    
    def predict(self, X_test):
        preds = self.to_basket(X_test)
        X_test["cnt"] = 1
        support = X_test.groupby(["item_id"])[["cnt"]]\
                    .count()\
                    .reset_index()\
                    .rename({'cnt':'x1', 'item_id':'item_id_left'}, axis=1)
                    
        
        candidates = X_test.merge(self.ptrs1, on=["item_id"])
        candidates = candidates.merge(self.ptrs2, 
                        on=["item_id_left", "buyer_id"], 
                        how="left").fillna(1.)
        
        candidates = candidates.merge(support, 
                        on=["item_id_left"], 
                        how="left").fillna(support.x1.mean())
        
        candidates["y'|x"] = candidates["y|x"] * candidates["x|u"] * candidates["x1"]
        
        candidates = candidates.groupby(["item_id", "item_id_left"])["y'|x"]\
                                    .sum()\
                                    .reset_index()
        
        candidates = candidates.sort_values(by="y'|x", ascending=False)
        # print(candidates)
        
        def process(df):
            return [*zip(df.item_id_left, df["y'|x"])]
        
        most_freq_dict = candidates.groupby("item_id").apply(process).to_dict()
        
        
        preds["preds"] = preds['basket'].map(
            lambda x: rec_by_basket(x, most_freq_dict=most_freq_dict, top_n=self.top_n))
        return preds


    
