
import pandas as pd
import numpy as np
from collections import Counter 
from models.core import CartModel

def softmax(x):
    w = np.exp(x)
    rz = w / np.sum(w)
    return rz


import pandas as pd
import numpy as np
from collections import Counter 

def softmax(x):
    w = np.exp(x)
    rz = w / np.sum(w)
    return rz

class LazyBayesModel(CartModel):

    def __init__(self, **kwargs):
        CartModel.__init__(self, **kwargs)
        self.top_n = self.param("top_n", 10)
        self.most_freq_dict = None
        self.eps = self.param("eps", 1e-5)
    
    def train(self, X_train, C0 = 50, C1 = 2):
        top_n = self.top_n
        pairs = X_train[['item_id', 'pav_order_id']]\
                .sort_values(['item_id', 'pav_order_id'])\
                .merge(X_train[['item_id', 'pav_order_id']], 
                       how='left', on=['pav_order_id'], suffixes=('', '_left'))
        
        q = pairs.item_id != pairs.item_id_left
        
        # совстречаемость товаров c(xy)
        pairs['cnt'] = 1.
        xy = pairs[q][['item_id', 'item_id_left', 'cnt']]\
                    .groupby(['item_id', 'item_id_left'])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'xy'}, axis=1)\
                    .reset_index()
        
        q = xy['xy'] >= C0
        xy = xy.loc[q]
        
        # частотность товара в парах c(x)
        x = pairs.groupby(["item_id"])[["cnt"]]\
                    .count()\
                    .rename({'cnt':'x'}, axis=1)\
                    .reset_index()
        
        # если купили товар X, грубо оцениваем вероятность
        # C - shrinkage term
        # p(y|x) = с(xy) / с(x)
        ptrs1 = xy.merge(x, on="item_id")
        ptrs1["y|x"] = ptrs1["xy"] / (ptrs1["x"])
        
        # buyers = X_train[["buyer_id", "item_id", "count"]]#.drop_duplicates()
        # buyers = np.exp(-(buyers['count'].max() - buyers['count']) / (buyers['count'].mean()))
        
        buyers = X_train[["buyer_id", "item_id"]] #.drop_duplicates()
        buyers["count"] = 1
        u = buyers.groupby(["buyer_id"])[["count"]]\
                    .count()\
                    .rename({'count':'u'}, axis=1)\
                    .reset_index()
        
        xu = buyers.groupby(["item_id", "buyer_id"])[["count"]]\
                    .count()\
                    .rename({'count':'xu'}, axis=1)\
                    .reset_index()
        
        q = xu['xu'] >= C1
        ptrs2 = xu.merge(u, on="buyer_id")
        ptrs2["x|u"] = ptrs2["xu"] / (ptrs2["u"])
        
        self.ptrs1 = ptrs1[ ptrs1["y|x"] >= self.eps]
        self.ptrs2 = ptrs2[ ptrs2["x|u"] >= self.eps]
        self.ptrs2 = self.ptrs2.rename({"item_id":"item_id_left"}, axis=1)
    
    def predict(self, X_test):
        # add "recently_bought" feature
        preds = self.to_basket(X_test)
        X_test["cnt"] = 1
        # support = X_test.groupby(["item_id"])[["cnt"]]\
        #             .count()\
        #             .reset_index()\
        #             .rename({'cnt':'x1', 'item_id':'item_id_left'}, axis=1)
        
        basket_items = X_test[["buyer_id", "pav_order_id", "item_id"]]
        basket_recs = basket_items.merge(self.ptrs1, on=["item_id"])
        basket_recs1 = basket_recs.groupby(
            ["buyer_id", "pav_order_id", "item_id_left"], as_index=False)[["y|x"]].sum()
        
        user_recs = basket_items.merge(self.ptrs2, on=["buyer_id"])
        basket_recs2 = user_recs.groupby(
            ["buyer_id", "pav_order_id", "item_id_left"], as_index=False)[["x|u"]].sum()
        
        idx = preds.reset_index()
        idx = idx[["buyer_id", "pav_order_id"]]
        relevance = idx.merge(basket_recs1, on=["buyer_id", "pav_order_id"], how="left")\
                       .merge(basket_recs2, on=["buyer_id", "pav_order_id", "item_id_left"], how="left")\
                       .fillna(self.eps)
        
        
        relevance["score"] = relevance["y|x"] + relevance["x|u"]
        relevance = relevance.merge(preds, on=["buyer_id", "pav_order_id"])
        def select(df):
            q = ~df.item_id_left.isin(df.basket)
            df = df[q].sort_values(by="score", ascending=False).head(self.top_n)
            return [*df.item_id_left]
        
        predicted = relevance.groupby(["buyer_id", "pav_order_id"]).apply(select)
        preds["preds"] = predicted
        
        preds = preds.sort_values(by="pav_order_id")
        
        return preds


    


