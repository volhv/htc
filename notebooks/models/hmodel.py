import pandas as pd
import numpy as np
from collections import Counter 
from models.core import CartModel


class HModel(CartModel):

    def __init__(self, **kwargs):
        CartModel.__init__(self, **kwargs)
        self.seed = self.param("seed", 64)
        self.top_n = self.param("top_n", 10)
        self.shash = None
    
    def train(self, X_train):
        top_n = self.top_n
        X_train["sid"] = X_train["item_id"] % self.seed
        pairs = X_train[['item_id', 'pav_order_id', 'sid']]\
                .sort_values(['item_id', 'pav_order_id', 'sid'])\
                .merge(X_train[['item_id', 'pav_order_id', 'sid']], 
                       how='left', on=['pav_order_id'], suffixes=('', '_left'))
        
        q = pairs.sid != pairs.sid
        self.shash = pairs[q].groupby(['item_id'])['sid_left'].agg(
            lambda x: Counter(x).most_common(top_n))
        
    
    def predict(self, X_test):
        preds = self.to_basket(X_test)
        preds["preds"] = preds['basket'].map(
            lambda x: rec_by_basket(x, most_freq_dict=self.most_freq_dict))
        return preds


    

