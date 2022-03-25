
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
            res += recs
    
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
        top_n_pairs = pairs[q].groupby(['item_id'])['item_id_left'].agg(
            lambda x: Counter(x).most_common(top_n))
        self.most_freq_dict = {k: v for (k, v) in top_n_pairs.iteritems()}
    
    
    def predict(self, X_test):
        preds = self.to_basket(X_test)
        preds["preds"] = preds['basket'].map(
            lambda x: rec_by_basket(x, most_freq_dict=self.most_freq_dict))
        return preds


    
