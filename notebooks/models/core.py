import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# метрики оцениваются для вектора релевантности. пример:
# реальные item_id, которые приобрел покупатель: [1 ,4, 5, 69]
# рекомендованные алгоритмом item_id: [4, 6, 7, 8, 1, 2, 67, 90]
# тогда вектор релеватности будет выглядеть следующим образом: [1, 0, 0, 0, 1, 0, 0, 0]
# и уже по не му будет расчитываться ndcg
def dcg(y_relevance):
    return np.sum([(2**i - 1) / np.log2(k + 1) for (k, i) in enumerate(y_relevance, start=1)])

def ndcg(y_relevance, k):
    if y_relevance.sum() == 0:
        return 0.0
    DCG = dcg(y_relevance[:k])
    IDCG = dcg(-np.sort(-y_relevance)[:k])
    return DCG / IDCG

def apply_relevance(x):
    return [int(item in x['basket']) for item in x['preds']]

def create_relevance(pred):
    d = pred.copy()
    d['basket'] = d['basket'].apply(set)
    d = d.apply(apply_relevance, axis=1)
    return d

def ndcg_full_dataset(d):
    dd = pd.DataFrame(d.to_list()).fillna(0).to_numpy()
    k = dd.shape[1]
    scores = [ndcg(dd[i], k) for i in range(len(dd))]
    return np.mean(scores)

def compute_ndcg_score(pred):
    relevance = create_relevance(pred)
    return ndcg_full_dataset(relevance)




def split_data(data, test_size=0.3):
    orders_sort = data[['pav_order_id', 'created']].drop_duplicates().sort_values(by=['created', 'pav_order_id'])
    train_orders, test_orders = train_test_split(orders_sort['pav_order_id'].tolist(), test_size=test_size, shuffle=False)
    train_orders, test_orders = set(train_orders), set(test_orders)
    train = data[data['pav_order_id'].apply(lambda x: x in train_orders)]
    test = data[data['pav_order_id'].apply(lambda x: x in test_orders)]
    return train, test, orders_sort, train_orders, test_orders






class CartModel(object):
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def param(self, name, default=None):
        if name in self.kwargs: 
            return self.kwargs[name]
        else: 
            return default
    
    def train(self, X_train):
        raise RuntimeException("Not Implemented")
    
    def predict(self, X_test):
        raise RuntimeException("Not Implemented")
    
    
    def to_basket(self, X_test):
        basket = X_test.groupby(['buyer_id', 'pav_order_id'])['item_id'].agg([('basket', list)])
        return basket
    
    def quality(self, X_val):
        basket = self.predict(X_val)
        score = compute_ndcg_score(basket)
        return score