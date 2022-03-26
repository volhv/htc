import random; random.seed(2019)
import pandas as pd
import numpy as np
from collections import Counter 
from models.core import CartModel

import torch
from torch import optim, nn
from torch.nn import functional as F

def flatten(t):
    return [item for sublist in t for item in sublist]

def sample2(src, n_smpl, n_ctx):
    max_size = len(src)
    vals = [*src]
    result = []
    dups = {}
    for i in range(n_smpl):
        if max_size - n_ctx - 1 <= 1:
            break
        j = random.randint(1, max_size - n_ctx -1)
        if j in dups:
            continue
        dups[j]=1
        context = vals[j-1]
        target = vals[j:j+n_ctx]
        result += [(target, context)]
    return result


def prep_input(X_train, n_ctx):
    pairs = X_train[['pav_order_id', 'item_id']]\
                .groupby('pav_order_id')\
                .agg(set)
    x = pairs['item_id'].apply(lambda x: sample2(x, n_smpl=5, n_ctx=n_ctx))
    return x
    


# из списка кандидатов по совстречаемости удаляем повторяющиеся item_id, сохраняя порядок
def get_unique_recs(recs: list, top_n: int) -> list:
    rec_dict = {}
    counter = 0
    for k in recs:
        if k not in rec_dict:
            rec_dict[k] = 1
            counter += 1
        if counter == top_n:
            break
    return list(rec_dict.keys())

def rec_by_item(item_id: int, model) -> list:
    return model(item_id)

# для каждого item_id соберем top_n самых часто встречающихся item_id, отсортируем по частоте и выберем уникальные
def rec_by_basket(basket: list, model, top_n: int = 20) -> list:
    
    res = []
    for item in basket:
        recs = rec_by_item(item, model)
        if recs is not None:
            res += recs
    
    # res = sorted(res, key=lambda x: x[1], reverse=True)
    
    return get_unique_recs(res, top_n)



class SkipgramModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(SkipgramModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 32)
        self.linear2 = nn.Linear(32, vocab_size)
        self.context_size = context_size
        #self.parameters['context_size'] = context_size

    def forward(self, inputs):
        embeds = self.embeddings(inputs)\
                    .mean(axis=0).view(1, -1)  # -1 implies size inferred for that index from the size of the data
        #print(np.mean(np.mean(self.linear2.weight.data.numpy())))
        out1 = F.relu(self.linear1(embeds)) # output of first layer
        out2 = self.linear2(out1)           # output of second layer
        #print(embeds)
        log_probs = F.log_softmax(out2, dim=1).view(1,-1)
        return log_probs

    def predict(self,context_idxs):
        res = self.forward(context_idxs)
        res_val, res_ind = res.sort(descending=True)
        indices = [res_ind[i].item() for i in np.arange(0, self.context_size)]
        return indices


    def freeze_layer(self,layer):
        for name,child in model.named_children():
            print(name,child)
            if(name == layer):
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())
                    params.requires_grad= False

    def print_layer_parameters(self):
        for name,child in model.named_children():
                print(name,child)
                for names,params in child.named_parameters():
                    print(names,params)
                    print(params.size())

    def write_embedding_to_file(self,filename):
        for i in self.embeddings.parameters():
            weights = i.data.numpy()
        np.save(filename,weights)
        




class NNModel(CartModel):

    def __init__(self, **kwargs):
        CartModel.__init__(self, **kwargs)
        self.top_n = self.param("top_n", 10)
        self.most_freq_dict = None
        self.prod_to_ix = None
        self.ix_to_prod = None
        self.model = None
        self.cache = dict()
   
    def train_nn(self, X_train, n_epochs, n_batch, lr):
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_function = nn.NLLLoss()
        for epoch in range(n_epochs):
            total_loss = 0
            #------- Embedding layers are trained as well here ----#
            #lookup_tensor = torch.tensor([word_to_ix["poor"]], dtype=torch.long)
            #hello_embed = model.embeddings(lookup_tensor)
            #print(hello_embed)
            # -----------------------------------------------------#
            
            print("Epoch: ", epoch)

            carts = X_train.pav_order_id.drop_duplicates().sample(n_batch).values
            q = X_train.pav_order_id.isin(carts)
            ngrams = flatten([*prep_input(X_train[q], n_ctx=self.n_ctx).values])
            #print(ngrams[:10])
            self.model.zero_grad()
            
            for context, target in ngrams:
        
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                #print(context,target)


                context_idxs = torch.tensor([self.prod_to_ix[c] for c in context], dtype=torch.long)
                context_idxs.to(self.device)
                #print("Context id",context_idxs)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = self.model(context_idxs)
                #print(log_probs)
                #print(log_probs)
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                target_list = torch.tensor([self.prod_to_ix[target]], dtype=torch.long)
                loss = loss_function(log_probs, target_list)
                #print(loss)

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                self.optimizer.step()
                
                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            print("Loss: ", total_loss) 
            if total_loss < self.min_loss:
                self.min_loss = total_loss
    
    
    def train(self, X_train, n_epochs=5, n_batch=15000, n_embed=8, n_ctx=3, lr=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_loss = np.inf
        prod_to_ix = {
            itemd_id: ix for ix, itemd_id in\
                enumerate(X_train["item_id"].unique()) }
        self.ix_to_prod = {
            ix: item_id for item_id, ix in prod_to_ix.items()}
        self.prod_to_ix = prod_to_ix
        losses = []
        n_vocab = len(prod_to_ix)
        model = SkipgramModeler(n_vocab, n_embed, n_ctx)
        model.to(self.device)
        self.model = model
        self.n_ctx = n_ctx
        
        self.train_nn(X_train, n_epochs, n_batch, lr=lr)
         
        
    def predict_item(self, basket):
        zz = []
        for ctx, target in sample2(basket, 5, 3):
            print(ctx)
            context_idxs = torch.tensor([ctx], dtype=torch.long)
            context_idxs.to("cuda")
            zi = self.model.predict(context_idxs)
            zz += [zi]
        return zz
    
    def predict(self, X_test):
        #q = X_test.item_id.isin(self.prod_to_ix)
        #X_test = X_test.loc[q]
        preds = self.to_basket(X_test)
        with torch.inference_mode():
            preds["preds"] = preds['basket'].map(
                lambda x: rec_by_basket(x, 
                    model=self.predict_item))
        return preds


    
