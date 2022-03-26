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
        self.linear1 = nn.Linear(embedding_dim, 1024)
        self.linear2 = nn.Linear(1024, context_size * vocab_size)
        self.context_size = context_size
        #self.parameters['context_size'] = context_size

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))  # -1 implies size inferred for that index from the size of the data
        #print(np.mean(np.mean(self.linear2.weight.data.numpy())))
        out1 = F.relu(self.linear1(embeds)) # output of first layer
        out2 = self.linear2(out1)           # output of second layer
        #print(embeds)
        log_probs = F.log_softmax(out2, dim=1).view(self.context_size,-1)
        return log_probs

    def predict(self,input):
        context_idxs = torch.tensor([input], dtype=torch.long)
        res = self.forward(context_idxs)
        res_arg = torch.argmax(res)
        res_val, res_ind = res.sort(descending=True)
        indices = [res_ind[i][0].item() for i in np.arange(0, self.context_size)]
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
    
    def train(self, X_train, n_epochs=5, n_batch=1000, n_embed=10, n_ctx=5, lr=1e-5):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        prod_to_ix = {
            itemd_id: ix for ix, itemd_id in\
                enumerate(X_train["item_id"].unique()) }
        self.ix_to_prod = {
            ix: item_id for item_id, ix in prod_to_ix.items()}
        self.prod_to_ix = prod_to_ix
        losses = []
        n_vocab = len(prod_to_ix)
        loss_function = nn.NLLLoss()
        model = SkipgramModeler(n_vocab, n_embed, n_ctx)
        self.model = model
        optimizer = optim.Adam(model.parameters(), lr=lr)

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
            ngrams = flatten([*prep_input(X_train[q], n_ctx=n_ctx).values])
            #print(ngrams[:10])
            model.zero_grad()
            
            for target, context in ngrams:
        
                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                #print(context,target)


                context_idxs = torch.tensor([prod_to_ix[context]], dtype=torch.long)
                context_idxs.to(device)
                #print("Context id",context_idxs)

                # Step 2. Recall that torch *accumulates* gradients. Before passing in a
                # new instance, you need to zero out the gradients from the old
                # instance
                

                # Step 3. Run the forward pass, getting log probabilities over next
                # words
                log_probs = model(context_idxs)
                #print(log_probs)
                #print(log_probs)
                # Step 4. Compute your loss function. (Again, Torch wants the target
                # word wrapped in a tensor)
                target_list = torch.tensor(
                    [prod_to_ix[w] for w in target], dtype=torch.long)
                loss = loss_function(log_probs, target_list)
                #print(loss)

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                total_loss += loss.item()
            print("Loss: ", total_loss)
            losses.append(total_loss)  
            self.losses = losses
        
    def predict_item(self, item_id):
        if item_id in self.cache:
            return self.cache[item_id]
        if not item_id in self.prod_to_ix:
            return []
        ix = self.prod_to_ix[item_id]
        z = self.model.predict(ix)
        rz = [self.ix_to_prod[zi] for zi in z]
        self.cache[item_id] = rz
        return rz
    
    def predict(self, X_test):
        #q = X_test.item_id.isin(self.prod_to_ix)
        #X_test = X_test.loc[q]
        preds = self.to_basket(X_test)
        with torch.inference_mode():
            preds["preds"] = preds['basket'].map(
                lambda x: rec_by_basket(x, 
                    model=self.predict_item))
        return preds


    
