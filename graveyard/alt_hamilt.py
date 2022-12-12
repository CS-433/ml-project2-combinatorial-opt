import torch.sparse

for k in range(n):
    for l in range(n):
        for i in range(n-1):
            W = G.get_edge_data(k, l)['weight']
            J = torch.sparse.zeros(size = [n, n, n, n] requires_grad=True)