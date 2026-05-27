import torch

def trajectory_loss(kkt_list, gamma=0.85):
    T = len(kkt_list)

    weights = [gamma ** t for t in range(T)]  # poids décroissant -> early iters comptent plus
    loss = sum(w * k.mean() for w, k in zip(weights, kkt_list))
    return loss