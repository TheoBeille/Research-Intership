
#%% Run to have everything 

import torch
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath("../tgv_pycuda-master"))

sys.path.append(os.path.abspath(".."))




from Algo_setuptorch import Params
#from Algo_setup_tomo import Params
from data.dataset import build_train_test_data
#from data.dataset_tomo import build_train_test_data

from algorithm.unrolled_model import UnrolledFBS
from training.train import train
from NN.plots import *
from NN.run import *


from denoise_pycuda import tgv_denoise






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

params = Params()

TRAIN_SEEDS = list(range(50))  
TEST_SEEDS = list(range(50,60))


size = params.size
SHAPES = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH = sum(s[1] for s in SHAPES)
N_CH_primal = sum(s[1] for s in SHAPES[:2])
print("ok")

train_data, test_data = build_train_test_data(
    train_seeds=TRAIN_SEEDS,
    test_seeds=TEST_SEEDS,
    params=params,
    device=device,
)


initial_state, clean, functions = test_data[0]



model = UnrolledFBS(
    params=params,
    shapes=SHAPES,
    n_channels=N_CH_primal,
    T=10,
    alpha=0.99,
).to(device).float()



  
model, train_hist,val_loss_hist = train(
    model=model,
    train_data=train_data,
    n_epochs=200,
    lr=1e-4,
    device=device,
    print_every=5
)

torch.save(model.state_dict(), "model_final_10.pt") 



""" model.load_state_dict(torch.load("model_final_u_prev.pt", map_location="cpu"))  """

model.eval()




res_learned,F_vals_learned = run_learned(model,initial_state,clean,functions,T_test=1000)
res_zero, _= run_zero(initial_state,functions, params, SHAPES, 1000, device)

plot_convergence_2(res_zero,res_learned) 
""" 

#%%TESTING DIFFERENT e TO SEE THE CONVERGENCE RATE 

results ={}
es = [0.0, 0.3, 0.5, 0.7, 1.0]
for e in es:
    print(f"Training for e={e}")

    params = Params()
    
    # override lambda
    params.lam = lambda n, e=e: params.lam0 * (1 + n)**e

    model = UnrolledFBS(
        params=params,
        shapes=SHAPES,
        n_channels=N_CH_primal,
        T=10,
        alpha=0.99,
    ).to(device).float()

    model, train_hist,val_loss_hist = train(
        model=model,
        train_data=train_data,
        n_epochs=200,
        lr=1e-4,
        device=device,
        print_every=5
    )

    torch.save(model.state_dict(), f"model_e_{e}.pt")

    model.eval()

    res_learned,F_vals_learned = run_learned(model,initial_state,clean,functions,T_test=500)
    
    results[e] = {
        "learned": res_learned,

    }
    
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(7,5))

for e in es:
    res = np.array(results[e]["learned"])
    iters = np.arange(1, len(res)+1)

    res = np.maximum(res, 1e-12)

    plt.loglog(iters, res, label=f"learned e={e}")



iters0 = np.arange(1, len(res)+1)



# références théoriques
plt.loglog(iters0, 1/iters0, 'k:', label=r"$O(1/t)$")
plt.loglog(iters0, 1/iters0**2, 'k-.', label=r"$O(1/t^2)$")

plt.xlabel("Iterations")
plt.ylabel("Residual")
plt.title("Effect of e on convergence")
plt.legend()
plt.grid(True, which="both")

plt.savefig("test_e.pdf")

for e in es:
    res = np.array(results[e]["learned"])
    iters = np.arange(1, len(res)+1)

    slope = np.polyfit(np.log(iters[50:]), np.log(res[50:]), 1)[0]
    print(f"e={e}, slope ≈ {slope:.2f}")

 """

#%%COMPARE WITH CP






""" model.eval()
 





F_vals_learned,res_learned = run_learned(model,initial_state,clean,functions,T_test=500)


res_zero, _= run_zero(initial_state,functions, params, SHAPES, 500, device)

plot_convergence_2(res_zero,res_learned)  """




f_np = initial_state.squeeze().cpu().numpy()  # (H, W)
clean_np=clean.squeeze().cpu().numpy() 


u_tgv, F_vals = tgv_denoise(f_np,clean_np, 0.1, 0.1, 2, 1000, -1)


gaps = np.array(F_vals) - min(F_vals)
gaps_learned = np.array(F_vals_learned) - min(F_vals_learned)


gaps = gaps[gaps > 1e-12]
gaps_learned = gaps_learned[gaps_learned > 1e-12]
gaps /= gaps[0]
gaps_learned /= gaps_learned[0]
N = min(len(gaps), len(gaps_learned))
t = np.arange(1, N + 1)

plt.loglog(gaps[:N], label="PyCUDA TGV")
plt.loglog(gaps_learned[:N], label="My version TGV")


C1 = gaps[0]     
C2 = gaps[0]

ref1 = C1 / t         # O(1/t)
ref2 = C2 / (t**2)    # O(1/t^2)

plt.loglog(t, ref1, '--', label=r'$O(1/t)$')
plt.loglog(t, ref2, '--', label=r'$O(1/t^2)$') 

plt.xlabel("Iterations")
plt.ylabel("Residuals")
plt.legend()
plt.minorticks_on()  # active les sous-graduations
plt.grid(which='major', linestyle='-', linewidth=0.8)
plt.grid(which='minor', linestyle='--', linewidth=0.4, alpha=0.7)

plt.savefig("energy.pdf")
plt.show() 