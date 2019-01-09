from hybrid_similarities.new_hybrid import HybridRS
from loader.loader import train_data, test_data, tracks_data, target_data, full_data, save_dataframe
from utils.auxUtils import Evaluator, submit_dataframe_to_kaggle
import pandas as pd
from mail_notification.notify import NotifyMail

# so far best hybrid with pureSVD = alpha=0.3 beta=10 gamma=1 eta=10

r = HybridRS(tracks_data)
e = Evaluator()

r.fit(train_data)
# content filter
gammas = [0, 0.8, 1]
# collaborative user user
alphas = [0, 0.2, 0.3]
# collaborative item item
betas = [0, 10]
# pureSVD
etas = [0, 10]
# graph based
thetas = [0, 20, 30]
# slim BPR
deltas = [0, 0.8, 1]
# slim EN
omegas = [0, 10, 30]
list_res = []
# 0.2 10 1.0 10 1 40.0 30
sigmas = [0, 20]
for gamma in gammas:
    for alpha in alphas:
        for beta in betas:
            for eta in etas:
                for theta in thetas:
                    for delta in deltas:
                        for omega in omegas:
                            for sigma in sigmas:
                                pred = r.recommend(target_data['playlist_id'], alpha=alpha, beta=beta,
                                                   gamma=gamma, eta=eta, theta=theta, delta=delta, omega=omega,
                                                   sigma=sigma)
                                temp_map = e.evaluate(pred, test_data)
                                # print(pred[:10])
                                print("alpha: ", alpha,
                                      "beta: ", beta,
                                      "gamma: ", gamma,
                                      "eta: ", eta,
                                      "theta: ", theta,
                                      "delta: ", delta,
                                      "omega: ", omega,
                                      "sigma: ", sigma)
                                list_res.append({'map': temp_map, 'alpha': alpha, 'beta': beta, 'gamma': gamma,
                                                 'eta': eta, 'theta': theta, 'delta': delta, 'omega': omega,
                                                 'sigma': sigma})


print(list_res)
df = pd.DataFrame(list_res).sort_values(by='map')
df = df.sort_values(by='map')
df.to_csv('result.csv', '\t')

notify = NotifyMail(to_address='arto', dataframe=df)
notify.send_email()

# save_dataframe('output/submission_hybrid', ',', pred)
# submit_dataframe_to_kaggle('output/submission_hybrid', '0.2 10 1.0 10 1 40.0 30 alpha  beta  delta  eta  gamma omega  theta')

