from hybrid_similarities.new_hybrid import HybridRS
from loader.loader import train_data, test_data, tracks_data, target_data
from utils.auxUtils import Evaluator
import pandas as pd
from mail_notification.notify import NotifyMail

# so far best hybrid with pureSVD = alpha=0.3 beta=10 gamma=1 eta=10

r = HybridRS(tracks_data)
e = Evaluator()

r.fit(train_data)
gammas = [1]
alphas = [0.3]
betas = [10]
etas = [10]
thetas = [1, 10, 20, 50, 100]
list_res = []

gamma = 1

for gamma in gammas:
    for alpha in alphas:
        for beta in betas:
            for eta in etas:
                for theta in thetas:
                    pred = r.recommend(target_data['playlist_id'], alpha=alpha, beta=beta, gamma=gamma, eta=eta,
                                       theta=theta)
                    temp_map = e.evaluate(pred, test_data)
                    # print(pred[:10])
                    print(alpha, beta, gamma, eta)
                    list_res.append({'map': temp_map, 'alpha': alpha, 'beta': beta, 'gamma': gamma, 'eta': eta})

print(list_res)
df = pd.DataFrame(list_res)
df.to_csv('result.csv', '\t')


notify = NotifyMail(to_address='arto', dataframe=df)
notify.send_email()

