import tslearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from tslearn.clustering import TimeSeriesKMeans
from tslearn import metrics

from scipy.signal import hilbert, butter, filtfilt
from scipy.fftpack import fft,fftfreq,rfft,irfft,ifft
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
seed = 42
np.random.seed(seed)
df = pd.read_csv('train_data5.csv',index_col=0)
df.head(10)

fig = plt.plot(StandardScaler().fit_transform(df))
# axs.plot(len(df))
plt.show()

from tslearn.clustering.utils import silhouette_score
elbow_data = []
silhouette_data = []
for n_clusters in range (1,2):
  print("DBA k-means")
  dba_km = TimeSeriesKMeans(n_clusters=n_clusters,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
  y_pred = dba_km.fit_predict(df)
  silhouette_avg=silhouette_score(df,y_pred)
  silhouette_data.append((n_clusters,silhouette_avg))
  elbow_data.append((n_clusters,dba_km.inertia_))
  print(f'Inertia: {dba_km.inertia_}')
  print(f'cluster_center {dba_km.cluster_centers_}')

result = pd.DataFrame(elbow_data,columns=['clusters', 'distance']).plot(x='clusters',y='distance')
result2 = pd.DataFrame(silhouette_data, columns=["n_clusters","silhouette_score"])

pivot_km = pd.pivot_table(result2, index="n_clusters", values="silhouette_score")

plt.figure()
sns.heatmap(pivot_km, annot=True, linewidths=.5, fmt='.3f', cmap=sns.cm._rocket_lut)
plt.tight_layout()
plt.show()