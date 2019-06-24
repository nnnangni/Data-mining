## 데이터마이닝

#### Cluster

<hr>

#### Partitional Algorithms

<hr>

#### K-means Clustering Algorithm

- 처음에 랜덤하게 그룹화
- 어떻게 랜덤하게 partition하고 시작하냐에 따라 찾아지는 clustering결과가 달라짐
- 단점
  - 클러스터의 사이즈가 크거나 작을 경우에 잘 못찾음
  - 평균점으로부터 공같은 클러스트만 잘 찾을 수 있다
  - 이상치가 있게 되면 평균점을 계산했을 때 실제 데이터가 없는 쪽의 평균점이 계산됨
- K-Medoids
  - 실제 있는 포인터 중에 centrally located 된 점을 선택함

<hr>

#### Agglomerative Hierarchical Clustering Algorithms

```python
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("cluster2.csv")
%matplotlib inline
X=df.values
from sklearn.cluster import AgglomerativeClustering
for i, linkage in enumerate(('single','complete','average','ward')):
    for j in range(2,5):
        clustering = AgglomerativeClustering(
        linkage = linkage,n_clusters =j )
        y_pred = clustering.fit_predict(X)
        plt.figure(i+1, figsize =(5,5)) #그래프 사이즈
        plt.scatter(X[:,0] , X[:,1],c=y_pred, s=4 ) # XY좌표 정해주기
        plt.title(linkage)
        plt.show()
```



<hr>

#### DBSCAN  Clustering Algorithms

- 모수 2개 필요, 점하나를 기준으로 주변에 어디까지 볼 것인지
- 원 안의 개수가 충분한가 보기
- 빽빽한 부분만 타고 지나가며 cluster 하면 안됨

```python
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=20)
y_pred = dbscan.fit_predict(X)
print(y_pred[:10])
```

```python
for i, (eps, min_samples) in enumerate(((0.05, 20), (0.06, 20), (0.06, 15), (0.06, 6))):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = dbscan.fit_predict(X)
    plt.figure(i+1, figsize=(5,5))
    plt.scatter(X[:,0],X[:,1],s=4,c=y_pred)
    plt.title("eps:{}, min_samples:{}".format(eps,min_samples))
plt.show()
```

