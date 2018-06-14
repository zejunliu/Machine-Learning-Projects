import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids or means, membership, number_of_updates )
            Note: Number of iterations is the number of time you update means other than initialization
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership untill convergence or untill you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)
        K = self.n_cluster
        mu_k = [[0 for i in range(D)] for j in range(K)] ### mu_k, Array of size K × D of means
        for k in range(K):###initialize mu_k to k points selected from x randomly
            unique_k = np.random.randint(N, size=1)
            mu_k[k] = x[unique_k[0]]
        ###set initial J to 0
        J = 0
        for iter in range(self.max_iter):
            r_ik = [[0 for i in range(K)] for j in range(N)] ### Membership vector R of size N, where R[i] = k, such that rik = 1
            for i in range(N): #find ith data's cluster
                min_distance = 2147483647
                k_a = -1
                for k in range(K): ###compare every cluster
                    distance = 0
                    distance += np.square(np.linalg.norm(np.array(mu_k[k]) - np.array(x[i])))
                    if distance < min_distance:
                        min_distance = distance
                        k_a = k ###k_a is the cluster ith data belongs to
                r_ik[i][k_a] = 1 ### ith data belongs to k_ath cluster, assign r_ik to 1
            dis = 0
            for i in range(N):#compute distortion measure
                for k in range(K):
                    dis += r_ik[i][k] * np.square(np.linalg.norm(np.array(mu_k[k]) - np.array(x[i])))    
            J_new = dis / N

            if np.abs(J-J_new) <= self.e:
                break
            else:
                J = J_new
                for k in range(K):
                    a = np.zeros(D)
                    b = 0
                    for i in range(N):
                        b += r_ik[i][k]
                        a_i = r_ik[i][k] * np.array(x[i])
                        a += a_i
                    mu_k[k] = (a/b).tolist()

        R =[0 for i in range(len(r_ik))]
        for i in range(len(r_ik)):
            for j in range(len(r_ik[0])):
                if r_ik[i][j] == 1:
                    R[i] =j
        return np.array(mu_k),np.array(R), iter
class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering
            max_iter - maximum updates for kmeans clustering
            e - error tolerance
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - N size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering
                self.centroid_labels : labels of each centroid obtained by
                    majority voting
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels
        K = self.n_cluster
        mu_k = [[0 for i in range(D)] for j in range(K)] ### mu_k, Array of size K × D of means
        for k in range(K):###initialize mu_k to k points selected from x randomly
            unique_k = np.random.randint(N, size=1)
            mu_k[k] = x[unique_k[0]]
        ###set initial J to 0
        J = 0
        for iter in range(self.max_iter):
            r_ik = [[0 for i in range(K)] for j in range(N)] ### Membership vector R of size N, where R[i] = k, such that rik = 1
            for i in range(N): #find ith data's cluster
                min_distance = 2147483647
                k_a = -1
                for k in range(K): ###compare every cluster
                    distance = 0
                    distance += np.square(np.linalg.norm(np.array(mu_k[k]) - np.array(x[i])))
                    if distance < min_distance:
                        min_distance = distance
                        k_a = k ###k_a is the cluster ith data belongs to
                r_ik[i][k_a] = 1 ### ith data belongs to k_ath cluster, assign r_ik to 1
            dis = 0
            for i in range(N):#compute distortion measure
                for k in range(K):
                    dis += r_ik[i][k] * np.square(np.linalg.norm(np.array(mu_k[k]) - np.array(x[i])))    
            J_new = dis / N

            if np.abs(J-J_new) <= self.e:
                break
            else:
                J = J_new
                for k in range(K):
                    a = np.zeros(D)
                    b = 0
                    for i in range(N):
                        b += r_ik[i][k]
                        a_i = r_ik[i][k] * np.array(x[i])
                        a += a_i
                    mu_k[k] = (a/b).tolist()
        centroids = mu_k
        centroid_labels = []
        for k in range(K):
            res=[]
            for i in range(len(r_ik)):
                if r_ik[i][k] != 0:
                    res.append(y[i])
            a = np.array(res)
            unique,counts = np.unique(a, return_counts = True)
            dic = dict(zip(unique, counts))
            max_val = -2147483647
            max_key = 0
            for key in dic.keys():
                if dic[key] > max_val:
                    max_val = dic[key]
                    max_key = key
            centroid_labels.append(max_key)
        centroids = np.asarray(centroids)
        centroid_labels = np.asarray(centroid_labels)
        
        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a vector of shape {}'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels
        K = self.n_cluster
        labels = []
        for i in range(N):
            min_dist = 2147483647
            label =0
            for j in range(K):
                dist = 0
                dist = np.linalg.norm(np.array(x[i]) - np.array(self.centroids[j]))
                if dist < min_dist:
                    min_dist = dist
                    label = self.centroid_labels[j]
            labels.append(label)
        return np.asarray(labels)
        '''
        # DONOT CHANGE CODE ABOVE THIS LINE
        raise Exception(
            'Implement predict function in KMeansClassifier class (filename: kmeans.py')
        # DONOT CHANGE CODE BELOW THIS LINE
        '''
