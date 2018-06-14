import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k
            
            K =self.n_cluster

            pi_k = np.zeros(K)
            kmeans_obj = KMeans(self.n_cluster,self.max_iter,self.e)
            means, membership, iter_count = kmeans_obj.fit(x) ###initialize means using k-means clustering
            
            l_ik = np.zeros((N,K))###initilaize how much the component k is responsible for x.
            for i in range(N):
                l_ik[i][membership[i]] = 1
            
            ###compute initial pi_k
            N_k = np.zeros(K)   
            for k in range(K):    
                for i in range(N):
                    N_k[k] += l_ik[i][k]
            pi_k = N_k/N

            ###estimate initial variance
            variances = []
            for k in range(K):
                total_matrix = np.zeros((D,D))
                for i in range(N):
                    line = np.matrix(x[i])-np.matrix(means[k])
                    matrix =  np.transpose(line) * line
                    a = matrix * l_ik[i][k]
                    total_matrix += a
                c = total_matrix / N_k[k]
                variances.append(c)
            self.means = means
            self.variances = np.array(variances)
            self.pi_k = pi_k            
        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k
            K = self.n_cluster
            l_ik = np.zeros((N,K))
            pi_k_random = np.empty(K)
            pi_k_random.fill(1/K)###Initialize pi k to be uniform 1/K
            
            variances_random = []
            for k in range(K):###set the covariance matrix Σk = I
                variances_random.append(np.identity(D))

            means_random = []
            for k in range(K):###generate μk randomly (each element uniformly in [0, 1] ) 
                line = np.random.uniform(0,1,D)
                means_random.append(line)
            
            self.pi_k = pi_k_random
            self.variances = np.array(variances_random)
            self.means = np.array(means_random)
            for i in range(N): ###E Step: Compute responsibilities 
                p_sum = 0
                for k in range(K):
                    p = 0
                    line = np.array(x[i])-np.array(self.means[k])
                    p = np.exp(-0.5*line.T.dot(np.linalg.inv(np.array(self.variances[k]))).dot(line))
                    p *= (1/(((2*np.pi)**(D/2))*(np.linalg.det(self.variances[k])**(1/2))))    
                    p_sum += p*self.pi_k[k]

                for k_i in range(K):
                    p_i = 0
                    res = 0
                    line = np.array(x[i])-np.array(self.means[k_i])
                    p_i = np.exp(-0.5*line.T.dot(np.linalg.inv(np.array(self.variances[k_i]))).dot(line))
                    p_i *= (1/(((2*np.pi)**(D/2))*(np.linalg.det(self.variances[k_i])**(1/2))))
                    p_i *= self.pi_k[k_i]
                    res = p_i/p_sum
                    l_ik[i][k_i]=res
        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity
        
        best_likelihood =0
        best_likelihood = self.compute_log_likelihood(x) ###compute the initial likelihood 
        K = self.n_cluster
        for iter in range(self.max_iter):###repeat until maximum iteration is reached
            for i in range(N): ###E Step: Compute responsibilities 
                p_sum = 0
                for k in range(K):
                    p = 0
                    line = np.array(x[i])-np.array(self.means[k])
                    p = np.exp(-0.5*line.T.dot(np.linalg.inv(np.array(self.variances[k]))).dot(line))
                    p *= (1/(((2*np.pi)**(D/2))*(np.linalg.det(self.variances[k])**(1/2))))    
                    p_sum += p*self.pi_k[k]

                for k_i in range(K):
                    p_i = 0
                    res = 0
                    line = np.array(x[i])-np.array(self.means[k_i])
                    p_i = np.exp(-0.5*line.T.dot(np.linalg.inv(np.array(self.variances[k_i]))).dot(line))
                    p_i *= (1/(((2*np.pi)**(D/2))*(np.linalg.det(self.variances[k_i])**(1/2))))
                    p_i *= self.pi_k[k_i]
                    res = p_i/p_sum
                    l_ik[i][k_i]=res
            ###compute initial pi_k
            N_k = np.zeros(K)   
            for k in range(K):    
                for i in range(N):
                    N_k[k] += l_ik[i][k]
            self.pi_k = N_k/N

            ###estimate means
            for k in range(K):
                line =np.zeros(D)
                for i in range(N):
                    line += l_ik[i][k] * np.array(x[i])
                self.means[k] = line/N_k[k]

            ###estimate variance
            variances = []
            for k in range(K):
                total_matrix = np.zeros((D,D))
                for i in range(N):
                    line = np.matrix(x[i])-np.matrix(self.means[k])
                    matrix =  np.transpose(line) * line
                    a = matrix * l_ik[i][k]
                    total_matrix += a
                c = total_matrix / N_k[k]
                variances.append(c)
            self.variances = np.array(variances)

            l_new = self.compute_log_likelihood(x)

            if np.abs(best_likelihood - l_new) <= self.e:
                break
            else:
                best_likelihood = l_new
        return iter
        
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples
        num = self.n_cluster
        res = []
        for i in range(N): ###generate N sample
            k = np.random.choice(num,1,p=self.pi_k)[0]
            data = np.random.multivariate_normal(np.array(self.means[k]), np.array(self.variances[k]), 1)
            res.append(data[0])
        return np.array(res)

        
    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        N, D = x.shape
        likelihood =0.0
        for i in range(N):
            p_sum = 0
            for k in range(self.n_cluster):
                p = 0
                line = np.array(x[i])-np.array(self.means[k])
                while np.linalg.det(self.variances[k]) == 0:
                    a = self.variances[k]
                    b = 0.001*np.identity(D)
                    a += b
                    self.variances[k] = a

                p = np.exp(-0.5*line.T.dot(np.linalg.inv(self.variances[k])).dot(line))
                p *= (1/(((2*np.pi)**(D/2))*(np.linalg.det(self.variances[k])**(1/2))))
                p *= self.pi_k[k]
                p_sum+= p
            likelihood += np.log(p_sum)
        return float(likelihood)
        