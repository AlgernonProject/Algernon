import numpy as np
from sklearn.cluster import KMeans
import itertools
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

def create_estimators(estimators, number):
    if not estimators:
        return None
    
    i = itertools.cycle(estimators)
    clfs = []
    for _ in range(number):
        clfs.append(clone(next(i)))
    return clfs

class MEClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, 
                 base_estimators=[MLPClassifier()], 
                 n_estimators=1, 
                 gaussian_threshold=0,
                 cov_factor=2,
                 random_state=None):
    
        self._params = []
        self.base_estimators = base_estimators 
        self.n_estimators = n_estimators
        self.gaussian_threshold = gaussian_threshold
        self.cov_factor = cov_factor
        self.random_state = random_state  
        self._estimators = self._generate_estimators()
        self._km = KMeans(n_clusters=self.n_estimators, random_state=self.random_state)
        
    def _generate_estimators(self):
        """Sequentially generates a list of models using base_estimators

        Returns
        -------
        list
            list of models, ideally built using sklearn API
        """
        i = itertools.cycle(self.base_estimators)
        clfs = []
        for _ in range(self.n_estimators):
            clfs.append(clone(next(i)))
        return clfs

    def __str__(self):
        return f'MEClassifier(estimators={self._estimators},  \
                gt={self.gaussian_threshold}, random_state={self.random_state})'

    def _generate_params(self, X):
        """Generates parameters for each gaussian given each cluster found in k-Means

        Parameters
        ----------
        X : np.array
            samples used to generate the parameters

        Returns
        -------
        (center:np.array, inv_cov:np.array)
            Tuple of parameters used by the gaussian kernels, which are the mean and 
            inverse of the covariance matrix
        """
        self._km.fit(X)
        for i, center in enumerate(self._km.cluster_centers_):
            cluster_samples = X[self._km.labels_ == i]
            cluster_cov = np.cov(cluster_samples.T, bias=True)
            cluster_cov += np.eye(cluster_cov.shape[1]) * self.cov_factor
            inv_cov = np.linalg.inv(cluster_cov)
            self._params.append((center, inv_cov))


    def _mvpdf_single(self, x, center, inv_cov):
        """Estimates a simplified version of multivariate normal probability density

        Parameters
        ----------
        x : np.array
            input data/sample to be estimated
        center : np.array
            mean parameter used to estimate the normal distribution
        inv_cov : np.array
            inverse of the covariance matrix

        Returns
        -------
        float
            activation value of the gaussian kernel
        """
        o = 1
        p = np.exp( -.5 * ( np.dot( (x - center).T, inv_cov ).dot( x-center ) ))

        return o * p

    def _mvpdf(self, X, center, inv_cov):
        """Applies gaussian kernel to all samples

        Parameters
        ----------
        X : np.array
            array of samples (must be a 2D matrix)
        center : np.array
            mean value for the gaussian kernel
        inv_cov : np.array
            inverse of the covariance matrix used by the gaussian kernel

        Returns
        -------
        np.array
            gaussian kernel activation for all samples passed in X
        """
        return np.apply_along_axis(self._mvpdf_single, 1, X, center, inv_cov)

    def _softmax(self, Z):
        exps = np.exp(Z - np.max(Z))
        return exps / exps.sum()

    def _normalize(self, Z):
        return Z / Z.sum()

    def _distribute_train(self, X, y):
        """Performs distribution of samples given kernel activations.

        Parameters
        ----------
        X : np.array
            input data samples matrix
        y : np.array
            label/class vector

        Returns
        -------
        list
            returns a list of dictionaries containing training data and samples (X,Y) for each expert
        """
        dist = []

        all_rel_idx = [] # Relevant samples indexes list
        unselected_idx = [] # Samples that weren't distributed to any expert
        pdfs = []

        self._generate_params(X)

        for (center, inv_cov) in self._params:
            
            pdf = self._mvpdf(X, center, inv_cov)
            
            pdfs.append(pdf)
            rel_idx = np.argwhere(pdf >= self.gaussian_threshold).flatten()

            all_rel_idx.append(rel_idx)

            X_rel = np.take(X, rel_idx, axis=0)
            y_rel = np.take(y, rel_idx, axis=0)

            dist.append({"X" : X_rel, "y" : y_rel})

        P = np.vstack(pdfs).T #group activations altogether in a single matrix

        all_rel_idx = np.unique(np.concatenate([arr for arr in all_rel_idx]))
        unselected_idx = np.setdiff1d(np.arange(len(y)), all_rel_idx)
        
        unselected_P = np.take(P, unselected_idx, axis=0)
        
        #use the cluster/expert with the highest value for unselected samples
        unselected_dist_idx = np.argmax(unselected_P, axis=1) 
        
        #append unselected samples to their corresponding expert
        for i, _ in enumerate(dist):
            uns_X = X[np.argwhere(unselected_dist_idx == i).flatten()]
            uns_y = y[np.argwhere(unselected_dist_idx == i).flatten()]
            dist[i]["X"] = np.concatenate((dist[i]["X"], uns_X))
            dist[i]["y"] = np.concatenate((dist[i]["y"], uns_y))
        
        self._dist = dist
        return dist



    def _compute_g(self, X, norm_mode="softmax", is_predict=False):
        """Computes the g(.) factor for each sample and expert.
           This can be interpreted as the output of the gating
           network. The g(x,i) value defines how important is the 
           influence of expert i on sample x.

        Parameters
        ----------
        X : np.array
            input data samples matrix
        norm_mode : str, optional
            normalization method, by default "softmax"
        is_predict : bool, optional
            whether g(.) is being calculated for training or inference phase, 
            by default False

        Returns
        -------
        np.array
            matrix of g(.) factors for each sample/expert.
        """
        if not self._params:
            self._generate_params(X)
        
        pdfs = [] 

        for (center, inv_cov) in self._params:
            pdfs.append(self._mvpdf(X, center, inv_cov))
        
        P = np.array(pdfs).T

        # Experts whose gaussian outputs were lesser than threshold must not
        # influence in prediction.
        if is_predict:
            P[P < self.gaussian_threshold] = 0

        if norm_mode == "softmax":
            return np.apply_along_axis(self._softmax, 0, P)
        return np.apply_along_axis(self._normalize, 0, P)
      
    
    def fit(self, X, y):  
        self._classes = np.unique(y)

        for estimator, dist in zip(self._estimators, self._distribute_train(X,y)):

            if len(np.unique(dist["y"])) == 1: # if just one class appears in training set
                estimator.classes_ = np.unique(dist["y"])
                continue

            estimator.fit(dist["X"],dist["y"])

    def predict(self, X, mode="softmax"):
        """Predicts target values for X using matrix G

        Parameters
        ----------
        X : np.array
            input data samples matrix
        mode : str, optional
            normalization method of G, by default "softmax"

        Returns
        -------
        np.array
            prediction vector containing inference values for all samples
        """
        G = self._compute_g(X, mode, is_predict=True)

        probas_list = []

        for i, estimator in enumerate(self._estimators):
            
            proba_m = np.zeros((X.shape[0], self._classes.shape[0]))
            
            clf_classes = estimator.classes_.astype(int) - 1
            
            if len(clf_classes) > 1:
                clf_probas = estimator.predict_proba(X)
                proba_m[:, clf_classes] = clf_probas
            else:
                clf_probas = np.zeros((X.shape[0],self._classes.shape[0]))
                clf_probas[:, clf_classes] = 1
                proba_m = clf_probas
                
            probas_list.append(proba_m)
        
        probas = np.array(probas_list)

        y = np.zeros((probas.shape[1], probas.shape[2]))

        for i in range(0, probas.shape[0]):
            y += probas[i] * G[:, i, None]
        predictions = np.argmax(y, axis=1) + 1

        return predictions
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    