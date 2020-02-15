#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Initialize the cluster centers and choose k number of clusters
def init_clusters(arr_reshaped, k):
    '''
    k is the number of clusters.
    arr_reshaped is a 2 dimensional array of the image
    Output: randomized initial clusters
    '''
    centers = arr_reshaped[np.random.randint(arr_reshaped.shape[0], size=(1, k))[0]]
    return centers


# In[ ]:


def dist_matrix(matrix, centers, order=2):
    '''
    Calculating the distance matrix
    Input: image matrix array, centers and order of norm distance
    Output: Distance matrix nxk
    '''
    n = len(matrix)
    c = len(centers)
    Dmatrix = np.empty((n,c))
    if len(centers.shape) == 1:
        centers = centers.reshape(1,3)
    
    for i in range(n):
        d_i = np.linalg.norm(matrix[i,:] - centers, ord=order, axis=1)
        Dmatrix[i, :] = np.power(d_i, order)
    return Dmatrix


# In[ ]:


def cluster_assignment(Dmatrix):
    '''
    Seperating the data points into clusters
    Input: Distance matrix
    Output: Labels of the clusters for each data point
    '''
    labels = np.argmin(Dmatrix, axis=1)
    return labels


# In[ ]:


def J_sum(Dmatrix):
    '''
    Calculating the loss function value
    Input: Distance matrix
    Output: a scalar
    '''
    return np.sum(np.amin(Dmatrix, axis=1))


# In[ ]:


def cluster_update(matrix, labels, k):
    '''
    Finding the new center means for k-means
    input: Image matrix 2D, labels, number of clusters k
    '''
    n, d = matrix.shape
    new_centers = np.empty((k,d))
    
    for i in range(k):
        new_centers[i, :] = np.mean(matrix[labels==i, :], axis=0)

        
    return new_centers
        


# In[ ]:


def center_convergence(old_centers, new_centers):
    return [list(center) for center in old_centers] == [list(center) for center in new_centers]


# In[ ]:


def kmeans(matrix, k):
    
    centers = init_clusters(matrix, k)
    converged = False
    
    while (converged != True):
            
        distance_matrix = dist_matrix(matrix,centers)
        labels = cluster_assignment(distance_matrix)
        new_centers = cluster_update(matrix, labels, k) 
        converged = center_convergence(centers , new_centers)
        #updating
        centers = new_centers
    
    
    return centers, labels, J_sum(distance_matrix)
        
        


# In[ ]:




