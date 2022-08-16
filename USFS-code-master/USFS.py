import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from numpy import genfromtxt

def mydist(X,Y):
    d = np.sqrt(sum((X-Y)**2))
    return d


def ClusteringMeasure(Y, predY):

    if Y.shape[1] != 1: 
        Y = Y.T
    if np.ndim(predY) != 1: # np.shape(predY)[1] != 1:
        predY = predY.T
    elif predY.shape[0] > 0:
        predY = predY.T
    
    n = Y.shape[1] # len(Y)
    uY = np.unique(Y)
    nclass = len(uY)
    Y0 = np.zeros((n,1))
    if nclass != np.max(Y)+1:
        for i in range(nclass):
            Y0[np.where(Y == uY[i])] = i + 1
        Y = Y0
    
    uY = np.unique(predY)
    nclass = len(uY)
    predY0 = np.zeros([n])
    if nclass != np.max(predY)+1:
        for i in range(nclass):
            predY0[np.where(predY[i] == uY[i])] = i + 1
        predY = predY0

    # Lidx = np.unique(Y)
    # classnum = len(Lidx)
    predLidx = np.unique(predY)
    pred_classnum = len(predLidx)

    #purity
    correnum = 0
    for ci in range(pred_classnum):
        incluster = Y[np.where(predY == predLidx[ci])]
        inclunub = np.histogram(incluster, [np.arange(1,np.max(incluster)+1,1)][0])[0]
        if inclunub.shape[0] == 0:
            inclunub = 0
        correnum = correnum + np.max(inclunub)

    Purity = correnum/len(predY)

    # if pred_classnum
    res = bestMap(Y, predY)
    #  accuarcy
    ACC = sum(Y == res)/len(Y)
    #  NMI
    MIhat = MutualInfo(Y,res)
    
    result = [ACC, MIhat, Purity]
    return result


##
def bestMap(L1,L2):
    #bestmap: permute labels of L2 match L1 as good as possible
    #   [newL2] = bestMap(L1,L2);

    #===========    
    L1 = np.ndarray.flatten(L1)
    L2 = np.ndarray.flatten(L2)

    if np.shape(L1) != np.shape(L2):
        raise ValueError('size(L1) must == size(L2)')

#############################
    L1 = genfromtxt(r'C:\School\Master\Courses\Computational Learning\Ass4\USFS-code-master\L1.csv', delimiter=' ')
    L2 = genfromtxt(r'C:\School\Master\Courses\Computational Learning\Ass4\USFS-code-master\L2.csv', delimiter=' ')

##############################
    L1 = L1 - min(L1) # + 1      #   min (L1) <- 0 #1;
    L2 = L2 - min(L2) # + 1      #   min (L2) <- 0 #1;
    #===========    make bipartition graph  ============
    nClass = int(np.max([max(L1)+1, max(L2)+1]))
    G = np.zeros([nClass,nClass])
    for i in range(nClass):
        for j in range(nClass):
            L1i = np.where(L1 == i)
            L2j = np.where(L2 == j)
            G[i,j] = len(np.intersect1d(L1i, L2j)) # np.sum(np.where(L1i == L2j)) # sum(np.where(np.where(L1 == i) and np.where(L2 == j))[0])
            debug = True

    #===========    assign with hungarian method    ======
    c,t = hungarian(-G)
    newL2 = np.zeros((nClass,1))
    for i in range(nClass):
        newL2[L2 == i] = c[i]

    return newL2, c


##
def MutualInfo(L1,L2):
#   mutual information

    #===========    
    L1 = np.ndarray.flatten(L1)
    L2 = np.ndarray.flatten(L2)

    if np.shape(L1) != np.shape(L2):
        raise ValueError('size(L1) must == size(L2)')

    L1 = L1 - min(L1) + 1;      #   min (L1) <- 1;
    L2 = L2 - min(L2) + 1;      #   min (L2) <- 1;

    #===========    make bipartition graph  ============
    nClass = max(max(L1), max(L2))
    G = np.zeros((nClass))
    for i in range(nClass):
        for j in range (nClass):
            G[i,j] = np.sum(L1 == i and L2 == j) + np.finfo.eps(np.float64)

    sumG = np.sum(np.sum(G))
    #===========    calculate MIhat
    P1 = np.sum[G,2]
    P1 = P1/sumG
    P2 = np.sum[G,1]
    P2 = P2/sumG
    H1 = np.sum(-P1*np.log2(P1))
    H2 = np.sum(-P2*np.log2(P2))
    P12 = G/sumG
    PPP = P12/np.matlib.repmat(P2,nClass,1)/np.matlib.repmat(P1,1,nClass)
    PPP[abs(PPP) < 1e-12] = 1
    MI = sum(np.ndarray.flatten(P12) * np.log2(np.ndarray.flatten(PPP)))
    MIhat = MI / max(H1,H2)
    #############   why complex ?       ########
    MIhat = np.real(MIhat)

    return MIhat


##
def hungarian(A):
    #HUNGARIAN Solve the Assignment problem using the Hungarian method.
    #
    #[C,T]=hungarian(A)
    #A - a square cost matrix.
    #C - the optimal assignment.
    #T - the cost of the optimal assignment.
    #s.t. T = trace(A(C,:)) is minimized over all possible assignments.

    # Adapted from the FORTRAN IV code in Carpaneto and Toth, "Algorithm 548:
    # Solution of the assignment problem [H]", ACM Transactions on
    # Mathematical Software, 6(1):104-111, 1980.

    # v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.
    #                 Department of Computing Science, Umeå University,
    #                 Sweden. 
    #                 All standard disclaimers apply.

    # A substantial effort was put into this code. If you use it for a
    # publication or otherwise, please include an acknowledgement or at least
    # notify me by email. /Niclas

    m,n = np.shape(A)

    if m != n:
        raise ValueError('HUNGARIAN: Cost matrix must be square!')


    # Save original cost matrix.
    orig = A

    # Reduce matrix.
    A = hminired(A).astype(np.int16)

    # Do an initial assignment.
    A,C,U = hminiass(A)

    # Repeat while we have unassigned rows.
    while U(n+1):
        # Start with no path, no unchecked zeros, and no unexplored rows.
        LR = np.zeros((1,n))
        LC = np.zeros((1,n))
        CH = np.zeros((1,n))
        RH = np.zeros((1,n)).append(-1)

        # No labelled columns.
        SLC = []

        # Start path in first unassigned row.
        r = U[n+1]
        # Mark row with -of-path label.
        LR[r] = -1
        # Insert row first in labelled row set.
        SLR = r

        # Repeat until we manage to find an assignable zero.
        while (1):
            # If there are free zeros in row r
            if A[r,n+1] !=0:
                # ...get column of first free zero.
                l = -A[r,n+1]

                # If there are more free zeros in row r and row r in not
                # yet marked as unexplored..
                if (A[r,l]!=0 and RH[r]==0):
                    # Insert row r first in unexplored list.
                    RH[r] = RH[n+1]
                    RH[n+1] = r

                    # Mark in which column the next unexplored zero in this row
                    # is.
                    CH[r] = -A[r,l]

            else:
                # If all rows are explored..
                if RH[n+1]<=0:
                    # Reduce matrix.
                    [A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)


                # Re-start with first unexplored row.
                r = RH[n+1]
                # Get column of next free zero in row r.
                l = CH[r]
                # Advance "column of next free zero".
                CH[r] = -A[r,l]
                # If this zero is last in the list..
                if A[r,l]==0:
                    # ...remove row r from unexplored list.
                    RH[n+1] = RH[r]
                    RH[r] = 0



            # While the column l is labelled, i.e. in path.
            while LC[l] != 0:
                # If row r is explored..
                if RH[r] == 0:
                    # If all rows are explored..
                    if RH[n+1] <= 0:
                        # Reduce cost matrix.
                        A, CH, RH = hmreduce(A,CH,RH,LC,LR,SLC,SLR)

                    # Re-start with first unexplored row.
                    r = RH[n+1]

                # Get column of next free zero in row r.
                l = CH[r]

                # Advance "column of next free zero".
                CH[r] = -A[r,l]

                # If this zero is last in list..
                if A[r,l] == 0:
                    # ...remove row r from unexplored list.
                    RH[n+1] = RH[r]
                    RH[r] = 0

            # If the column found is unassigned..
            if C[l] == 0:
                # Flip all zeros along the path in LR,LC.
                A,C,U = hmflip(A,C,LC,LR,U,l,r) 
                # ...and exit to continue with next unassigned row.
                break
            else:
                # ...else add zero to path.

                # Label column l with row r.
                LC[l] = r

                # Add l to the set of labelled columns.
                SLC = SLC.append(l)

                # Continue with the row assigned to column l.
                r = C[l]

                # Label row r with column l.
                LR[r] = l

                # Add r to the set of labelled rows.
                SLR = SLR.append(r)
        
    # Calculate the total cost.

    sparse = csr_matrix((np.ones((20,)),(C,np.arange(np.shape(orig)[1])))).toarray().astype(int)
    T = np.sum(orig[sparse==1])
    
    return C, T


def hminired(A):
    #HMINIRED Initial reduction of cost matrix for the Hungarian method.
    #
    #B=assredin(A)
    #A - the unreduced cost matris.
    #B - the reduced cost matrix with linked zeros in each row.

    # v1.0  96-06-13. Niclas Borlin, niclas@cs.umu.se.

    m, n = np.shape(A)

    # Subtract column-minimum values from each column.
    colMin = A.min(axis=0)
    A = A - colMin  # colMin[np.ones(n,),:]

    # Subtract row-minimum values from each row.
    rowMin = A.min(axis=1)
    A = (A.T - rowMin).T

    # Get positions of all zeros.
    j,i = np.where(A.T==0)

    # Ext A to give room for row zero list header column.
    A = (np.vstack([A.T,np.zeros((1,len(A)))])).T
    for k in range(n):
        # Get all column in this row. 
        cols = j[np.where(k==i)].T
        A_cols = list(cols)
        A_cols.insert(0,n)
        V_cols = list(-1*cols)
        V_cols.extend([0])
        # Insert pointers in matrix.
        A[k,A_cols] = V_cols

    return A


def hminiass(A):
    #HMINIASS Initial assignment of the Hungarian method.
    #
    #[B,C,U]=hminiass(A)
    #A - the reduced cost matrix.
    #B - the reduced cost matrix, with assigned zeros removed from lists.
    #C - a vector. C(J)=I means row I is assigned to column J,
    #              i.e. there is an assigned zero in position I,J.
    #U - a vector with a linked list of unassigned rows.

    # v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n, np1 = np.shape(A)

    # Initalize return vectors.
    C = np.zeros((n,))
    C[:] = np.nan
    U = np.zeros((n+1,))
    U[:] = np.nan

    # Initialize last/next zero "pointers".
    LZ = np.zeros((n,))
    LZ[:] = np.nan
    NZ = np.zeros((n,))
    NZ[:] = np.nan

    for i in range(n):
        # Set j to first unassigned zero in row i.
        lj = n
        j = -A[i,lj]

        # Repeat until we have no more zeros (j==0) or we find a zero
        # in an unassigned column (c(j)==0).

        while not np.isnan(C[j]): # !=0:
            # Advance lj and j in zero list.
            lj=j
            j=-A[i,lj]
    
    		# Stop if we hit  of list.
            if (j==0):
                break
    
        if j != 0:
            # We found a zero in an unassigned column.

            # Assign row i to column j.
            C[j] = i

            # Remove A(i,j) from unassigned zero list.
            A[i,lj] = A[i,j]
            # Update next/last unassigned zero pointers.
            NZ[i] = -A[i,j]
            LZ[i] = lj
            # Indicate A(i,j) is an assigned zero.
            A[i,j] = 0
        else:
            # We found no zero in an unassigned column.
            # Check all zeros in this row.
            lj = n
            j = -A[i,lj]

            # Check all zeros in this row for a suitable zero in another row.
            while j != 0:
                # Check the in the row assigned to this column.
                r =C[j].astype(np.int16)

                # Pick up last/next pointers.
                lm = LZ[r].astype(np.int16)
                m= NZ[r].astype(np.int16)

                # Check all unchecked zeros in free list of this row.
                while not np.isnan(m): # != 0:
    				# Stop if we find an unassigned column.
                    if np.isnan(C[m]): # == 0:
                        break
    
    				# Advance one step in list.
                    lm = m
                    m = -A[r,lm]
    
                if np.isnan(m): # == 0:
                	# We failed on row r. Continue with next zero on row i.
                    lj = j
                    j = -A[i,lj]
                else:
                    # We found a zero in an unassigned column.

                    # Replace zero at (r,m) in unassigned list with zero at (r,j)
                    A[r,lm] = -j
                    A[r,j] = A[r,m]

                    # Update last/next pointers in row r.
                    NZ[r] = -A[r,m]
                    LZ[r] = j

                    # Mark A(r,m) as an assigned zero in the matrix . . .
                    A[r,m] = 0

                    # ...and in the assignment vector.
                    C[m] = r

                    # Remove A(i,j) from unassigned list.
                    A[i,lj] = A[i,j]

                    # Update last/next pointers in row r.
                    NZ[i] = -A[i,j]
                    LZ[i] = lj

                    # Mark A(r,m) as an assigned zero in the matrix . . .
                    A[i,j] = 0

                    # ...and in the assignment vector.
                    C[j] = i

                    # Stop search.
                    break
    
    # Create vector with list of unassigned rows.

    # Mark all rows have assignment.
    r = np.zeros((1,n))
    rows = C[np.where(~np.isnan(C))]#C[C != 0]
    r[0][rows] = rows
    empty = np.where(r == 0)

    # Create vector with linked list of unassigned rows.
    U = np.zeros((1,n+1))
    U[empty.insert(0,n+1)] = empty.insert(-1,0)

    
    return A,C,U 


def hmflip(A,C,LC,LR,U,l,r):
    #HMFLIP Flip assignment state of all zeros along a path.
    #
    #[A,C,U]=hmflip(A,C,LC,LR,U,l,r)
    #Input:
    #A   - the cost matrix.
    #C   - the assignment vector.
    #LC  - the column label vector.
    #LR  - the row label vector.
    #U   - the 
    #r,l - position of last zero in path.
    #Output:
    #A   - updated cost matrix.
    #C   - updated assignment vector.
    #U   - updated unassigned row list vector.

    # v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n = np.shape(A)[0]

    while (1):
        # Move assignment in column l to row r.
        C[l] = r

        # Find zero to be removed from zero list..

        # Find zero before this.
        m = np.where(A[r,:] == -l)

        # Link past this zero.
        A[r,m] = A[r,l]

        A[r,l] = 0

        # If this was the first zero of the path..
        if LR[r] < 0:
            # ...remove row from unassigned row list and return.
            U[n+1] = U[r]
            U[r] = 0
            return
        else:

            # Move back in this row along the path and get column of next zero.
            l = LR[r]

            # Insert zero at (r,l) first in zero list.
            A[r,l] = A[r,n+1]
            A[r,n+1] = -l

            # Continue back along the column to get row of next zero in path.
            r = LC[l]

    return A,C,U 



def hmreduce(A,CH,RH,LC,LR,SLC,SLR):
#HMREDUCE Reduce parts of cost matrix in the Hungerian method.
#
#[A,CH,RH]=hmreduce(A,CH,RH,LC,LR,SLC,SLR)
#Input:
#A   - Cost matrix.
#CH  - vector of column of 'next zeros' in each row.
#RH  - vector with list of unexplored rows.
#LC  - column labels.
#RC  - row labels.
#SLC - set of column labels.
#SLR - set of row labels.
#
#Output:
#A   - Reduced cost matrix.
#CH  - Updated vector of 'next zeros' in each row.
#RH  - Updated vector of unexplored rows.

# v1.0  96-06-14. Niclas Borlin, niclas@cs.umu.se.

    n = np.shape(A,1)

    # Find which rows are covered, i.e. unlabelled.
    coveredRows = LR==0

    # Find which columns are covered, i.e. labelled.
    coveredCols = LC!=0

    r = np.where(not coveredRows)
    c = np.where(not coveredCols)

    # Get minimum of uncovered elements.
    m = min(min(A[r,c]))

    # Subtract minimum from all uncovered elements.
    A[r,c] = A[r,c] - m

    # Check all uncovered columns..
    for j in c:
        # ...and uncovered rows in path order..
        for i in SLR:
            # If this is a (new) zero..
            if A[i,j] == 0:
                # If the row is not in unexplored list..
                if RH[i] == 0:
                    # ...insert it first in unexplored list.
                    RH[i] = RH[n+1]
                    RH[n+1] = i
                    # Mark this zero as "next free" in this row.
                    CH[i] = j

                # Find last unassigned zero on row I.
                row = A[i,:]
                colsInList = -row[row < 0]
                if len(colsInList) == 0:
                    # No zeros in the list.
                    l = n+1
                else:
                    l = colsInList[row[colsInList]==0]

                # App this zero to  of list.
                A[i,l] = -j

    # Add minimum to all doubly covered elements.
    r=np.where(coveredRows)
    c=np.where(coveredCols)

    # Take care of the zeros we will remove.
    [i,j]=np.where(A[r,c] <= 0)

    i = r[i]
    j = c[j]

    for k in range(len(i)):
        # Find zero before this in this row.
        lj = np.where(A[i[k],:] == -j[k])
        # Link past it.
        A[i[k],lj] = A[i[k],j[k]]
        # Mark it as assigned.
        A[i[k],j[k]] = 0


    A[r,c] = A[r,c] + m

    return A,CH,RH



def EProjSimplex_new(v, k=1):

    ft=1
    n = len(v)
    v0 = v - np.mean(v) + k/n
    vmin = np.min(v0)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 10**-10:
            v1 = v0 - lambda_m
            posidx = v1>0
            npos = sum(posidx)
            g = -npos
            f = sum(v1[posidx]) - k
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                x = np.zeros((v1.shape))
                x[np.argmax(v1)] = 1
                break
        x = np.zeros((v1.shape),dtype=np.int16)
        x[np.argmax(v1)] = 1
        
    else:
        x = v0

    return x, ft

def USFS(X, class_num, m, alpha, beta,gamma):

    """ input:
         X(d*n): data matrix, each column is a data point
         m: d*m projection matrix
         class_num: the number of class class_num
         alpha: parameter for updateing Y
         beta: parameter for updateing W
         gamma:parameter for updateing P
     output:
         projective matrix W(d*m)
         cluster soft label matrix Y(n*c)
         feature selection matrix P(d*c)
         the center matrix of clustering M(m*c)
         """

    def initfcm(cluster_n, data_n):
        U = np.random.uniform(0, 1, [cluster_n, data_n])
        col_sum = sum(U)
        U = U/col_sum
        return U

    # initial
    [d, n] = np.shape(X)
    epsil = 0.01
    INTER = 5
    P = np.random.uniform(0,1,[d, class_num])
    M = np.random.uniform(0,1,[m, class_num])
    Y = initfcm(class_num, n)
    Y = Y.T

    # optimization
    for _ in range(INTER):
        B = np.eye(n,n)
        D = np.zeros((class_num,class_num))
        for i in range(class_num):
            D[i,i] = np.sum(Y[:,i])

        Sw = X@(B-(Y@D@Y.T))@X.T
        val,vec = np.linalg.eig(Sw)
        val = val * np.eye(len(val))
        di = np.argsort(np.diag(val))
        W = vec[:, di[:m]]
        if not np.isreal(W).any():
            W = abs(W)
        # update M
        for j in range(class_num):
            aa = 0
            for i in range(n):
                aa = aa + Y[i,j]*np.dot( W.T,X[:,i])

            M[:,j] = aa/sum(Y[:,j])
        # update P
        temp_gamma = np.zeros((d,1))
        for i in range(d):
            temp_gamma[i] = 1/(2*np.sqrt(P[i,:]@P[i,:].T)+epsil)
        
        Gamma  = np.diag(temp_gamma)
        P = np.linalg.solve(X@X.T + (gamma/alpha)*Gamma,X@Y)
        # update Y
        distance = np.zeros((n,class_num))
        for i in range(n):
            for j in range(class_num):
                distance[i,j] = mydist(W.T@X[:,i],M[:,j])
                distance[i,j] = distance[i,j]**2
            ad = 0.5*(alpha*(X[:,i].T@P) - (distance[i,:]/(2*beta)))
            Y[i,:], _ = EProjSimplex_new(ad)  
    
    return W, Y, P, M  


def main():
    data = loadmat(r'C:\School\Master\Courses\Computational Learning\Ass4\USFS-code-master\Datasets\COIL20.mat')
    X = data['fea'].T
    label = data['gnd']
    label = data['gnd'] - 1 # matlab...

    class_num = len(np.unique(label))
    d, n = X.shape
    FeaNumCandi = 20
    alpha = 10000
    gamma = 10000
    beta = 1
    m = 45
    # run SLUFS
    [W, Y, P, M] = USFS(X, class_num, m, alpha, beta,gamma)
    W1 = []
    for k in range(d):
        W1.extend([np.linalg.norm(P[k,:], ord=2)])
    
    index = np.argsort(W1, )[::-1] #descing
    index = np.expand_dims(index, axis=0)
    new_fea = np.squeeze(X[index[0:FeaNumCandi],:])
    idx = KMeans(n_clusters=class_num, random_state=110891).fit_predict(new_fea.T)
    result = ClusteringMeasure(label, idx)
    a = 1


if __name__=='__main__':
    main()
