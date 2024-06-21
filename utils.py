import numpy as np
import  matplotlib.pyplot as plt
import warnings, os
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from scipy.linalg import lapack
import statsmodels.stats.correlation_tools as corr 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds
from uncertainties import unumpy as unp



global partitions 
partitions = {
    'savio2': {'nodes': 163, 'cores_per_node': 24},
    'savio2_bigmem': {'nodes': 44, 'cores_per_node': 24},
    'savio2_htc': {'nodes': 20, 'cores_per_node': 12},
    'savio2_gpu': {'nodes': 25, 'cores_per_node': 8},
    'savio2_knl': {'nodes': 28, 'cores_per_node': 64},
    'savio3': {'nodes': 192, 'cores_per_node': 32},
    'savio3_bigmem': {'nodes': 20, 'cores_per_node': 32},
    'savio3_htc': {'nodes': 24, 'cores_per_node': 40},
    'savio3_xlmem': {'nodes': 4, 'cores_per_node': 32},
    'savio3_gpu': {'nodes': 33, 'cores_per_node': 8},
    'savio4_htc': {'nodes': 108, 'cores_per_node': 56},
    'savio4_gpu': {'nodes': 26, 'cores_per_node': 32}
}



def array_txt(array):
    """takes an array and transforms it into a space seperated string to be inputed in Serpent txt file"""
    if np.array(array).ndim==2:
        # 2-dimensional array
        return '\n'.join([' '.join(map(str, row)) for row in array])
    else:
        # 1-dimensional array
        return ' '.join(map(str, array))

def zai_to_nuc_name(zaid):
    
    '''
    Convert from ZAI format to nuclide name
    e.g. nuclide 10010 for Hydrogen (Z=1) or 90190 for Fluorine (Z = 9)
    zaid (str)
    '''
    if isinstance(zaid, int):
        zaid=str(zaid)
    if 'lwtr' in zaid:
        isoName = 'S(alpha, beta) light water'

    elif "total" in zaid:
        return "total"

    else:
        # Read Z, A, I for single-digit Z values
        if len(zaid) == 5:
            Z = int(zaid[0:1])       # Atomic number
            A = str(int(zaid[1:4]))  # Mass number
            if A=="0":
                A="nat"
            if zaid[-1]!=0 or zaid!=0 == 1:
                warnings.warn("Make sure you are using the zaid format ending in 0 for ground state")
            # ID = zai[4:5]           # Isomeric state ID

        # Read Z, A, I for double-digit Z values
        elif len(zaid) == 6:
            Z = int(zaid[0:2])
            A = str(int(zaid[2:5]))
            if A=="0":
                A="nat"
            # ID = zai[5:6]
        else:
            raise ValueError("Make sure the length of the zaid is 5 or 6")

        ZDict = {1:  "H" ,  2: "He",  3: "Li", 4: "Be" , 5: "B"  , 6:  "C" ,
                 7:  "N" ,  8: "O" ,  9: "F" , 10: "Ne", 11: "Na", 12: "Mg",
                 13: "Al", 14: "Si", 15: "P" , 16: "S" , 17: "Cl", 18: "Ar",
                 19: "K" , 20: "Ca", 21: "Sc", 22: "Ti", 23: "V" , 24: "Cr",
                 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
                 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
                 37: "Rb", 38: "Sr", 39: "Y" , 40: "Zr", 41: "Nb", 42: "Mo",
                 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
                 49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I" , 54: "Xe",
                 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
                 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy",
                 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu", 72: "Hf",
                 73: "Ta", 74: "W" , 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
                 79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po",
                 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
                 91: "Pa", 92: "U" , 93: "Np", 94: "Pu", 95: "Am", 96: "Cm",
                 97: "Bk", 98: "Cf", 99: "Es"}

        isoName = ZDict[Z] + '-' + A

    return isoName

def nuc_name_to_zaid(nuc):
    "nuc name in Cl-35 format"
    Z,A=nuc.split("-") 
    ZDict= {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19,
    'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28,
    'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37,
    'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
    'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55,
    'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73,
    'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91,
    'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99}
    mass_number = int(A)
    
    # Get the atomic number using the reversed dictionary
    atomic_number = ZDict[Z]
    
    # Form the ZAI number
    zai_number = f"{atomic_number:02d}{mass_number:03d}"
    
    return zai_number


def MTtoRX(MFMT):
    '''
    Convert from MFx/MTx to reaction string
    '''

    # Dictionary mapping MFx/MTx to reaction string
    MFMT_Dict = {"1": "total", "2": "elastic",
                 "4": "inelastic", "16": "n,2n", "18": "fission","28":"n,np","51":"inelastic (1st)","52":"inelastic (2nd)","53":"inelastic (3rd)","54":"inelastic (4th)","55":"inelastic (5th)",
                 "91": "inelastic (cont)","103": "n,p", "102": "n,gamma",
                 "104": "n,d", "105": "n,t", "106": "n,3he",
                 "107": "n,alpha","111":"n,2p", "182": "chi delayed", "452":"nubar total","455":"nubar delayed","456": "nubar prompt","1018": "chi total"}

    ret_to_str=False
    # Check if MFx/MTx is in the dictionary
    if isinstance(MFMT,str):
        MFMT=[MFMT]
        ret_to_str=True
    list_RX=[]
    for MT in MFMT:
        if MT in MFMT_Dict:
            # If so, return the corresponding reaction string
            list_RX.append(MFMT_Dict[MT])
        else:
            list_RX.append(MT)
            # If not, raise a ValueError
            #raise ValueError("Mapping not available for given MFx/MTx: {}".format(MFMT))
    if ret_to_str:
        return list_RX[0]
    return list_RX
def sssmtlist_to_RXlist(list_MT):
    """Convert a list from serpent output in MT format (i.e. mt n xs) into RX

    Args:
        list (list): list of str with each reaction
    """
    conv={'ela scatt xs': 'ela', 'sab scatt xs': 'sab', 'inl scatt xs': 'inl', 'fission xs': 'fiss', 'nxn xs': 'nxn', 'total xs': 'total', 'capture xs': 'capt'}

    
    list_RX=[]
    for MT in list_MT:
        num=MT.split()[1]
        if num.isdigit():
            list_RX.append(MTtoRX(num))
        else:
            if MT.split()[-1]=="xs":
                list_RX.append(conv[MT])
            else:
                list_RX.append(MT)
    return list_RX

def rel_diff(total,total_s,sum,sum_s):
    rel_dif=np.abs((total-sum)/total)*100
    rel_dif_s=np.sqrt((np.sqrt(total_s**2+sum_s**2)/np.abs(total-sum))**2+(total_s/total)**2)*rel_dif
    return rel_dif, rel_dif_s

def MT_to_serpent_MT(list_MT):
    """takes mt number or names of sum reaction and transforms it into serpent output file readable

    Args:
        list_MT (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    conv={"ela":"ela scatt","sab":"sab scatt","inl":"inl scatt","fiss":"fission","nxn":"nxn","total":"total","capt":"capture"}
    serpent_MT=[]
    for MT in list_MT: 
        if isinstance(MT,int) or MT.isdigit():
            serpent_MT.append(f"mt {MT} xs")
        elif MT.isalpha() :
            if MT in conv:
                serpent_MT.append(f"{conv[MT]} xs")
            else:
                serpent_MT.append(MT)
#    else:
#        raise ValueError("Make sure list is only MT or only sum reactions can't be both")
    return serpent_MT

def plot_two_sens(array1,array2,label_array1="",label_array2="",plot_name="sens_compare.png"):
    """rank the sensitivity with respect to array1 and see what the values of array2 are for the same sens, it is worth swapping array 1 and 2. Bar plot of the two compared

    Args:
        array1 (array): array in the format of [[zaid,MT,sens,sens_s]]
        array2 (array): same
        label_array1 (str, optional): label for legend. Defaults to "".
        label_array2 (str, optional): label for legend. Defaults to "".
        plot_name (str, optional): for saving can be a path and a name. Defaults to "sens_compare.png".

    Raises:
        ValueError: must be looking at the same reaction or same isotope in plot
    """
    
    def transform_array(array):
        """takes an array in format [[ZAID, MT, sens,sens_s]] and checks whether one of the column is constant meanning the array is only for one isotope or one reaction, and removes that column
        into the title_str. If neither Zaid nor the MT is constant combines these two columns into a single connected with a -. This function is for bar plotting purposes.

        Args:
            array (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Check if the first row is constant
        if np.all(array[:,0] == array[0,0]):
            # Remove the first column
            transformed_array = np.hstack((array[:,1].reshape(-1,1),array[:,2:]))
            title_str=array[0,0]
        else:
            # Check if the second row is constant
            if np.all(array[:,1] == array[0,1]):
                # Remove the second column
                transformed_array = np.delete(array, 1, axis=1)
                title_str=array[0,1]+" xs"
            else:
                # Combine the first and second columns into one string separated by a dash
                combined_col = np.array([[f"{a1}-{[a2]}"] for a1, a2 in zip(array[:,0], array[:,1])])
                transformed_array = np.hstack((combined_col, array[:,2:]))
                title_str="all"

        return transformed_array, title_str
    
    array1,str_title=transform_array(array1)
    array2,str_title_temp=transform_array(array2)
    if str_title!=str_title_temp:
        raise ValueError("To plot two sensitivity must be for the same reaction or same isotope")

    #remove all the values 100 times smaller than the max (all in absolute value)
    max_value = np.max(np.abs(np.float64(array1[:, 1])))
    array1 = array1[np.abs(np.float64(array1[:, 1])) >= max_value / 100]

    #unique_labels = np.unique(np.concatenate((array1[:, 0], array2[:, 0])))
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()

    # Plot bars for array1
    x = np.arange(len(array1))
    do_first=0
    for i, label in enumerate(array1[:,0]):
        index2 = np.where(array2[:, 0] == label)[0]

    
        value1 = array1[i, 1]
        uncertainty1 = array1[i, 2]
        ax.bar(x[i] - width/2, float(value1), yerr=float(uncertainty1), width=width, color='blue', alpha=0.5, label=label_array1 if i==0 else None)
        
        if len(index2) > 0:
            do_first+=1
            value2 = array2[index2, 1]
            uncertainty2 = array2[index2, 2]
            ax.bar(x[i] + width/2, float(value2), yerr=float(uncertainty2), width=width, color='red', alpha=0.5, label=label_array2 if do_first==1 else None)

    # Set labels and title
    ax.set_xticks(x)
    ax.set_xticklabels(array1[:,0])
    ax.set_title(f"Sensitivity for {str_title}")
    ax.set_xlabel('')
    plt.xticks(rotation=45, ha='right')
    ax.set_ylabel('Sensitivity')
    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_name,dpi=400)

def schrink_psd(matrix,tol=1):
    M1=np.identity(len(matrix))
    def S(a):
        return a*M1+(1-a)*matrix
        
    al=0; ar=1
    while ar-al>tol:
        am=(al+ar)/2
        if isPD(S(am)):
            al=am
        else:
            ar=am
    return S(ar)

def deflate_sparse_matrix(sparse_matrix, k):
    """
    Deflates a sparse matrix using Singular Value Decomposition (SVD).

    Parameters:
    sparse_matrix (scipy.sparse.csc_matrix): Input sparse matrix.
    k (int): Number of singular values and vectors to compute.

    Returns:
    scipy.sparse.csc_matrix: Deflated matrix.
    """
    # Perform SVD
    u, s, vt = svds(sparse_matrix, k=k,which="LM")

    # Reconstruct the deflated matrix
    s_matrix = np.diag(s)
    deflated_matrix = u @ s_matrix @ vt

    return csc_matrix(deflated_matrix).toarray()

def nearest_pd(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def nearest_psd(matrix):
    """Find the nearest Positive Semi-Definite (PSD) matrix to a given matrix."""
    # Perform eigenvalue decomposition
    matrix = (matrix + matrix.T) / 2

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    
    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 0
    
    # Reconstruct the matrix with non-negative eigenvalues
    psd_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Ensure symmetry (to counteract numerical errors)
    #psd_matrix = (psd_matrix + psd_matrix.T) / 2
    
    return psd_matrix

def psd_closeness(matrix):
    """Calculate how close the matrix is to being PSD."""
    # Ensure the matrix is symmetric
    sym_matrix = (matrix + matrix.T) / 2
    
    # Perform eigenvalue decomposition
    eigenvalues = np.linalg.eigvalsh(sym_matrix)
    
    # Identify negative eigenvalues
    negative_eigenvalues = eigenvalues[eigenvalues < 0]
    
    # Calculate the sum of the absolute values of negative eigenvalues
    closeness_metric = np.sum(np.abs(negative_eigenvalues))
    
    # Alternative metric: maximum negative eigenvalue (absolute value)
    max_negative_eigenvalue = np.min(eigenvalues)
    
    # Output both metrics
    return {
        'sum_of_negative_eigenvalues': closeness_metric,
        'max_negative_eigenvalue': max_negative_eigenvalue
    }

def deflate_sparse_matrix_eigen(sparse_matrix):
    """
    Deflates a symmetric sparse matrix using eigenvalue decomposition.

    Parameters:
    sparse_matrix (scipy.sparse.csc_matrix): Input symmetric sparse matrix.

    Returns:
    scipy.sparse.csc_matrix: Deflated matrix.
    """
    sparse_matrix=(sparse_matrix+sparse_matrix.T)/2.

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(sparse_matrix)
    #eigenvectors=eigenvectors[:,1:]
    #find the rayleigh quotient and the residual to find the bound of rounding error the bound is given by abs(lam_i-rho_i)+norm2(residual)
    rayleigh_quotients=[]
    residuals=[]
    non_zero_indices = np.where(eigenvalues != 0)[0]
    eigenvalues = eigenvalues[non_zero_indices]
    eigenvectors = eigenvectors[:, non_zero_indices]
    if True:
        l=0
        k=0
        for i,x in enumerate(eigenvectors.T):
            #find the parameter 
            rq =  np.dot(x.T, np.dot(sparse_matrix, x))/np.dot(x.T, x)
            rayleigh_quotients.append(rq)
            residual = np.dot(sparse_matrix, x) - rq * x
            residuals.append(residual)
            #bound=np.abs(eigenvalues[i]-rq)+np.linalg.norm(residual) 
        
        for i,v in enumerate(eigenvalues):
            diff=np.abs(v-rayleigh_quotients[i])
            res_norm=np.linalg.norm(residual[i]) #norm 2
            bound=diff+res_norm
            g_list=[]
            for j,rq in enumerate(rayleigh_quotients):
                if i==0 and j==0 or j==1:
                   g_list.append(np.abs(rayleigh_quotients[i]-rq)-res_norm)
                if j==i-1 or j==i+1:
                    g_list.append(np.abs(rayleigh_quotients[i]-rq)-res_norm)
            g=np.min(g_list)
            tight_bound=diff+res_norm**2/g
            bound=np.min([bound,tight_bound])
            if not eigenvalues[i]<0:
                k+=1
            if not v<-bound:
                l+=1
                eigenvalues[i]=0

    
    #bound=np.abs(eigenvalues-rayleigh_quotients)+np.linalg.norm(residuals)
    # Set positive eigenvalues to zero to deflate the matrix of only the negative eigenvalues
    #eigenvalues[eigenvalues > 0] = 0
    mat_mul=eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    H=3*np.finfo(float).eps*mat_mul
    # Reconstruct the deflated matrix
    deflated_matrix =sparse_matrix- mat_mul+H
    
    return csc_matrix(deflated_matrix).toarray()

def get_unc_covar(mat,S_nuc):
    """propagate uncertainty through a matrix, but also propagate uncertainty from statistical origin in serpent
    using the fact that var=sum_i(dA/dSi*var_si)^2 where A=S.TMS, then expand out and you get unc_stat=2*sqrt(S.TM var_S M.TS).

    mat (numpy.array): matrix of covariance
    S_nuc (numpy.array): uncertainty.uarray containing stat uncertainty from serpent from sensitivity.
    """
    S_nom = unp.nominal_values(S_nuc)
    S_std = unp.std_devs(S_nuc)
    # Construct the diagonal matrix of variances
    Sigma_S = np.diag(S_std**2)
    # Compute S^T M
    SM = S_nom @ mat
    covar=SM @S_nom
    # Compute the uncertainty in the scalar A = S^T M S
    sigma_A = 2 * np.sqrt(SM @ Sigma_S @ SM.T)
    return unp.uarray(covar,sigma_A)

def isPD(matrix):
    try:
        # Attempt Cholesky decomposition
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False
