import numpy as np
import  matplotlib.pyplot as plt
import warnings, os
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
from scipy.linalg import lapack
import statsmodels.stats.correlation_tools as corr


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

def MTtoRX(MFMT):
    '''
    Convert from MFx/MTx to reaction string
    '''

    # Dictionary mapping MFx/MTx to reaction string
    MFMT_Dict = {"1": "total", "2": "elastic",
                 "4": "inelastic", "16": "n,2n", "18": "fission","28":"n,np","51":"inelastic (1st excited)","52":"inelastic (2nd excited)","53":"inelastic (3rd excited)","54":"inelastic (4th excited)","55":"inelastic (5th excited)",
                 "103": "n,p", "102": "n,gamma",
                 "104": "n,d", "105": "n,t", "106": "n,3he",
                 "107": "n,alpha","111":"n,2p", "182": "chi delayed"}

    # Check if MFx/MTx is in the dictionary
    if MFMT in MFMT_Dict:
        # If so, return the corresponding reaction string
        return MFMT_Dict[MFMT]
    else:
        return MFMT
        # If not, raise a ValueError
        #raise ValueError("Mapping not available for given MFx/MTx: {}".format(MFMT))

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
    if all(isinstance(MT,int) or MT.isdigit()for MT in list_MT):
        serpent_MT=[f"mt {x} xs" for x in list_MT]
    elif all(MT.isalpha() and MT in conv for MT in list_MT) :
        serpent_MT=[ f"{conv[x]} xs"  for x in list_MT ]
    else:
        raise ValueError("Make sure list is only MT or only sum reactions can't be both")
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

def get_ND_cov_matrix(zaid_rx, N_E_G, cov_path):

    '''
    Load the covariance data that are in the form of .npy binary files that
    were created by the empy script supplied by Caleb Mattoon

    N_E_G : number of energy groups
    zais_rx : first column is zaid number, second column is MT (rx)
    '''

    N_rxs = len(zaid_rx)  # number of nuclear data sets
    

    M_sigma = np.zeros((N_rxs*N_E_G, N_rxs*N_E_G))  # initialize nuclear data cov matrix
    for i, row1 in enumerate(zaid_rx):
        #since correlation between block do 2 loops for the correlation between the two.  
        # i is the index of the row and row1 contains the two columns of the being the zaid and rx.
        iStart1 = i*N_E_G   # indices for M_sigma
        iEnd1 = (i+1)*N_E_G

        zaid1 = row1[0][:-1]
        #rx1 = RXtoMFMT(row1[1])
        rx1 = row1[1]

        for j, row2 in enumerate(zaid_rx):

            iStart2 = j*N_E_G    # indices for M_sigma
            iEnd2 = (j+1)*N_E_G

            zaid2 = row2[0][:-1]
            rx2=row2[1]#rx2 = RXtoMFMT(row2[1])

            if zaid1 != zaid2:  # no inter nuclide correlations, skip loop
                continue

            file_name = f"{zaid1}-{rx1}-{zaid2}-{rx2}.npy"
            file = os.path.join(cov_path, file_name)

            # File that is the opposite transpose
            file_name = f"{zaid2}-{rx2}-{zaid1}-{rx1}.npy" 
            file_opposite = os.path.join(cov_path,  file_name)

            # If the npy exists, otherwise, the matrix will stay as zeros
            if os.path.exists(file):
                M_sigma[iStart1:iEnd1, iStart2:iEnd2] = np.load(file)
            elif os.path.exists(file_opposite):  # if the transpose matrix exists, load it and transpose it
                M_sigma[iStart1:iEnd1, iStart2:iEnd2] = np.load(file_opposite).T
            #else:
            #    pass
                #print('%s not found. Covariances set to 0' % file)

    #TODO check positive semi-definite
    # Loads a triangular matrix, need to make it symmetric
    # M_sigma = M_sigma + M_sigma.T - np.diag(M_sigma)
    #M_sigma = (M_sigma + M_sigma.T)/2.
    M_sigma= deflate_sparse_matrix_eigen(M_sigma,300).toarray()#deflate_sparse_matrix(M_sigma,1000).toarray()
    
    return M_sigma
        
        
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

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
    print(s)

    # Reconstruct the deflated matrix
    s_matrix = np.diag(s)
    deflated_matrix = u @ s_matrix @ vt

    return csc_matrix(deflated_matrix)

def deflate_sparse_matrix_eigen(sparse_matrix, k):
    """
    Deflates a symmetric sparse matrix using eigenvalue decomposition.

    Parameters:
    sparse_matrix (scipy.sparse.csc_matrix): Input symmetric sparse matrix.
    k (int): Number of eigenvalues and eigenvectors to compute.

    Returns:
    scipy.sparse.csc_matrix: Deflated matrix.
    """
    if not np.allclose(sparse_matrix, sparse_matrix.T):
        raise ValueError("The input matrix must be symmetric for eigenvalue decomposition.")

    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = sla.eigs(sparse_matrix, k=k, which='LR')
    print(eigenvalues,eigenvectors[-1])

    # Reconstruct the deflated matrix
    deflated_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return csc_matrix(deflated_matrix)

# Example usage
if __name__ == "__main__":
    # Create a sample sparse matrix
    data = np.array([1, 2, 3, 4, 5])
    row_indices = np.array([0, 1, 2, 3, 4])
    col_indices = np.array([0, 1, 2, 3, 4])
    sparse_matrix = csc_matrix((data, (row_indices, col_indices)), shape=(5, 5))

    print("Original Sparse Matrix:")
    print(sparse_matrix.toarray())

    # Deflate the sparse matrix
    k = 2  # Number of singular values to keep
    deflated_matrix = deflate_sparse_matrix(sparse_matrix, k)

    print("\nDeflated Sparse Matrix:")
    print(deflated_matrix.toarray())
