import numpy as np
import sandy,os,matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from utils import *

E_GRID={"ANL_33":np.array( [1e-11, 4.17458120e-07, 5.31578507e-07, 3.92786341e-06, 8.31528691e-06
, 1.37095904e-05, 2.26032933e-05, 3.72665304e-05, 6.14421214e-05
, 1.01300933e-04, 1.67017002e-04, 2.75364484e-04, 4.53999282e-04
, 7.48518273e-04, 1.23409800e-03, 2.03468362e-03, 3.35462616e-03
, 5.53084351e-03, 9.11881934e-03, 1.50343914e-02, 2.47875209e-02
, 4.08677130e-02, 6.73794677e-02, 1.11089962e-01, 1.83156383e-01
, 3.01973824e-01, 4.97870667e-01, 8.20849958e-01, 1.35335279e+00
, 2.23130153e+00, 3.67879429e+00, 6.06530639e+00, 9.99999966e+00
, 1.41906750e+01])*1e6,"ANL_dani":np.array([1.00000E-05, 4.17460E-01, 5.31580E-01, 3.92790E+00, 8.31530E+00,
1.37100E+01, 2.26030E+01, 3.72670E+01, 6.14420E+01, 1.01300E+02, 
1.67020E+02, 2.75360E+02, 4.54000E+02, 7.48520E+02, 1.23410E+03, 
2.03470E+03, 3.35460E+03, 5.53080E+03, 9.11880E+03, 1.50340E+04,
2.47880E+04, 4.08680E+04, 6.73790E+04, 1.11090E+05, 1.83160E+05,
3.01970E+05, 4.97870E+05, 8.20850E+05, 1.35340E+06, 2.23130E+06, 
3.67880E+06, 6.06530E+06, 1.00000E+07, 1.41910E+07]),"ANL_116":np.array([1.e-5,4.1746e-01, 5.3158e-01, 6.8256e-01, 8.7642e-01, 1.4450e+00,
       2.3824e+00, 3.9279e+00, 6.4759e+00, 1.0677e+01, 1.7603e+01,
       2.9023e+01, 4.7851e+01, 7.8893e+01, 1.3007e+02, 2.1445e+02,
       3.5357e+02, 5.8295e+02, 9.6112e+02, 1.0622e+03, 1.1739e+03,
       1.3639e+03, 1.5846e+03, 1.8411e+03, 2.2487e+03, 2.6126e+03,
       2.8634e+03, 3.3546e+03, 4.0973e+03, 4.5283e+03, 5.5308e+03,
       7.1017e+03, 9.1188e+03, 1.1709e+04, 1.5034e+04, 1.9305e+04,
       2.1335e+04, 2.4787e+04, 2.8088e+04, 3.1828e+04, 3.6066e+04,
       4.0868e+04, 5.2475e+04, 5.9462e+04, 6.7379e+04, 8.6517e+04,
       1.1109e+05, 1.1679e+05, 1.2907e+05, 1.4264e+05, 1.5764e+05,
       1.6573e+05, 1.8316e+05, 2.0242e+05, 2.2371e+05, 2.4724e+05,
       2.7324e+05, 2.8725e+05, 3.0197e+05, 3.1746e+05, 3.3373e+05,
       3.6883e+05, 4.0762e+05, 4.5049e+05, 4.9787e+05, 5.2340e+05,
       5.7844e+05, 6.3928e+05, 7.0651e+05, 7.8082e+05, 8.6294e+05,
       9.5369e+05, 1.0026e+06, 1.0540e+06, 1.1648e+06, 1.2246e+06,
       1.3534e+06, 1.4957e+06, 1.5724e+06, 1.6530e+06, 1.8268e+06,
       1.9205e+06, 2.0190e+06, 2.2313e+06, 2.2688e+06, 2.3851e+06,
       2.4660e+06, 2.7253e+06, 3.0119e+06, 3.1664e+06, 3.3287e+06,
       3.6788e+06, 4.0657e+06, 4.4933e+06, 4.7237e+06, 4.9659e+06,
       5.2205e+06, 5.7695e+06, 5.9156e+06, 6.2189e+06, 6.5377e+06,
       6.7032e+06, 7.0469e+06, 7.4082e+06, 7.7880e+06, 8.1873e+06,
       8.6071e+06, 9.0484e+06, 9.5123e+06, 1.0000e+07, 1.0513e+07,
       1.1052e+07, 1.1618e+07, 1.2214e+07, 1.2840e+07, 1.3499e+07,
       1.4191e+07])
}

class Covariance():
    def __init__(self,iso,lib="ENDF8",iwt=8,temp=900,e_grid_name="ANL_33",name_ext=""):
        self.iso=iso #ZAID format
        self.MT={}
        self.lib=lib
        libs=["LANL","TENDL2023","ENDF8","JEFF33"]
        if self.lib not in libs:
            raise ValueError(f"Make sure the library {self.lib} is correctly spelt from one of these {libs}")
        self.iwt=iwt
        self.temp=temp
        self.e_grid=E_GRID[e_grid_name]#TODO create e_grid lib?
        self.neg=len(self.e_grid)-1
        #path to input files for TENDL and LANL not available in sandy by default
        self.path_endf=f"C:\\Users\\jbval\\PDM\\02_cov\\input_data\\{self.lib}" #TODO
        
        #output path for the binary files to be written
        self.path_npy=f"C:\\Users\\jbval\\PDM\\02_cov\\cov_npy\\{self.lib}-{e_grid_name}{name_ext}"
        if not os.path.exists(self.path_npy):
            os.makedirs(self.path_npy)
        
    def write_binary(self,**kwargs):
        #TODO run serpent, get the reactions from the covariance and write them into binary like Daniel's code, check if MT exist before, 
        #make sure that the transpose of cross term always exists
        for iso in self.iso:
            str_iso=str(iso)[:-1]
            #get the record of which reaction are available from ENDF file
            if self.lib=="ENDF8":
                tape=sandy.get_endf6_file("ENDFB_80", "xs", iso)
                
            if self.lib=="TENDL2023":
                endf_file=os.path.join(self.path_endf,f"n-{str_iso}.tendl")
                tape=sandy.Endf6.from_file(endf_file)
            if self.lib=="LANL":
                if not str_iso=="17035":
                    raise ValueError(f"LANL evaluation is only for Cl-35 not for {str_iso}")
                endf_file=os.path.join(self.path_endf,f"n-{str_iso}.evl")
                tape=sandy.Endf6.from_file(endf_file)

            if self.lib=="JEFF33":
                tape=sandy.get_endf6_file("JEFF_33", "xs", iso)

            rec=tape.get_records()
            self.MT[str_iso]=rec[rec["MF"]==33]["MT"].values
            self.MT[str_iso]=np.append(self.MT[str_iso],rec[rec["MF"]==31]["MT"].values)
            MAT=rec["MAT"][0]
            
            if 18 in rec[rec["MF"]==35]["MT"].values:
                self.MT[str_iso]=np.append(self.MT[str_iso], 1018)
            print(self.MT[str_iso])
            if len(self.MT[str_iso])==0:
                continue

            errorr_hot = tape.get_errorr(#sandy.get_endf6_file("ENDFB_80", "xs", iso).get_errorr(
                    err=0.001,
                    errorr33_kws=dict(
                        #mt=[102],
                        iread=1,
                        irespr=1,  # faster
                        ek=self.e_grid,  # only above threshold
                        iprint=1,
                        iwt=self.iwt,
                        lord=1,
                    ),
                    errorr31_kws=dict(
                        #mt=[102],
                        iread=1,
                        irespr=1,  # faster
                        ek=self.e_grid,  # only above threshold
                        iprint=1,
                        iwt=self.iwt,
                        lord=1,
                    ),
                    errorr35_kws=dict(
                        #mt=[102],
                        iread=1,
                        irespr=1,  # faster
                        ek=self.e_grid,  # only above threshold
                        iprint=1,
                        iwt=self.iwt,
                        lord=1,
                    ),
                    verbose=True,
                    nubar=True,
                    mubar=False,
                    chi=True,
                    temperature=self.temp,
                    purr=False,
                    heatr=False,
                    gaspr=False,
                    thermr=False,
                    groupr=True,
                    groupr_kws=dict(ek=self.e_grid,iwt=self.iwt,irespr=1,lord=1, iprint=1,)
                    
                )
            
            cov33 = errorr_hot['errorr33'].get_cov().data
            if 456 in self.MT[str_iso]:
                cov31 = errorr_hot["errorr31"].get_cov().data
            if 1018 in self.MT[str_iso]:
                cov35 = errorr_hot["errorr35"]

            for MT1 in self.MT[str_iso]:
                for MT2 in self.MT[str_iso]:
                    bin_name=f"{str_iso}-{MT1}-{str_iso}-{MT2}.npy"
                    bin_name_perm=f"{str_iso}-{MT2}-{str_iso}-{MT1}.npy"
                    if not os.path.exists(os.path.join(self.path_npy,bin_name_perm)):
                        #check if permuted version doesn't already exist
                        no_mat=False
                        if MT1 in [452,455,456] or MT2 in [452,455,456]:
                            if  MT1 in [452,455,456] and MT2 in [452,455,456] :
                                #get the nubar covariances from MF=31 file
                                matrix=cov31[MAT,MT2].loc[(MAT,MT1)]
                            else:
                                no_mat=True
                        elif MT1 == 1018 or MT2==1018:
                            if MT1==1018 and MT2 == 1018:
                                #get the chi disitribution covariance from the MF=35
                                matrix=sandy.errorr.read_mf35(cov35,int(MAT),18)["COVS"][18]
                            else:
                                no_mat=True
                        else:
                            matrix=cov33[MAT,MT2].loc[(MAT,MT1)]
                            if np.max(np.abs(matrix))==0:
                                no_mat=True
                        
                        if not no_mat:
                            np.save(os.path.join(self.path_npy,bin_name),matrix)
                   #check if MT pair exist in cov, but also if binary not already written for permutation. 
            print(f"Written {str_iso} binary files for Covariances")
            
        
    def compare_old(self,iso,MT1,MT2,path_old):
        iso=str(iso)[:-1]
        file=f"{iso}-{MT1}-{iso}-{MT2}.npy"
        JB=np.load(os.path.join(self.path_npy,file))
        Dani=np.load(os.path.join(path_old,file))
        vmax=np.max(np.abs(JB-Dani))
        plt.figure()
        if vmax==0:
            img=plt.imshow(JB-Dani, cmap="bwr",interpolation="nearest" )
        else:
            img = plt.imshow(JB-Dani, cmap="bwr",interpolation="nearest",norm=matplotlib.colors.SymLogNorm(0.01*vmax,vmin=-vmax,vmax=vmax) )
        plt.colorbar(img)
        plt.show()
    
    def load_npy(self,iso,MT1,MT2=None):
        if isinstance(iso,int):
            iso=str(iso)
        if MT2 is None:
            MT2=MT1
        return np.load(os.path.join(self.path_npy,f"{iso}-{MT1}-{iso}-{MT2}.npy"))
    
             
    def get_PSD(self,MTs):
        Cov={}
        for iso in self.iso:
            str_iso=str(iso)[:-1]
            Cov[iso]=np.zeros((len(MTs)*self.neg,len(MTs)*self.neg))
            for i,MT1 in enumerate(MTs):
                iStart1 = i*self.neg   # indices for M_sigma
                iEnd1 = (i+1)*self.neg
                for j,MT2 in enumerate(MTs):
                    iStart2 = j*self.neg    # indices for M_sigma
                    iEnd2 = (j+1)*self.neg
                    file_name = f"{str_iso}-{MT1}-{str_iso}-{MT2}.npy"
                    file = os.path.join(self.path_npy, file_name)

                    # File that is the opposite transpose
                    file_name_opp = f"{str_iso}-{MT2}-{str_iso}-{MT1}.npy"

                    file_opposite = os.path.join(self.path_npy,  file_name_opp)
                    mat=np.zeros((self.neg,self.neg))
                    if os.path.exists(file):
                        mat=np.load(file)
                        print(iStart1,iEnd1)
                        Cov[iso][iStart1:iEnd1, iStart2:iEnd2] = mat
                    elif os.path.exists(file_opposite):  # if the transpose matrix exists, load it and transpose it
                        mat = np.load(file_opposite).T
                        Cov[iso][iStart1:iEnd1, iStart2:iEnd2] = mat
                    else:
                        pass
            plt.figure()
            plt.title(iso)
            img=plt.imshow(Cov[iso])
            plt.colorbar(img)
            plt.savefig(f"{iso}.png")
            print(iso,psd_closeness(Cov[iso]))
        return Cov