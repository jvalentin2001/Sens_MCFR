import numpy as np
from math import log10, floor
import re, os, time, warnings, json, serpentTools
from serpentTools.settings import rc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib import colors
import csv
from utils import *



class MCFR():
    def __init__(self,e_cl,e_U_stock=0.1975,lib_cl="ENDF8",lib_all="ENDF8",pop=1e5,active=500,inactive=25,reactor_type="MCFR_C",get_flux=False,prefix=""):
        """  
          Args:
            e_cl (float): Chlorine weight enrichment
            e_U_stock (float): Uranium stock (in line feeding) enrichment
            lib_cl (str, optional): Nuclear library for Cl35 . Defaults to "ENDF8".
            lib_all (str, optional): Nuclear library for all the other isotopes. Defaults to "ENDF8".
            pop (int, optional): neutron population. Defaults to 1e5.
            active (int, optional): number of active batche. Defaults to 1e3.
            inactive (int, optional): number of inactive batches. Defaults to 50.
            reactor_type (str, optional): reactor type for the template file to read and parameters from mat_prop.json. Defaults to MCFR_C.
            get_flux (bool, optional): wheter or not to get flux profile from fuel. Defaults to False.
            prefix (str, optional): name prefix for folder generation. Defaults to ""
        """
        with open('template/properties.json', 'r') as file:
            data = json.load(file)
        self.reactor_type=reactor_type
        self.param=data["mat_prop"][reactor_type]
        
        self.e_cl=self.check(e_cl) #weight enrichment
        self.e_U=self.check(self.param["initial_enrich_U"]) #weight enrichement
        self.U_cl_split=self.check(self.param["U_cl_split"]) # percentage of UCl3 in molar %
        self.e_U_stock=self.check(e_U_stock)
        self.lib_cl=lib_cl
        self.lib_all=lib_all
        self.pop=int(pop)
        self.active=int(active)
        self.inactive=int(inactive)
        self.get_flux=get_flux
        self.prefix=prefix
        path_lib="\"/global/home/groups/co_nuclear/serpent/xsdata/"
        self.label_lib={"ENDF7":{"mat":".09c","path":path_lib+"endfb7/sss_endfb7u.xsdata\""},"ENDF8":{"mat":".02c","path":path_lib+"endf8/endf8.xsdata\""},"LANL":{"mat":".02l","path":path_lib+"LANL_Cl35/LANL_Cl35.xsdata\""}}
        self.temp_salt=self.param["temp"] #temperature in K of the salt
        
        #data for composition 
        comp_fuel=[92235,92238,11023,17037,17035]
        self.M_salt=np.array([int(str(x)[-3:]) for x in comp_fuel]) #molar mass for U235, U238, Na23, Cl37, Cl35
        self.mat_salt=self.gen_mat_SERPENT(comp_fuel)
        
        #cladding composition
        n_clad= self.param["n_clad"] #atomic percentage
        self.m_clad=np.array([x * y for x, y in zip(n_clad, self.param["M_clad"])]) # mass percentage clad Ico625
        self.m_clad=self.m_clad/np.sum(self.m_clad)
        self.mat_clad=self.gen_mat_SERPENT(self.param["mat_clad"])
        
        #reflector composition
        n_refl=self.param["n_refl"]#atomic percentage for the reflector
        self.m_refl=np.array([x * int(str(y)[-3:]) for x, y in zip(n_refl, self.param["mat_refl"])])#  mass percentage relfector
        self.m_refl=self.m_refl/np.sum(self.m_refl)
        self.mat_refl=self.gen_mat_SERPENT(self.param["mat_refl"])
        self.rho=np.array([self.param["rho_refl"],self.salt_density(),self.param["rho_clad"]]) #density in g/cm3 for reflector, salt, cladding 
        self.ratio_reffuel=np.array([0.85,0.1,0.05]) #volume percentage reflector, salt, cladding  in fuel reflector
        self.vol_salt=self.param["vol_salt"]
        self.units={"adens":"/b-cm","mdens": "g/cm3","a":"bq","ingTox":"Sv","inhTox":"Sv"}
        
        #virtual attributes to be attributed in the child class, paths, also generation of composition
        self.gen_comp()
        self.folder_name=None
        self.file_name=None
        self.out_path=None
        self.exec_name=None
        self.template_folder="template"
        self.ssspath_template=os.path.join(self.template_folder,self.param["template"])

    def check(self,e):
        """checks if percentage between 0 and 1
        """
        if 100>e>1.:
            return e/100.
        elif e<1:
            return e
        else:
            ValueError("Make sure enrichment is in percent or between 0 and 1")
            
    def gen_mat_SERPENT(self,mats):
        """generates the string input array for the material def in SERPENT

        Args:
            mats (list): list of the material definition using the ZA format (i.e. U235 is 92235)

        Returns:
            list: list of the material definition with the librairies associated 
        """
        matname=[]
        for mat in mats:
            if mat==17035:
                add=self.label_lib[self.lib_cl]["mat"] 
            elif mat%1000==0:
                add=self.label_lib["ENDF7"]["mat"]
            else:
                add=self.label_lib[self.lib_all]["mat"]
            matname.append(str(mat)+add) 
            
        return matname
    
    def salt_density(self):
        """generate density based on A-B*T(K) equation given in MCFR design
        Returns:
            float: density [g/cm^3]
        """
        rho=(4.2126e3-1.0686*self.temp_salt)*1e-3
        return round(rho, 3-int(floor(log10(abs(rho))))-1)
    
    def gen_comp(self):
        """generate composition for all the different materials to be used in the input file
        """
        
        self.N_salt,self.m_salt=self.comp_ucl_nacl(self.e_U,self.e_cl) #relative atomic concentration for salt, relative mass concentration for salt
        
        #stock salt composition
        self.N_salt_stock,self.m_salt_stock=self.comp_ucl_nacl(self.e_U_stock,self.e_cl)
        
        #reflector mixed with fuel and cladding composition
        prod=self.rho*self.ratio_reffuel
        m_reffuel=np.concatenate((self.m_refl*prod[0],self.m_salt*prod[1],self.m_clad*prod[2]))
        self.m_reffuel=m_reffuel/np.sum(m_reffuel)
        self.mat_reffuel=np.concatenate((self.mat_refl,self.mat_salt,self.mat_clad))
        
        self.rho_reffuel=np.sum(prod) #density of reffuel
        
    def comp_ucl_nacl(self,e_U,e_cl):  
        """generate the composition for molar and mass fraction for the Ucl3 NaCl mixture (with the split obtained from class attributes)
        for a given weight enrichment of Uranium and Cl

        Args:
            e_U (float): weight enrichment Uranium
            e_cl (float): weight enrichment Cl
        """
        def em_to_ea(M_majo,M_enrich,em):
            """transform an mass enrichment to an atomic enrichment

            Args:
                M_majo (float): mass number for the majorant isotope (e.g. 238)
                M_enrich (float): mass number for the isotoped enriched
                em (float): enrichment in mass
            """
            return 1/(1+M_enrich*(1-em)/(M_majo*em))   
         
        ea_U=em_to_ea(self.M_salt[1],self.M_salt[0],e_U) #molar enrichment U
        ea_cl=em_to_ea(self.M_salt[4],self.M_salt[3],e_cl) #molar enrichment Cl
        Ncl=(1+2*self.U_cl_split)/self.U_cl_split #atomic density Cl [a.u.]
        N_salt=np.array([ea_U,1-ea_U,(1-self.U_cl_split)/self.U_cl_split,ea_cl*Ncl,(1-ea_cl)*Ncl])
        m_salt=N_salt*self.M_salt
        N_salt=N_salt/np.sum(N_salt) #relative atomic concentration for salt
        m_salt=m_salt/np.sum(m_salt) #relative mass concentration for salt
        return N_salt, m_salt
    
    def gen_serpent(self):
         #the placeholder names to be searched in template file
        sss_str=["Replacepop","Replacepath","ReplaceT","Replacelib","Replacevol","Replaceflux"]
        prefs=["r","f","c","s","rf",] #prefix for each material being reflector, fuel, clad, stock, reffuel
        suffixes=[".1",".2",".3"] #for density, temperature and composition
        for prefix in prefs:
            for i in range(1,4):
                sss_str.append(f"Replace{prefix}.{i}")
        
        #variables to replace the placeholders
        param=[self.pop,self.active,self.inactive]   #running parameters
        path_lib=[self.label_lib[self.lib_cl]["path"]]
        if self.lib_cl!=self.lib_all:
            path_lib.append(self.label_lib[self.lib_all]["path"])
        
        #data to fill in material definition serpent
        temp=[self.temp_salt,self.temp_salt,self.temp_salt,self.temp_salt,self.temp_salt] #temperature of refl, fuel, cladding, uranium stock, reflfuel
        rho=np.concatenate((-1*self.rho,[-1*self.rho[1]],[-1*self.rho_reffuel])) #density of ...
        # Mat card with the atomic or massique concentrations 
        mat=[np.vstack((self.mat_refl,-1*self.m_refl)).T,np.vstack((self.mat_salt,self.N_salt)).T,
             np.vstack((self.mat_clad,-1*self.m_clad)).T,np.vstack((self.mat_salt,self.N_salt_stock)).T,np.vstack((self.mat_reffuel,-1*self.m_reffuel)).T] 
        raw=[]
        raw.append(rho)
        raw.append(temp)
        raw.append(mat)
        #reformat the data into a list in the correct order
        variables=[array_txt(param),array_txt(path_lib),f"{self.temp_salt}",self.label_lib[self.lib_all]["mat"],self.vol_salt]
        if self.get_flux:
            variables.append(f"det flux n dm fuel de ANL33 dv {self.vol_salt} ")
        else:
            variables.append("")
        for i in range(len(prefs)):
            for j in range(len(suffixes)):
                if j==2:
                    variables.append(array_txt(raw[j][i]))
                else:
                    variables.append(raw[j][i])
        return sss_str, variables
    
    def get_exec(self,time,nodes,partition):
        #list the fillwords to be replaced in the template
        sss_str=["Name","REPLACEPART","REPLACEn","Replacetime","ReplaceCPU","REPLACESERPENT"]

        cpu=partitions[partition]["cores_per_node"]   
        nodes_max=partitions[partition]["nodes"]      
        if nodes>nodes_max:
            raise ValueError(f"the number of nodes you asked, {nodes}, is bigger than the number of available nodes in {partition} which has {nodes_max} nodes")     
        str_time=f"{int(np.floor(time))}:{int(60*(time-np.floor(time)))}:00"
        #variables to replace the fillwords, in order: out log file, terminal input line, number of nodes 
        variables=[self.file_name,partition,nodes,str_time,cpu,f"mpirun -np $SLURM_JOB_NUM_NODES --map-by ppr:$SLURM_TASKS_PER_NODE:node:pe=$SLURM_CPUS_PER_TASK $SERPENT_EXE -omp $SLURM_CPUS_PER_TASK {self.file_name}"]
        return sss_str, variables

    def simu_wait(self,o_path,number=1):
        def check_file_for_line(filename,number):
            """Check the given file for the specified line."""
            itter=0
            with open(filename, 'r') as file:
                for line in file:
                    if line.strip() == "Simulation aborted.":
                        raise RuntimeError("The simulation aborted")
                    if line.strip() == "Simulation completed.":
                        itter=itter+1
                        if itter>=number:
                            return True
            return False

        while not os.path.exists(o_path):
            time.sleep(1)

        while True:
            if check_file_for_line(o_path,number):
                print("Simulation completed.")
                break
            else:
                time.sleep(10)
                
    def write_to_file(self,sss_str,variables,template_paths,filename):
        """write a new file based on the templates

        Args:
            sss_str (list): list of name placers to be replaced
            variables (list): variables to replace the name placers
            template_path (list): list of templates to be used to write a new file
            filename (str): name of new file to be written
        """
        saved_input=[]
        if isinstance(template_paths,str):
            template_paths=[template_paths]
        for template_path in template_paths:
            with open(template_path,"rt") as template:
                for line in template:
                    saved_input.append(line) 
        
        #Replace the fillwords with the variables
        for i in range(0,len(saved_input)):
            for n in range(0,len(variables)):
                s = re.sub(sss_str[n],str(variables[n]),saved_input[i])
                if s is not None:
                    saved_input[i]=s
                    
        #write into a copy of template file
  
        print(f"The following input file has been written: {filename} in the following directory: {self.folder_name}")
        with open(os.path.join(self.folder_name,filename), "w") as file:
            file.write("".join(saved_input))

    def run_serpent(self,nodes,partition):
        """run the generated SERPENT file

        Args:
            nodes (int, optional): number of nodes to be run on the cluster. Defaults to 1.
        """
         # List all files in the directory
        files = os.listdir(self.folder_name)
        self.gen_exec(nodes=nodes,partition=partition)
        # Iterate through files and delete those starting with the specified prefix
        for file in files:
            if file.startswith(self.file_name+"_") or file.startswith(self.file_name+"."):
                file_path = os.path.join(self.folder_name, file)
                os.remove(file_path)
        # os.system("nohup")
        o_path=f"{self.out_path}.o"
        if os.path.exists(o_path):
            os.remove(o_path)
        os.system(f"cd {self.folder_name}; sbatch {self.exec_name} ")
        self.simu_wait(o_path) 
                 
    def extract_res_m(self,variables):
        """extract from the res.m file the desired variables.

        Args:
            variables (list or str): list or single value str of variables to be considered for extraction, such as absKeff, conversionRatio 
            (note format is first word no caps, and second word starts with a cap)

        Returns:
            list: the variable output (could be a single value if no burnup)
        """
        resFile=f"{self.out_path}_res.m"
        res = serpentTools.read(resFile)
        var=[]
        for variable in variables:
            if len(res.resdata[variable])==2:
                # if not burnup calculation then vector of dimension 1.
                var.append(res.resdata[variable][:2])
            else:
                var.append(res.resdata[variable][:,:2])
        return np.array(var)

    def plot_pathgen(self,plot_dir):
        plot_path=os.path.join(self.folder_name,plot_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        return plot_path
        
    def extract_flux(self,plot_dir="",BU_years=[0]):
        """extract the flux 

        BU_points (int,optional): the number of plots of spectrum you want. Default 1 for the first BU step or static case
        """
        resFile=f"{self.out_path}_dep.m"
        dep = serpentTools.read(resFile)
        BU=dep.days/365

        def find_nearest_index(number):
            nearest_index = np.abs(BU - number).argmin()
            return nearest_index

        plt.figure()
        for step in BU_years:
            ind=find_nearest_index(step)
            resFile=self.out_path+f"_det{ind}.m"
            res=serpentTools.read(resFile,reader="det")
            data=res.detectors["flux"]
            flux=data.tallies
            flux_s=data.errors
            E=data.grids["E"][:,1]
            plt.step(E,flux,where="post",label=f"{step} years")
            plt.fill_between(E,flux-flux*flux_s,flux+flux*flux_s,alpha=0.4,step="post")
            #print(f"{np.sum(flux):.3e} +/- {np.sum(flux*flux_s):.3e}")

        plt.xscale('log')
        plt.yscale('linear')
        plt.xlabel("BU years")
        plt.ylabel("flux [cm$^{-2}$s$^{-1}$]")
        plt.title(f'Evolution of flux for ${{\lambda_{{in}}}}$={self.mflow_in} /s, Cl_e={self.e_cl*100} w%',fontsize = 10)
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(self.folder_name,plot_dir,f"flux_{self.file_name}.png"))
        plt.close()
        
        return flux


class Static(MCFR):
    def __init__(self,e_cl,e_U_stock=0.1975,lib_cl="ENDF8",lib_all="ENDF8",pop=1e5,active=500,inactive=25,reactor_type="MCFR_C",get_flux=False,prefix=""):
        #initialise the mother class MCFR
        super().__init__(e_cl,e_U_stock,lib_cl,lib_all,pop,active,inactive,reactor_type,get_flux,prefix)
        
        #create the name and path for the input and output file inside a common directory
        if self.lib_cl==self.lib_all:
            self.folder_name=f"{self.prefix}A_Cl_{self.lib_cl}_{self.reactor_type}"
        else:
            self.folder_name=f"{self.prefix}A_{self.lib_all}_Cl_{self.lib_cl}_{self.reactor_type}"
            
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name) 
        self.file_name=f"{self.prefix}Cl_{self.e_cl}.txt"
        self.out_path=os.path.join(self.folder_name,self.file_name)
        self.template_path=os.path.join(self.template_folder,"temp_execute.sub")
        self.gen_serpent()    
        
    def gen_serpent(self):
        sss_str,variables=super().gen_serpent()
        

        self.write_to_file(sss_str,variables,self.ssspath_template,self.file_name)
        
    def gen_exec(self,time=0.5,nodes=1, partition="savio3"):
        """generate execute file

        Args:
            time (int, optional): time of simulation. Defaults to 1.
            nodes (int, optional): number of nodes. Defaults to 1.
            partition (str, optional): which cluster partition to use. Defaults to "savio3".
        """
        sss_str,variables=super().get_exec(time,nodes,partition)
        self.exec_name=f"execute_{self.lib_cl}_{self.e_cl}.sub"
        self.write_to_file(sss_str,variables,self.template_path,self.exec_name)
        
    def run_serpent(self, nodes=1, partition="savio3"):
        print(f"Starting Static simulation for Cl35-lib: {self.lib_cl}, for All-lib {self.lib_all} and with Cl enrichment {self.e_cl*100} w% ")
        super().run_serpent(nodes, partition)
        
    def extract_flux(self,plot_dir=""):
        resFile=self.out_path+f"_det0.m"
        res=serpentTools.read(resFile,reader="det")
        data=res.detectors["flux"]
        E=data.grids["E"][:,1]
        du=np.log(E/data.grids["E"][:,0])
        flux=data.tallies/du
        flux_s=data.errors
        plt.figure()
        plt.step(E*1e6,flux,where="post")
        plt.fill_between(E*1e6,flux-flux*flux_s,flux+flux*flux_s,alpha=0.4,step="post")
        plt.xscale('log')
        plt.yscale('linear')
        plt.xlabel("E [eV] ")
        plt.ylabel("flux per unit lethargy [n.cm$^{-2}$s$^{-1}$]")
        plt.title(f'Evolution of flux for Cl_e={self.e_cl*100} w%',fontsize = 10)
        plt.grid()
        plot_path=self.plot_pathgen(plot_dir)        
        plt.savefig(os.path.join(plot_path,f"flux_{self.file_name}.png"))
        plt.close()
        
        return flux,E


class Depletion(MCFR):
    def __init__(self,mflow_in,mflow_out=None,e_cl=0.75,e_U_stock=0.1975,lib_cl="ENDF8",lib_all="ENDF8",pop=1e5,active=500,inactive=25,reactor_type="MCFR_C",get_flux=False,prefix="",BU_years=60,restart=False):
        """_summary_

        Args:
            mflow_in (float): mflow rate inside 
            mflow_out (float, optional): mflow rate in over flow system. Defaults to None.
            e_cl (float): Chlorine weight enrichment
            e_U (float): Uranium weight enrichment
            U_cl_split (float): molar compsotion of UCl3-NaCl
            e_U_stock (float): Uranium stock (in line feeding) enrichment
            lib_cl (str, optional): Nuclear library for Cl35 . Defaults to "ENDF8".
            lib_all (str, optional): Nuclear library for all the other isotopes. Defaults to "ENDF8".
            BU_years (float,optional): Number of Burnup years to be run. Defaults to 30 years
            temp (int, optional): temperature in K. Defaults to 973.15K.
            pop (int, optional): neutron population. Defaults to 1e5.
            active (int, optional): number of active batche. Defaults to 1e3.
            inactive (int, optional): number of inactive batches. Defaults to 50.
            get_flux (bool, optional): wheter or not to get flux profile from fuel. Defaults to False.
            prefix (str, optional): name prefix for folder generation. Defaults to ""
            BU_years (int, optional): number of years of Burnup. Defaults to 60.
            restart (bool, optional): Whether or not to generate restart file (used for Sensitivity). Defaults to False.
        """
        super().__init__(e_cl,e_U_stock,lib_cl,lib_all,pop,active,inactive,reactor_type,get_flux,prefix)
        self.BU_years=BU_years
        self.mflow_in=mflow_in
        self.mflow_out=mflow_out
        self.Bu_steps=self.gen_BU_steps()
        
        if self.lib_cl==self.lib_all:
                self.folder_name=f"{self.prefix}A_Cl_{self.lib_cl}_{self.e_cl}_dep_{self.reactor_type}"
        else:
                self.folder_name=f"{self.prefix}A_{self.lib_all}_Cl_{self.lib_cl}_{self.e_cl}_dep_{self.reactor_type}"   
        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name) 
        self.file_name=f"{self.prefix}Cl_{self.e_cl}_mflow_{self.mflow_in}.txt"
        self.out_path=os.path.join(self.folder_name,self.file_name)
        self.template_path=os.path.join(self.template_folder,"temp_execute.sub")
        
        if self.mflow_out is None:
            self.mflow_out=self.get_out_flow()
            print(f"The necessary over flow rate for an inflow of {self.mflow_in} /s is of {self.mflow_out:.3e} /s")
        
        self.gen_serpent(gen_restart=restart)    
        
    def gen_serpent(self,gen_restart=False):
        sss_str,variables=super().gen_serpent()
        sss_str=np.concatenate((sss_str,["Replace_Uin","Replace_over","Replacerestart","ReplaceBU"]))
        variables=np.concatenate((variables,[self.mflow_in,self.mflow_out,int(gen_restart),array_txt(self.Bu_steps)]))

        template_path=[self.ssspath_template,os.path.join(self.template_folder,"temp_dep")]
        self.write_to_file(sss_str,variables,template_path,self.file_name)
    
    def gen_exec(self,time=3,nodes=1, partition="savio3"):
        """generate execute file

        Args:
            time (int, optional): time of simulation. Defaults to 3.
            nodes (int, optional): number of nodes. Defaults to 1.
            partition (str, optional): which cluster partition to use. Defaults to "savio3".
        """
        sss_str,variables=super().get_exec(time,nodes,partition)
        self.exec_name=f"execute_{self.lib_cl}_{self.mflow_in}.sub"
        self.write_to_file(sss_str,variables,self.template_path,self.exec_name)
        
    def gen_BU_steps(self):
        """Generate the BU steps through a simple doubling the of BU step interval every 5 steps to reduce the number BU steps"""
        total_days=self.BU_years*365
        Bu_steps = [10]
        interval = 100
        while sum(Bu_steps) + interval <= total_days:
            for _ in range(5):
                Bu_steps.append(interval)
                if np.sum(Bu_steps)>total_days:
                    break
            interval += 200
        return Bu_steps
            
    def run_serpent(self, nodes=1, partition="savio3_bigmem"):
        print(f"Starting Depletion simulation for Cl35-lib: {self.lib_cl}, for All-lib {self.lib_all} and with Cl enrichment {self.e_cl*100} w% and mflow={self.mflow_in:.3e} /s ")
        super().run_serpent(nodes, partition)
        
    def extract_dep_m(self,isotopes,variable,plot_dir="",do_plot=True,logy=True):
        """
        Extracts the evolution of the composition of the specified isotope in the fuel
        in function of burnup from a SERPENT results file.

        Parameters:
            results_file (str): Path to the SERPENT results file (.res).
            isotopes (list): List of isotope names (e.g., ['U235', 'Pu239']).

        Returns:
            Tuple of arrays: Tuple containing burnup values and composition of the isotope.
        """
        
        resFile=f"{self.out_path}_dep.m"
        dep=serpentTools.read(resFile)
        plot_path=self.plot_pathgen(plot_dir)
        fuel=dep.materials["fuel"]
        unit=self.units[variable]     
        BU = dep.days
        compositions = [fuel.getValues("days",variable,names=isotope) for isotope in isotopes] #{isotope: fuel.getValues("days",variable,names=isotope) for isotope in isotopes}
        if do_plot:
            plt.figure()
            for i in range(len(isotopes)):
                plt.plot(BU/365, compositions[i][0], label=isotopes[i])
                #plt.plot(BU[:-1]/365, np.diff(compositions[i][0])/np.diff(BU/365),label=isotopes[i])
                #plt.fill_between(BU,compositions[i][0]-compositions[i][1],compositions[i][0]+compositions[i][1],alpha=0.4)
            if logy:
                plt.yscale("log")
            #plt.plot(BU/365, compositions[0][0]/compositions[-1][0]+compositions[1][0]/compositions[-1][0])
            plt.xlabel('Burnup year')
            plt.ylabel(f'{variable} [{unit}]')
            plt.title(f'Evolution of {variable} for ${{\lambda_{{in}}}}$={self.mflow_in} /s, Cl_e={self.e_cl*100} w%',fontsize = 10)
            plt.legend()
            plt.grid()
            if variable=="adens":
                plt.ylim(bottom=1e-9)
            plt.savefig(os.path.join(plot_path,f"{variable}_{len(isotopes)}iso_BU_{self.file_name}.png"))
            plt.close()

        return compositions, BU

    def extract_res_m(self,var_names,plot_dir="",do_plot=True):
        resFile=f"{self.out_path}_dep.m"
        dep=serpentTools.read(resFile)
        plot_path=self.plot_pathgen(plot_dir)
        BU=dep.days
        
        if isinstance(var_names,str):
            var_names=[var_names]
        var=super().extract_res_m(var_names)


        if do_plot:
            plt.figure()
            for j in range(len(var_names)):
                label=f"{var_names[j]} "
                plt.plot(BU/365,var[j,:,0],label=label)
                plt.fill_between(BU/365,var[j,:,0]-var[j,:,1],var[j,:,0]+var[j,:,1],alpha=0.4)
            plt.grid()
            title=f"Macro variables for {self.lib_cl}/{self.lib_all} with Cl37={self.e_cl}w% & ${{\lambda_{{in}}}}$={self.mflow_in} /s"
            plt.title(title,fontsize=10)
            plt.xlabel('Burnup years')
            if not len(var_names)==1:
                plt.ylabel("a.u.")
                plt.legend()
            else:
                plt.ylabel(label)
            plt.savefig(os.path.join(plot_path,f"Bu_mflow_{*var_names,}.png"))
            plt.close()
        return var,BU
  
    def get_out_flow(self):
        """generate the over flow rate based on the mass flow rate in given and the enrichment, a quick Burnup simulation is run, if such a file doesn't exist, to find the gas out flow rate,from which the 
        overflow rate can be calculated to approximately keep the mass of the fuel constant. 
        
        return
        float: the time constant for the overflow mass flow rate [/s]
        """
        def get_mflow_out(dep,lam_in):
            fuel=dep.materials["fuel"]
            BU=dep.days
            
            gas=dep.materials["offgastankcore"]
            stock=dep.materials["U_stock"]
            mgas=np.array(gas.getValues("days","mdens",names="total"))
            mf=np.array(fuel.getValues("days","mdens",names="total"))
            mstock=np.array(stock.getValues("days","mdens",names="total"))
            dmgas=np.diff(mgas)
            dt=np.diff(BU)
            Vf=self.vol_salt
            Vgas=1e3
            lam=[]
        
            #finding the time constant for overflow using formula 𝜆_𝑜=𝜆_𝑠−(Δ𝜌_𝑔𝑎𝑠)/(𝜌_𝑓𝑢𝑒𝑙 Δ𝑡).𝑉_𝑔𝑎𝑠/𝑉_𝑓𝑢𝑒𝑙  
            for i in range(len(dmgas[0])):
                lam_over=(lam_in*mstock[0,0]-dmgas[0,i]*Vgas/(dt[i]*24*3600*Vf))/mf[0,0]
                lam.append(lam_over)
            return np.mean(lam[int(0.1*len(lam)):])
        
        lam_in=self.param["ref_mflow_in"]
        gas_rate=Depletion(lam_in,self.param["ref_mflow_out"],self.e_cl,self.e_U_stock,self.lib_cl,self.lib_all,1e4,25,reactor_type=self.reactor_type,prefix=self.prefix,BU_years=200,restart=False)
        if not os.path.exists(gas_rate.out_path+"_dep.m"):
            gas_rate.run_serpent()
        gas_rate.extract_dep_m(["total"],"mdens")

        dep=serpentTools.read(gas_rate.out_path+"_dep.m")
        return get_mflow_out(dep,self.mflow_in)


class Sensitivity(MCFR):
    def __init__(self,sens_iso="all",sens_MT="all_MT",sens_resp="keff",mflow_in=None,mflow_out=None,equi_comp=False,e_cl=0.75,e_U_stock=0.1975,lib_cl="ENDF8",lib_all="ENDF8",pop=1e5,active=500,inactive=25,reactor_type="MCFR_C",get_flux=False,prefix="",BU_years=60):
        """_summary_

        Args:
            sens_iso (list or int or str): isotopes of interest for sensitivity analysis if list in zaid format (dont forget 0 at the end for ground state), if int number of isotopes in decreasing adens, if all use all isotopes. default all
            sens_MT (list): Reactions of interest for sensitivity analysis, can be a list of MT number or sum reactions, or "all" for all sum reactions, or "all_MT" for all relevant MT numbers. default to all.
            sens_resp (str, optional): parameter to investigate the sensitivity. Defaults to "keff".
            equi_comp (bool, optional): Whether you are looking at initial composition (False) or equilibrium composition (True). Defaults to False.
            other parameters described in other class definitions
        """
        super().__init__(e_cl,e_U_stock,lib_cl,lib_all,pop,active,inactive,reactor_type,get_flux,prefix)
        
        self.sssinput_MT=self._process_sens_MT(sens_MT) #check whether the list is MT or RX or all or all_MT
        
        if mflow_in is not None and not equi_comp:
            warnings.warn("make sure equi_comp is true to actually run the depletion calculation till equilibrium composition is obtained")
        self.equi_comp=equi_comp
        self.sens_resp=sens_resp
        
        if self.lib_cl==self.lib_all:
                self.folder_name=f"{self.prefix}sens_A_Cl_{self.lib_cl}_{self.e_cl}_dep_{self.reactor_type}"
        else:
                self.folder_name=f"{self.prefix}sens_A_{self.lib_all}_Cl_{self.lib_cl}_{self.e_cl}_dep_{self.reactor_type}"   
        
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name) 
            
        
        add_txt=""
        if self.equi_comp:
        #create instance of Depletion to obtain the restart file to be read and used for sensitivity if looking at equilibirum concentration
            dep=Depletion(mflow_in,mflow_out,e_cl,e_U_stock,lib_cl,lib_all,1e5,100,25,reactor_type,get_flux,prefix+"sens_",BU_years,restart=True)
            self.restart_file=dep.file_name+".wrk"
            #if restart file doesn't exist run depletion
            if not os.path.exists(os.path.join(self.folder_name,self.restart_file)):
                dep.run_serpent(nodes=4,partition="savio2_bigmem")
            dep.extract_dep_m(["total"],"mdens","plot")
            add_txt="equi"
            resFile=f"{dep.out_path}_dep.m"
            dep=serpentTools.read(resFile)
            fuel=dep.materials["fuel"]
            names=fuel.zai
            iso=fuel.adens
            names = [str(num)  for num in names]
            self.iso_list_sorted=np.array(sorted(zip(iso[:,-1],names),reverse=True))

        #create the list of isotopes for senstivity analysis
        if isinstance(sens_iso,list):
            self.sens_iso=sens_iso
        elif isinstance(sens_iso,int):
            if not equi_comp:
                raise ValueError("Cannot ask for number of isotopes if not the equilibirium composition mode")
            self.sens_iso=self.iso_list_sorted[:sens_iso,1]

        elif sens_iso=="all":
            if equi_comp:
                #if "all" option selected for isotopes take top 50 isotopes from the equilibrium composition
                self.sens_iso=self.iso_list_sorted[:70,1]
                #self.sens_iso=np.append(self.sens_iso,["80160"])
                index = np.where(self.sens_iso=="0")
                self.sens_iso[index]="total" #the isotope called 0 corresponds to total perturbation

            else:
                #if "all" option but not initial composition all the isotopes present in composition used
                self.sens_iso=[str(num).partition(".")[0] + '0' for num in self.mat_reffuel]
               #self.sens_iso=np.concatenate((["total"],self.sens_iso))

        else:
            raise ValueError("Make sure the sens_iso variable is either a list, an int, or the str all,")
        
        
        self.sens = None #list of sensitivities in energy in format (len(zaid)*len(RX),33) for 33 Energy group.
        self.sens_s = None #absolute uncertainty in sensitivity
        self.zaid_MT = None #a list of format [[zaid1,RX1],[zaid1,RX2],...].
        self.zaid_MT_intEsens = None #the list of sensitivity for the integrated energy sensitivities in the format [[zaid1,RX1,sens11, sens11_s],[zaid1,RX2,sens12,sens12_s],...]

        self.file_name=f"{self.prefix}sens{add_txt}_{self.sens_resp}_{len(self.sens_iso)}iso{self.str_MT}.txt"
        self.out_path=os.path.join(self.folder_name,self.file_name)
        self.template_path=os.path.join(self.template_folder,"temp_execute.sub")

        self.gen_serpent()

    def _process_sens_MT(self,sens_MT):
        '''
        Process the sens_MT argument.

        Args:
            sens_MT (str or list): Argument specifying which reactions or MTs to process.

        Returns:
            processed_sens_MT (list): List of reactions or MTs to process.

        Raises:
            ValueError: If the sens_MT argument is not valid.
        '''

        # Define a set of all valid reactions
        valid_reactions = {"2","4","16","18","28","102","103","104","105","106","107","111"}
        valid_reactions_RX={"ela","sab","inl","capt","fiss","nxn"}

        # Check if sens_MT is "all"
        if sens_MT == "all":
            self.sens_MT=sens_MT
            self.str_MT=""
            self.is_MT=False
            return "all"

        # Check if sens_MT is "all_MT"
        elif sens_MT == "all_MT":
            self.sens_MT=valid_reactions
            self.str_MT="_MTall"
            self.is_MT=True
            return  "mtlist "+array_txt(self.sens_MT) 

        # Check if sens_MT is a list of reactions
        elif isinstance(sens_MT, list):
            if all(isinstance(item,int) for item in sens_MT) or all(item.isdigit() for item in sens_MT):
                if all(mt in valid_reactions for mt in sens_MT):
                    # Check if all elements in the list are valid reactions for MT
                    self.sens_MT=sens_MT
                    self.str_MT=f"_MT{len(self.sens_MT)}"
                    self.is_MT=True
                    return "mtlist "+array_txt(self.sens_MT)
                else:
                    raise ValueError(f"Invalid MT in the list, must be from {valid_reactions} or simple  \"all\" or \"all_MT\"")    
            elif all(item.isalpha() for item in sens_MT):
                # Check if all elements in the list are valid reactions for RX
                if all(reaction in valid_reactions_RX for reaction in sens_MT):
                    self.sens_MT=sens_MT
                    self.str_MT=f"_RX{len(self.sens_MT)}"
                    self.is_MT=False
                    return "realist "+array_txt(self.sens_MT) 
                else:
                    raise ValueError(f"Invalid reactions in the list, must be from {valid_reactions_RX} or simply \"all\" or \"all_MT\"")

        else:
            raise ValueError("Invalid sens_MT argument.")

    def gen_serpent(self):
            
        sss_str,variables= super().gen_serpent()  
        sss_str=np.concatenate((sss_str,["Replaceresp","Replacezailist","Replacemtlist","Replacefilerestart","ReplaceBU"]))
        variables=np.concatenate((variables,[self.sens_resp,array_txt(self.sens_iso),self.sssinput_MT,"",""]))
        if self.equi_comp:
            variables[-2]=f"set rfr continue {self.restart_file}"
            #single burnup step just to load nuclear data in memory, in practice only final BU step used from restart file used.
            variables[-1]=f"dep daystep 1" 
            
        template_path=[self.ssspath_template,os.path.join(self.template_folder,"temp_sens")]
        self.write_to_file(sss_str,variables,template_path,self.file_name)
 
    def gen_exec(self,time=2,nodes=1, partition="savio3"):
        """generate execute file

        Args:
            time (int, optional): time of simulation. Defaults to 1.
            nodes (int, optional): number of nodes. Defaults to 1.
            partition (str, optional): which cluster partition to use. Defaults to "savio3".
        """
        sss_str,variables=super().get_exec(time,nodes,partition)
        self.exec_name=f"execute_{self.lib_cl}_{len(self.sens_iso)}.sub"
        self.write_to_file(sss_str,variables,self.template_path,self.exec_name)
       
    def run_serpent(self, nodes=1, partition="savio3_bigmem",time=None):
        print(f"Starting Sensitivity simulation for Cl35-lib: {self.lib_cl}, for All-lib {self.lib_all} for the following isotopes {*self.sens_iso,} and the following reactions {*self.sens_MT,} ")
        super().run_serpent(nodes, partition,time)    
     
    def get_adjsens(self,zai,pert,material="total",integralE=False):
        """Get an individual sensitivity 

        Args:
            material (str): material. Defaults to total
            zai (str): zaid number int
            pert (str): perturbation cross section
            integralE (bool, optional): whether or not to have the integral perturbation over energy. Defaults to False.

        Returns:
            _type_: sensitivity and absolute uncertainty
        """
        if not integralE:
            ks=self.sens_file.sensitivities["keff"]
        else:
            ks=self.sens_file.energyIntegratedSens["keff"]
        if not isinstance(zai,int):
            if zai=="total":
                pass
            else:
                zai=int(zai)
        
        kslice = ks[
            self.sens_file.materials[material],  # index for sensitivity due to all materials
            self.sens_file.zais[zai],  # index for sensitivity due to isotope 
            self.sens_file.perts[pert],  # index for sensitivity due to fission xs
        ]
        # Normalize per unit lethargy
        if integralE:
            value=kslice[0]
            unc=np.abs(kslice[1]*value)
            return value, unc
        # expected value is in the 0 index 
        value = kslice[:, 0] 

        # Compute 1-sigma uncertainty
        unc = np.abs(kslice[:, 1]  * value)
        return value, unc
     
    def extract_sens_m(self,zais="all",perts="all",plot_dir="",do_plot=False):
        """extract the sensitivity data for pair of zais pert

        Args:
            zais (list): list of isotopes, either zaid or Cl-35 format
            pert (list): perturbation, that being MT number or name of sum reaction, goes in pair with zais (i.e. zai=[922380, 922380], pert=["capt","fiss"] or ["2","16"])
            plot_dir (str, optional): directory towards which to plot. Defaults to "".
            do_plot (bool, optional): whether or not to plot. Defaults to True.

        Returns:
            _type_: _description_
        """
        self.sens_file=serpentTools.read(f"{self.out_path}_sens0.m")
        self.pert_list=list(self.sens_file.perts.keys())
        if zais=="all":
            zais=self.sens_iso
        if perts=="all":
            perts=self.pert_list
        else:
            perts=MT_to_serpent_MT(perts)
        
        self.sens=[]
        self.sens_s=[]  
        self.zaid_MT = []
        self.zaid_MT_intEsens = []
        for i in zais:
            for j in perts:
                value,unc=self.get_adjsens(i,j)
                if not (i=="total" or j=="total xs"):
                    self.sens.append(value)
                    self.sens_s.append(unc)
                    #generate a list of format [[zaid1,RX1],[zaid1,RX2],...].
                    self.zaid_MT.append([i, j.split()[1]])
                #combine the zaid_MT list format with the inegralE sensitivity coefficient with abs uncertainty
                self.zaid_MT_intEsens.append([ zai_to_nuc_name(i),str(sssmtlist_to_RXlist( [j])[0]),*self.get_adjsens(i, j,integralE=True)])
        # plotting energy dependent sensitivity profile
        print(np.shape(self.sens))
        self.sens=np.array(self.sens)
        self.sens_s=np.array(self.sens_s)
        if do_plot:
            plot_path=self.plot_pathgen(plot_dir)
            line_styles = ['-',":", '--', '-.',] 
            colors = plt.cm.jet(np.linspace(0, 1, len(zais)))       
            plt.figure()
            for i in range(len(zais)):
                color=colors[i]
                for j in range(len(perts)):
                    # Draw errorbars
                    # The energy vector has one additional entry, so we will instead drop the first item
                    # by slicing from the second position forward
                    plt.errorbar(
                        self.sens_file.energies[1:] * 1E6,  # convert to eV
                        list(self.sens[j+i*len(perts)][:]/self.sens_file.lethargyWidths),  # expected value
                        yerr=list(self.sens_s[j+i*len(perts)][:]/self.sens_file.lethargyWidths),   # uncertainty
                        drawstyle="steps-mid",   # step-plot
                        linestyle=line_styles[j % len(line_styles)],
                        label=f"{zais[i]}-{perts[j]}",
                        color=color
                    )
            # Format the plot
            plt.xscale("log")
            plt.xlim(10, self.sens_file.energies[-1]*1E6)
            # Major and minor grids for the log-scaled x axis
            plt.grid("x", which="both")
            plt.grid("y")
            plt.legend()
            plt.xlabel("Energy (eV)")
            plt.ylabel("Sensitivity per unit lethargy")
            plt.title(f"K-eff sensitivity")
            plt.savefig(os.path.join(plot_path,f"sens_iso{len(zais)}_pert{len(perts)}_MT{perts}"),dpi=400)
        return self.sens,self.sens_s,self.sens_file.energies,self.sens_file.lethargyWidths
    
    def check_run_extract(self):
        if self.sens is None:
            raise ValueError("make sure you run: extract_sens_m() before ranking them ")
        else:
            pass
    
    def rank_sens(self,reaction=None,isotope=None,do_plot=True):
        """Ranks the zaid-MT pair on their contribution to sensitivity (the sum of the energy bin sensitivity), if an isotope or a reaction is given only ranks for that
        specific instance for example if Cl-35 or 170350 is given then ranks all the reactions for Cl35,
        and if reaction is given as 2 then will rank all the elastic scattering from different isotopes.
        
        isotope (int/str): the inputs can be given in zaid, or name
        reaction (int/str): the name of the reaction can be mt number, sum reactions names (see serpent manual for examples: such as ela, fiss, capt)

        """
        self.check_run_extract()
        total_xs_list=self.zaid_MT_intEsens
        str_reaction="all"
        if reaction is not None:
            if isinstance(reaction,int) or reaction.isdigit():
                reaction=MTtoRX(str(reaction))
            total_xs_list = [row for row in self.zaid_MT_intEsens if row[1] == reaction]
            str_reaction=reaction
            title_str="Isotopes"
        if isotope is not None:
            if isinstance(isotope,int) or isotope.isdigit():
                isotope=zai_to_nuc_name(str(isotope))
            total_xs_list = [row for row in self.zaid_MT_intEsens if row[0] == isotope]
            str_reaction=isotope
            title_str="all reactions"
        if reaction is not None and isotope is not None :
            raise ValueError("cannot rank for single reaction and single isotope")
        
        if len(total_xs_list)==0:
            raise ValueError("The reaction or Isotope requested doesn't exist in the model, make sure it was given in the model initialisation, or was part of the all tag")
            
        #rank_sens=np.column_stack((self.zaid_MT,k_Eint,k_Eint_s))
        rank_sens=np.array(sorted(total_xs_list,key=lambda x: abs(float(x[2])),reverse=True))
        save_folder=f"rank_{self.sens_resp}_{len(self.sens_iso)}iso{self.str_MT}"
        plot_path=self.plot_pathgen(save_folder)
        with open(os.path.join(plot_path,f"sens_rank_{str_reaction}.csv"),"w",newline="") as file:
            writer = csv.writer(file,delimiter=";",)
            writer.writerow(['Isotope', 'Reaction', 'sensitivity', 'absolut uncertainty'])  # Write headers
            writer.writerows(rank_sens)
            
        #find the first index which is 0 or is 100 times smaller than the max, exclude the rest of the array from that point for the plot
        ind_0=  next((index for index, value in enumerate(np.float64(rank_sens[:,2])) if (value == 0 or np.abs(value/np.max(np.abs(np.float64(rank_sens[:,2]))))<1e-2)), None)
 
        if do_plot:
            plt.figure(figsize=(10, 6))
            if  reaction is not None:

                #check if the values are in MT format and transform them back to RX format for readability in plot

                plt.bar(rank_sens[:ind_0,0], np.float64(rank_sens[:ind_0,2]),yerr=np.float64(rank_sens[:ind_0,3]),bottom=0,align="center", alpha=0.7)
                plt.xlabel(f'{title_str}')
                plt.ylabel('Sensitivity')
                if self.equi_comp:
                    plt.title(f'Equilibrium composition sensitivity for {title_str} with {rank_sens[0,1]} xs')
                else:
                    plt.title(f'initial composition sensitivity of {title_str} to {rank_sens[0,1]} xs')
            elif isotope is not None:
                #find the first index which is 0 or is 100 times smaller than the max, exclude the rest of the array from that point for the plot

                plt.bar(rank_sens[:ind_0,1], np.float64(rank_sens[:ind_0,2]),yerr=np.float64(rank_sens[:ind_0,3]),bottom=0,align="center", alpha=0.7)
                plt.xlabel(f'Reactions')
                plt.ylabel('Sensitivity')
                if self.equi_comp:
                    plt.title(f'Equilibrium composition sensitivity for {isotope}')
                else:
                    plt.title(f'Initial composition sensitivity for {isotope}')
                
            else :
                rank_sens_rm=self.remove_iso_MT(rank_sens,"total","total xs")
                #if rank_sens_rm[0,1].split()[1].isdigit():
                #   rank_sens_rm= np.array([(*item[:1], MTtoRX(item[1].split()[1]), *item[2:]) for item in rank_sens_rm])

                name=['-'.join(map(str, row)) for row in rank_sens_rm[:ind_0,0:2]]
                plt.bar(name, np.float64(rank_sens_rm[:ind_0,2]),yerr=np.float64(rank_sens_rm[:ind_0,3]),bottom=0,align="center", alpha=0.7)
                plt.xlabel('Isotope')
                plt.ylabel('Sensitivity')
                if self.equi_comp:
                    plt.title('Equilibrium composition sensitivity')
                else:
                    plt.title('initial composition sensitivity')
                reaction="all"
                
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.grid()
            if reaction is None:
                reaction=""
            if isotope is None:
                isotope=""   
            plt.savefig(os.path.join(plot_path,f"rank_bar_{reaction}{isotope}.png"))
        return rank_sens
     
    def remove_iso_MT(self,array,iso_remove=None,MT_remove=None):
        """remove all the rows with 

        Args:
            array (_type_): _description_
            iso_remove (str): The name of the isotopes to be removed in zaid format or name
            MT_remove (str): The name of the MT reaction to be removed
        """
        filtered_data = []
        for row in array:
        # Check if the row should be removed based on isotope and/or MT
            if (iso_remove is None or row[0] != iso_remove) and (MT_remove is None or row[1] != MT_remove):
                filtered_data.append(row)
        return np.array(filtered_data)
    
    def check_total_RX(self,reaction):
        """check if all the isotopes for a specfic reaction add up to total"""
        self.check_run_extract()

        sum=0
        sum_s=0
        for row in self.zaid_MT_intEsens:
            if row[1].strip()==reaction:
                if row[0].strip()=="total":
                    total=row[2]
                    total_s=row[3]
                else:
                    sum+=row[2]
                    sum_s=np.sqrt(row[3]**2+sum_s**2)
                    rel,rel_s=rel_diff(total,total_s,sum,sum_s)
        
        print(f"For reaction {reaction} the sens: total={total}+/-{total_s}, sum={sum}+/-{sum_s}, diff={rel}+/-{rel_s} %")
        return total, total_s, sum, sum_s

    def check_total_iso(self,isotope):
        """check if all the reactions for a specific isotopes add up to total xs""" 
        self.check_run_extract()

        sum=0
        sum_s=0
        isotope=zai_to_nuc_name(isotope)
        for row in self.zaid_MT_intEsens:
            if row[0].strip()==isotope:
                if row[1].strip()=="total":
                    total=row[2]
                    total_s=row[3]
                else:
                    sum+=row[2]
                    sum_s=np.sqrt(row[3]**2+sum_s**2)
                    rel,rel_s=rel_diff(total,total_s,sum,sum_s)

        print(f"For isotope {isotope} the sens: total={total}+/-{total_s}, sum={sum}+/-{sum_s}, diff={rel}+/-{rel_s} %")
        return total, total_s, sum, sum_s

    def abs_contribution_iso(self):
        """check if all the reactions for a specific isotopes add up to total xs""" 
        self.check_run_extract()

        sum_list=[]
        sum_list_s=[]

        for isotope in self.sens_iso:
            isotope=zai_to_nuc_name(isotope)
            sum=0
            sum_s=0
            for row in self.zaid_MT_intEsens:
                if row[0].strip()==isotope:
                    if row[1].strip()=="total xs":
                        pass
                    else:
                        sum+=np.abs(row[2])
                        sum_s=np.sqrt(row[3]**2+sum_s**2)
            sum_list.append(sum)
            sum_list_s.append(sum_s)
        plt.figure(figsize=(10, 6))
        ind_0=50
        abs_sens=np.array(sorted(zip(self.sens_iso,sum_list,sum_list_s),key=lambda x: float(x[1]),reverse=True))
        name=[]
        for i in abs_sens[1:ind_0,0]:
            name.append(zai_to_nuc_name(i))

        plt.bar(name, np.float64(abs_sens[1:ind_0,1])/np.float64(abs_sens[0,1])*100,bottom=0,align="center", alpha=0.7)
        plt.xlabel('Isotope')
        plt.ylabel('Relative contribution to sum of absolute sensitivities [%]')
        plt.xticks(rotation=45, ha='right')
        plt.yscale("log")
        plt.tight_layout()
        plt.grid()
        plt.ylim(0.1)
        plt.savefig(os.path.join(self.folder_name,"abs_contribution_sens_equi.png"))

        return  sum_list, sum_list_s
    
    def error_prop(self):
        """propagate error with covariance matrix and sensitivity vector using the sandwidch rule, 
        if 

        Args:
            isotope (_type_, optional): _description_. Defaults to None.
            reaction (_type_, optional): _description_. Defaults to None.


        Returns:
            _type_: _description_
        
        """

        self.path_cov=os.path.join('c:/',"Users","jbval","PDM","02_cov","cov_npy","33g_TENDL2023")

        N_E_G = 33# number of energy groups
        flat_sens=self.sens.flatten()
        cov=get_ND_cov_matrix(self.zaid_MT,33,self.path_cov)
        covar=flat_sens.dot(cov).dot(flat_sens.T)
        print(np.sqrt(covar)*1e5)

        # Step through zaids in sens_data pd dataframe to decompose covar of 2 nuc/rx pairs
        covar_dict = {}
        iStart = 0; iEnd = 0; iNuc = 0;
        zaid_finished = []
        for row1 in self.zaid_MT:
            zaid1 = row1[0][:-1]
            if zaid1 not in zaid_finished:
                for row2 in self.zaid_MT[iNuc:]:
                    if zaid1 == row2[0][:-1]:
                        iEnd += N_E_G
                zaid_finished.append(zaid1)

                S_nuc = flat_sens[iStart:iEnd]
                M_sigma_rel_nuc = cov[iStart:iEnd,
                                            iStart:iEnd]   
                # Then calculate covar
                covar = S_nuc.dot(M_sigma_rel_nuc).dot(S_nuc.T)
                covar_dict[zaid1] = covar
                
                iStart = iEnd
            iNuc += 1

        # Sort the dict so that it ranks covariance by absolute value
        covar_dict_abs = {}
        # Flip keys and values, all zeros and repeated values are lost
        for nuc, covar in covar_dict.items():
            covar_dict_abs[str(abs(covar))] = nuc

        # convert to float so covs can be sorted
        cov_abs = [abs(float(cov)) for cov in covar_dict_abs.keys()]
        cov_abs.sort()
        cov_abs = cov_abs[::-1]  # Flip so its descending

        ranking = []
        for abs_cov in cov_abs:
            # Flip back to original keys and values
            true_key = covar_dict_abs[str(abs_cov)]
            true_cov = covar_dict[true_key]
            ranking.append([true_key, true_cov])

        print(np.sum(np.float64(cov_abs)))
        print(ranking)

      


                
######################################################################################################

#                           Classes of class instances                                              #

######################################################################################################                 
    
class MCFRs():
    def __init__(self,MCFR=None):
        self.list_MCFR=MCFR if MCFR is not None else []
        if not isinstance(self.list_MCFR,list):
            self.list_MCFR=[MCFR]
        self.exec_name=None
        if MCFR is not None:
            self.folder_name=self.list_MCFR[0].folder_name
            self.check_same_lib()
            self.lib_all=self.list_MCFR[0].lib_all
            self.lib_cl=self.list_MCFR[0].lib_cl
            self.len=len(self.list_MCFR)
            self.template_path=self.list_MCFR[0].template_path
                    
    def add_MCFR(self,new_MCFR):
        self.list_MCFR.append(new_MCFR)
        self.__init__(self.list_MCFR)

    def wipe(self):
        """Wipe the directory of output files
        """
        # List all files in the directory
        files = os.listdir(self.folder_name)
        for file in files:
            for i in self.list_MCFR:
        # Iterate through files and delete those starting with the specified prefix
                if file.startswith(i.file_name+"_") or file.startswith(i.file_name+"."):
                    file_path = os.path.join(i.folder_name, file)
                    os.remove(file_path)
                                        
    def run_all(self,nodes=1,partition="savio3",run_msg="running"):
        """Run all the MCFR instances in the class using a single executing file
        """
        
        self.wipe()
        self.gen_exec_all(nodes,partition)
        # os.system("nohup")
        o_path=os.path.join(self.folder_name,self.folder_name+".o")
        if os.path.exists(o_path):
            os.remove(o_path) 
        print(run_msg)   
        os.system(f"cd {self.folder_name}; sbatch {self.exec_name} ")
        
       
        #wait for the final "simulation completed" before continuing to post processing
        self.list_MCFR[0].simu_wait(o_path,number=self.len)
        print("finished the simulation")
        
    def gen_exec_all(self,time,nodes,partition):
        sss_str,variables=self.list_MCFR[0].get_exec(time,nodes,partition)
        variables[0]=self.folder_name #change the .o path 
        cmd=[]
        #combine the serpent launch command line from each execute to put it into a single execute file
        for i in self.list_MCFR:
            _,variables_temp=i.get_exec(time,nodes,partition)
            cmd.append(variables_temp[-1])
            
        cmd=np.array(cmd).reshape(-1,1)
        variables[-1]=array_txt(cmd)
        return sss_str, variables
    
    def check_same_lib(self):
        """check if all the MCFR are from the same Cl librairy"""
        first=self.folder_name
        for  i in self.list_MCFR:
            if i.folder_name!=first:
                raise ValueError("To run sequentially the MCFRs must be of the same librairy (thus in the same output directory) with the same prefix")
 

class Statics(MCFRs):
    def __init__(self,Statics=None):
        super().__init__(Statics)
        self.e_cl=self.get_list_cl_e()

    def get_list_cl_e(self):
        e_cl=[]
        for i in self.list_MCFR:
            e_cl.append(i.e_cl)
        return np.array(e_cl) 
    
    def gen_exec_all(self, nodes=1, partition="Savio3"):
        time=0.5*self.len
        sss_str,variables= super().gen_exec_all(time, nodes, partition)
        self.exec_name=f"execute_{self.lib_cl}_{self.len}.sub"
        self.list_MCFR[0].write_to_file(sss_str,variables,self.template_path,self.exec_name)
        
    def run_all(self, nodes=1, partition="savio3"):
        run_msg=f"Starting static simulation for lib: {self.lib_cl}, for All-lib {self.lib_all} with {*self.e_cl,} different enrichments"
        super().run_all(nodes, partition,run_msg=run_msg)

    def extract_plot(self,variable, plot=True):
        """extract variable from the res.m out file, 

        Args:
            variable (str): variable to be extracted
            plot (bool): to plot variable in function of Cl_e

        Returns:
            array: variable in a list in function of Cl enrichment
        """
        var=[]
        plot_dir=f"{variable}"
        plot_path=os.path.join(self.folder_name,plot_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for i in self.list_MCFR:
            temp=i.extract_res_m(variable)
            var.append(temp)
        var=np.array(var)

        if plot:
            plt.figure()
            plt.errorbar(self.e_cl*100,var[:,0],yerr=var[:,1],fmt="o")
            degree = 1
            coefficients = np.polyfit(self.e_cl*100, var[:,0], degree)

            # Generating the fitted curve
            fitted_curve = np.polyval(coefficients, self.e_cl*100)
            plt.plot(self.e_cl*100,fitted_curve,label=f"fit of order {degree}, with slope of {coefficients[0]*1e5:.1f}pcm/w%")
            #plt.fill_between(self.e_cl*100,k[:,0]-k[:,1]*1,k[:,0]+k[:,1]*1,alpha=1)
            plt.xlabel("Cl-37 enrichment [w%]")
            plt.ylabel(f"{variable}")
            plt.legend()
            plt.grid()
            plt.title(f"{variable} in function of Cl enrichment, Cl35-{self.lib_cl}.0, All-{self.lib_all}.0")
            plt.savefig(f"{plot_path}/{variable}_e_{self.lib_cl}.png")
            plt.close()
        return var
  
    
class Depletions(MCFRs):
    def __init__(self,Depletions=None):
        super().__init__(Depletions)
        self.mflow_list=self.get_list_mflow()
        
    def get_list_mflow(self):
        mflow=[]
        for i in self.list_MCFR:
            mflow.append(i.mflow_in)
        return np.array(mflow) 
    
    def gen_exec_all(self, nodes=1, partition="Savio3_bigmem"):
        time=0.9*self.len
        sss_str,variables= super().gen_exec_all(time, nodes, partition)
        self.exec_name=f"execute_{self.lib_cl}_{self.len}.sub"
        self.list_MCFR[0].write_to_file(sss_str,variables,self.template_path,self.exec_name)
        
    def run_all(self, nodes=1, partition="savio3_bigmem"):
        run_msg=f"Starting  Depletion simulations for lib: {self.lib_cl}, for All-lib {self.lib_all} with {*self.mflow_list,}  mflow values"
        super().run_all(nodes, partition,run_msg=run_msg)
        
    def extract_dep_m_all(self,isotopes,var_name,do_plot=True,logy=True):
        units=self.list_MCFR[0].units
        if var_name not in units:
            raise ValueError(f"Make sure the variable is part of the following {units.keys()}")
        unit=units[var_name]
        plot_dir=f"{var_name}_#{len(isotopes)}iso"
        plot_path=os.path.join(self.folder_name,plot_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        var=[]  
        for i in self.list_MCFR:
            temp,Bu=i.extract_dep_m(isotopes,var_name,plot_dir,do_plot=do_plot,logy=logy)
            var.append(temp)
        var=np.array(var)
        if False:
            if do_plot:
                if len(isotopes)==1:
                    fig=plt.figure()
                    ax = fig.add_subplot(111)
                    enri_mesh,Bu_mesh=np.meshgrid(self.e_cl,Bu)
                    z_flat=var[:,0,0,:]#.flatten()
                    vmin=np.min(z_flat)
                    vmax=np.max(z_flat)
                    levels=15
                    surf=plt.contourf(Bu_mesh/365,enri_mesh*100,z_flat.T,levels=levels,norm=Normalize(vmin=vmin,vmax=vmax))
                    ax.set_xlabel('Burnup years')
                    ax.set_ylabel('Cl-37 Enrichment [w%]')
                    ax.grid()
                    cbar=plt.colorbar(surf,ticks=np.linspace(vmin, vmax, levels),label=f"{var_name} [{unit}]")
                    ax.set_title(f"{var_name}(BU,Cl_e) for {*isotopes,} using Cl35-{self.lib_cl}.0, All-{self.lib_all}.0",fontsize=10)
                    plt.savefig(f"{plot_path}/Bu_ecl_{var_name}.png")
                    plt.close()
        return  var, Bu

    def extract_res_m_all(self,var_names,do_plot=True):
        var=[]
        if not isinstance(var_names,list):
            var_names=[var_names]
        plot_dir=f"{*var_names,}"
        plot_path=os.path.join(self.folder_name,plot_dir)
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        for i in self.list_MCFR:
            temp,Bu=i.extract_res_m(var_names,do_plot=False)
            #print(temp)
            var.append(temp)
        var=np.array(var)
        if do_plot:
            plt.figure()
            for i in range(len(self.mflow_list)):
                for j in range(len(var_names)):
                    label=f"{var_names[j]} for ${{\lambda_{{in}}}}$={self.list_MCFR[i].mflow_in} /s"
                    plt.plot(Bu/365,var[i,j,:,0],label=label)
                    plt.fill_between(Bu/365,var[i,j,:,0]-var[i,j,:,1],var[i,j,:,0]+var[i,j,:,1],alpha=0.4)
            plt.grid()
            title=f"Macro variables for {self.lib_cl}/{self.lib_all} with Cl_e={self.list_MCFR[0].e_cl}w%"
            plt.title(title,fontsize=10)
            plt.legend()
            plt.xlabel('Burnup years')
            if len(var_names)==1:  
                plt.ylabel(f"{var_names[0]}")
            else:
                plt.ylabel("a.u.")
            plt.savefig(f"{plot_path}/Bu_mflow_{*var_names,}.png")
        return var, Bu


class Sens_pair(MCFRs):
    """create a pair or initial and equilibrium composition Sensitivity, 
    """
    def __init__(self,mflow_in,sens_iso="all",sens_MT="all",mflow_out=None,sens_resp="keff",e_cl=0.75, e_U_stock=0.1975,lib_cl="ENDF8",lib_all="ENDF8",pop=1e5,active=500,inactive=25,reactor_type="MCFR_C",get_flux=False,prefix="",BU_years=60):
        self.Initial=Sensitivity(sens_iso,sens_MT,sens_resp,None,None,False,e_cl,e_U_stock,lib_cl,lib_all,pop,active,inactive,reactor_type,get_flux,prefix,BU_years)
        self.Equi=Sensitivity(sens_iso,sens_MT,sens_resp,mflow_in,mflow_out,True,e_cl,e_U_stock,lib_cl,lib_all,pop,active,inactive,reactor_type,get_flux,prefix,BU_years)
        super().__init__([self.Initial,self.Equi])
        
    def gen_exec_all(self, nodes, partition):
        time=2*self.len
        sss_str,variables= super().gen_exec_all(time, nodes, partition)
        self.exec_name=f"execute_pair_{self.lib_cl}_{self.len}.sub"
        self.list_MCFR[0].write_to_file(sss_str,variables,self.template_path,self.exec_name)
        
    def run_all(self, nodes=1, partition="savio3_bigmem"):
        run_msg=f"Starting sensitivity simulation for lib: {self.lib_cl}, for All-lib {self.lib_all} with initial and equilibirum composition"
        super().run_all(nodes, partition,run_msg=run_msg)
    
    def extract_sens_m_all(self,zais,perts,zais_equi=None,perts_equi=None,do_plot=True):
        var=[]
        plot_dir="sensitivity_plots"
        if zais_equi is None:
            zais_equi=zais
            perts_equi=perts

        k,k_s,E,du=self.Initial.extract_sens_m(zais,perts,plot_dir=plot_dir,do_plot=do_plot)
        k_equi,k_sequi,_,_=self.Equi.extract_sens_m(zais_equi,perts_equi,plot_dir=plot_dir,do_plot=do_plot)

        return k,k_s,k_equi,k_sequi,E,du
