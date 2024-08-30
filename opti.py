import numpy as np
import serpentTools
import matplotlib.pyplot as plt
from MCFR import *
import csv


        
def find_e_crit(lib_cl,lib_all,itteration=5,min=0.5,max=0.9,maxloop=3,prefix="crit_e"):
    """find the critical enrichment of Cl37 for a given library pair

    Args:
        lib_cl (str): library name for Cl35
        lib_all (str): library name for all other isotopes
        itteration (int, optional): how many inner points for the linear fit . Defaults to 5.
        min (float, optional): the starting lower bound value . Defaults to 0.5.
        max (float, optional): starting upper bound value. Defaults to 0.9.
        maxloop (int, optional): maximum number of linear interpolation before loop is stopped. Defaults to 3.
        prefix (str, optional): prefix to be added to folder name to differentiate with other. Defaults to "crit_e"

    Returns:
        float: critical enrichment of Cl37
    """
    k_crit=[0,0]
    j=1
    k_best=[0,0]
    e_best=0
    prefix=prefix
    
    path=MCFR(1,lib_cl=lib_cl,lib_all=lib_all,prefix=prefix).folder_name
    with open('data.csv', 'w', newline='') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        
        # Write header
        csv_writer.writerow(['Keff', 'e_cl %'])
 
        while not k_crit[0]-1*k_crit[1]<1<k_crit[0]+1*k_crit[1]:
            #loop until k=1 is 2 sigma away for the critical enrichment simulation value of k
            e_list_cl=np.linspace(min,max,itteration)
            list_MCFR=MCFRs()
            #loop between min and max and run each enrichment
            for e in e_list_cl[1:]:
                list_MCFR.add_MCFR(MCFR(e,lib_cl=lib_cl,lib_all=lib_all,pop=1e4*j,active=100,prefix=prefix))
            
            list_MCFR.run_all(nodes=1)

            #add the critical simulation from previous step to avoid 1 simulation
            if j>1:
                list_MCFR.add_MCFR(test)

            #extract the keff from data and add linear fit 
            k=list_MCFR.plot_global_var("absKeff")
            degree = 1
            # coefficients=np.polyfit(list_MCFR.e_cl*100, np.log(k[:,0]), 1, w=np.sqrt(k[:,0]))
            coefficients = np.polyfit(list_MCFR.e_cl*100, k[:,0], degree)

            #interpolate to find critical enrichment of Cl
            e_crit=(1-coefficients[1])/coefficients[0]#np.log(1/np.exp(coefficients[1]))/coefficients[0] #(1-coefficients[1])/coefficients[0]

            #run simulation at critical enrichment 
            test=MCFR(e_crit,pop=4e4,active=100,prefix=prefix)
            test.run_serpent()
            k_crit,_=test.extract_global_var("absKeff")

            # if the simulation is not in fact critical to 1 sigma then find the distance in enrichment to critical according 
            # to the slope of the linear fit from the previous step                                                            
            d_e=(1-k_crit[0])/coefficients[0]
            print(f"Itteration {j+1}: The critical enrichment was interpolated as {e_crit} for a value of k={k_crit}")
            csv_writer.writerow([k_best, e_crit])
            min=e_crit
            max=e_crit+2*d_e
            j+=1
            #store in memory the best value of enrichment in case the max number of loop is reached
            if np.abs(1-k_crit[0])<np.abs(1-k_best[0]):
                k_best=k_crit
                e_best=e_crit

            if j>3:
                print(f"failure to converge after {maxloop} loops to 1 sigma of critical thus enrichment of {e_best} with k={k_best}")
                return e_best

    return e_crit



def find_mflow(lam_in_guess,lib_cl,lib_all,e_cl,prefix="mflow_conv"):
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
        Vf=69.272118e6
        Vgas=1e3
        
        lam=[]
    
        #finding the time constant for overflow using formula 𝜆_𝑜=𝜆_𝑠−(Δ𝜌_𝑔𝑎𝑠)/(𝜌_𝑓𝑢𝑒𝑙 Δ𝑡).𝑉_𝑔𝑎𝑠/𝑉_𝑓𝑢𝑒𝑙  
        for i in range(len(dmgas[0])):
            print(dmgas[0,i]/dt[i]*24*3600)
            lam_over=(lam_in*mstock[0,0]-dmgas[0,i]*Vgas/(dt[i]*24*3600*Vf))/mf[0,0]
            lam.append(lam_over)
        return np.mean(lam)
    

    #run a short simulation to find the ratio of dm_gas/dt for the given enrchment and lib. 
    gas_test=MCFR(e_cl,lib_cl=lib_cl,lib_all=lib_all,depletion=True,active=100,pop=5e3,prefix=prefix)
    if not os.path.exists(gas_test.out_path+"_dep.m"):
        gas_test.run_serpent()
    
    dep = serpentTools.read(gas_test.out_path+"_dep.m")
    lam_over=get_mflow_out(dep,lam_in_guess)
    test2=MCFR(e_cl,lib_cl=lib_cl,lib_all=lib_all,mflow_in=lam_in_guess,mflow_over=lam_over,depletion=True,active=100,pop=5e3,prefix=prefix)
    test2.run_serpent()