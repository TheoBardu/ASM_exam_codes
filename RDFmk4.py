#! /Users/theo/opt/anaconda3/bin/python3
#Script that calculates the radial distribution function (RDF) for water molecules.

from numpy import zeros, linspace, histogram, sum, array, linalg, zeros_like, pi, float64, shape
from time import process_time, sleep
from matplotlib import pyplot as plt
from numba import njit, float64


#Structures of data
data_type = [("atom", "U9"), ("x","f8"), ("y","f8"), ("z","f8")] 
rdf_type = [("g_OO", "f8"), ("g_OH", "f8"), ("g_HH", "f8")]

#Initialization of data
Rmax = 12
Nrmax = 401 #best 401





def initial(filename):
    """
    This function read initially the file in <filename> and save in an array all the positions for all the iterations

    INPUT:
        filename = <str>, the entire path of the file

    """
    print(f'-- Initialization of variables and inspecting the file --\nFie: {filename}')

    tic_1reading = process_time()
    fid = open(filename,'r') #we open the file in reading mode
    n_atoms = int(fid.readline()) #taking the number of the atoms and convert in integer
    comment = fid.readline().split() #splitting the second line of comment (iteration and cell param)
    n_iter = 0 #to taking trace the number of iterations

    #creating and memorizing the cell's structure
    cell = zeros((3,3)) #3x3 matrix
    p = 1
    for i in range(3):
        for j in range(3):
            cell[i,j] = float(comment[p])
            p += 1
    
    #counting the number of iterations
    l = 0
    while l != "": #loop over all the lines of the file
        for i in range(n_atoms):
            fid.readline()
        l = fid.readline() #read the iteration line
        fid.readline() #read the comment line
        n_iter +=1
    
    data = zeros(shape=(n_iter, n_atoms), dtype = data_type) #initialize the data matrix
    toc_1reading = process_time()

    #saving positions in data matrix
    tic_saving = process_time()
    fid.seek(0) #going back to the beginning of file
    for iter in range(n_iter):
        fid.readline() #getting the n_atoms
        fid.readline() #getting the comment
        read_pos(fid,n_atoms,iter,data) #reading the positions atomic positions
        
    toc_saving = process_time()

    
    fid.close() #closing the file



    #printing the values
    print(f"N atoms = {n_atoms}")
    print(f"Cell Parameters:\n {cell}")
    print(f'Number of iterations: {n_iter}')
    print("Time elapsed for getting n_iter: %.6f s "%(toc_1reading-tic_1reading))
    print("Time elapsed for saving data: %.6f s "%(toc_saving - tic_saving))
    


    return fid, n_atoms, cell, n_iter, data


def read_pos(fid, n_atoms, iter_num, data):
    """
    Function that read the positions at a specific iteration and store them in a matrix
    INPUT:
        fid = file identifier for the input file
        n_atoms = <int>, number of atoms
        iter_num = <int>, number of total iterations done
    """
    for i in range(n_atoms):
        line = fid.readline().split()
        data[iter_num,i] = (line[0],line[1],line[2], line[3])
    







@njit
def rdf(data,cell, n_iter):
    """
    Function that evaluate the radial distribution function for all the iteration and calculate the mean one for atoms couples
    INPUT:
        data = <ndarray>, all the position data
        cell = <3x3 array>, the cell parameters
        n_iter = <int>, number of iterations
    """
    gOO = zeros((n_iter,Nrmax-1), dtype=float64)
    gOH = zeros((n_iter,Nrmax-1),dtype=float64)
    gHH = zeros((n_iter,Nrmax-1), dtype=float64)

    for i in range(n_iter):
        print(i+1)
        gOO[i,:], r = evaluate_rdf(data, cell, i, "O","O")
        gOH[i,:], r = evaluate_rdf(data, cell, i, "O","H")
        gHH[i,:], r = evaluate_rdf(data, cell, i, "H","H")

    gOO = sum(gOO,0)/n_iter
    gOH = sum(gOH,0)/n_iter
    gHH = sum(gHH,0)/n_iter

    
    g = [gOO, gOH, gHH]
    
    
    
    return g, r


@njit
def evaluate_rdf(data, cell, iter_num, atom1, atom2):
    """
    Function that evaluate the radial distribution function at a specific iteration
    INPUT:
        data = <ndarray>, matrix with all the data
        iter_num = <int>, corrent iteration step
        atom1 = <str>, the central atom
        atom2 = <str>, the atoms that we have to consider for the evaluation of the pair distance
    """

    #let's find the dimensions of atoms and positions in the <data> matrix
    if atom1 == "H":
        n_atom1 = 128
        low_bound1 = 64
        up_bound1 = 191
    elif atom1 == "O":
        n_atom1 = 64
        low_bound1 = 0
        up_bound1 = 63        

    if atom2 == "H":
        n_atom2 = 128
        low_bound2 = 64
        up_bound2 = 191
    elif atom2 == "O":
        n_atom2 = 64
        low_bound2 = 0
        up_bound2 = 63    

    # print('Natom1 = ', n_atom1)
    # print('Natom2 = ', n_atom2)

    #Finding the dimension of dist array
    if atom1 != atom2:
        dist_dim = n_atom1 * n_atom2
    else:
        dist_dim = (n_atom1 * n_atom1) - n_atom1
    
    # print('dist Dim = ', dist_dim)

    dist = zeros(dist_dim, dtype = float64)
    p = 0

    for i in range(low_bound1,up_bound1+1):
        Ri = [data[iter_num, i]["x"], data[iter_num,i]["y"], data[iter_num,i]["z"] ] #position of ith atom1
        for j in range(low_bound2, up_bound2+1):
            if i != j:
                Rj = [data[iter_num,j]["x"], data[iter_num,j]["y"], data[iter_num, j]["z"] ] #position of ith atom2
                
                d = pbc_distance(Ri,Rj,cell)
                dist[p] = d
                p +=1
    
    # print(len(dist))
    # print(dist)

    bins = linspace(0,Rmax, Nrmax)
    hist, bins = histogram(dist, bins)

    radii = zeros_like(hist)
    volumes = zeros_like(hist)

    radii = (bins[0:-1] + bins[1:])/2.0
    volumes = 4.0/3.0 * pi * (bins[1:]**3 - bins[0:-1]**3)

    density = n_atom2 / cell[0,0]**3
    g = hist/(volumes * density * n_atom1)
   
    
    return g, radii


@njit
def pbc_distance(r1,r2, cell):
    """
    Calculate the PBC dinstance of two particle at positions r1 and r2
    """
    d = array(r1,dtype=float64) - array(r2,dtype=float64)
    d = (d + cell[0][0]/2.0) % (cell[0][0]) - cell[0][0]/2.0
    return linalg.norm(d)






def write_output(file, r, g):
    fid = open(file,'w') #open for writing
    fid.write("r\tg_OO\tg_OH\tg_HH\n")
    for i in range(len(r)):
        s = f'{r[i]}\t{g[0][i]}\t{g[1][i]}\t{g[2][i]}\n'
        # s = f'{r[i]}\t{g[i]}\t{g[i]}\t{g[i]}\n'
        fid.write(s)
    fid.close()
    print("Output file writed")
        

def plot_g(r,g, save = True, plot = False):
    plt.plot(r,g[0], label= '$g_{OO}$')
    plt.plot(r,g[1], label= '$g_{OH}$')
    plt.plot(r,g[2], label= '$g_{HH}$')
    plt.legend()
    plt.xlabel('r [Å]')
    plt.ylabel('g(r) [a.u]')
    plt.xlim((0,cell[0,0]/2))
    if save:
        plt.savefig(f"/Users/theo/Desktop/ASM/plots/{name_g}.svg", format = "svg")
        print("Figure created and saved")
    
    if plot:
        plt.show()










## -------------------------
##          MAIN
## -------------------------

run = [0, 2, 0]
i = int(input("File number: "))
# for i in range(run[0], run[2] + run[1],run[1]):
name_pos = "pos%04i.xyz"%(i)
name_g = "g%04i"%(i)

#Initialization of parameters and data
fid, n_atoms, cell, n_iter, data = initial(f"/Users/theo/Desktop/ASM/pos/{name_pos}")


## RDF loop for all iterations
tic = process_time()
g,r = rdf(data, cell, n_iter)
toc = process_time()
print("RDF loop terminated. Time spent: %.3f s"%(toc-tic))





# print(g)
write_output(f"/Users/theo/Desktop/ASM/rdf_data/{name_g}.txt",r, g)


#Plotting
# plot_g(r,g, save= False, plot = True)

print("@@@ PROGRAM TERMINATED @@@")


  


