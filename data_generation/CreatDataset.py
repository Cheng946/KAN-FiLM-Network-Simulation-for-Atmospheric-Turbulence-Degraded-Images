import numpy as np
import os
import sympy as sp
import math
from scipy.signal import correlate
import multiprocessing

from scipy.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA
from multiprocessing import Pool,cpu_count

# This file is used to generate the training dataset for the KAN+FiLM network,producing 50,000 data pairs.
# For the physical principles underlying the dataset generation methodology, please refer to the literature:
# "Chimitt, N.; Chan, S. H. Simulating anisoplanatic turbulence by sampling intermodal and spatially correlated Zernike coefficients. Optical Engineering. 2020, 59, 083101"
# During usage, it is recommended to modify the save path for each step's files to ensure compatibility with your environment.

def genZernikeCoeff(num_zern):
    '''
    Just a simple function to generate random coefficients as needed, conforms to Zernike's Theory. The nollCovMat()
    function is at the heart of this function.

    A note about the function call of nollCovMat in this function. The input (..., 1, 1) is done for the sake of
    flexibility. One can call the function in the typical way as is stated in its description. However, for
    generality, the D/r0 weighting is pushed to the "b" random vector, as the covariance matrix is merely scaled by
    such value.

    :param num_zern: This is the number of Zernike basis functions/coefficients used. Should be numbers that the pyramid
    rows end at. For example [1, 3, 6, 10, 15, 21, 28, 36]
    :param D_r0:
    :return:
    '''
    C = nollCovMat(num_zern, 1, 1)
    e_val, e_vec = np.linalg.eig(C)
    R = np.real(e_vec * np.sqrt(e_val))

    b = np.random.randn(int(num_zern), 1)
    a = np.matmul(R, b)
    a=a[3:]
    a=np.squeeze(a)
    a[np.isinf(a)] = 0
    a[np.isnan(a)] = 0

    return a

def nollCovMat(Z, D, fried):
    """
    This function generates the covariance matrix for a single point source. See the associated paper for details on
    the matrix itself.

    :param Z: Number of Zernike basis functions/coefficients, determines the size of the matrix.
    :param D: The diameter of the aperture (meters)
    :param fried: The Fried parameter value
    :return:
    """
    C = np.zeros((Z,Z))
    for i in range(Z):
        for j in range(Z):
            ni, mi = nollToZernInd(i+1)
            nj, mj = nollToZernInd(j+1)
            if (abs(mi) == abs(mj)) and (np.mod(i - j, 2) == 0):
                num = math.gamma(14.0/3.0) * math.gamma((ni + nj - 5.0/3.0)/2.0)
                den = math.gamma((-ni + nj + 17.0/3.0)/2.0) * math.gamma((ni - nj + 17.0/3.0)/2.0) * \
                      math.gamma((ni + nj + 23.0/3.0)/2.0)
                coef1 = 0.0072 * (np.pi ** (8.0/3.0)) * ((D/fried) ** (5.0/3.0)) * np.sqrt((ni + 1) * (nj + 1)) * \
                        ((-1) ** ((ni + nj - 2*abs(mi))/2.0))
                C[i, j] = coef1*num/den
            else:
                C[i, j] = 0
    C[0,0] = 1
    return C

def nollToZernInd(j):
    """
    This function maps the input "j" to the (row, column) of the Zernike pyramid using the Noll numbering scheme.

    Authors: Tim van Werkhoven, Jason Saredy
    See: https://github.com/tvwerkhoven/libtim-py/blob/master/libtim/zern.py
    """
    if (j == 0):
        raise ValueError("Noll indices start at 1, 0 is invalid.")
    n = 0
    j1 = j-1
    while (j1 > n):
        n += 1
        j1 -= n
    m = (-1)**j * ((n % 2) + 2 * int((j1+((n+1)%2)) / 2.0 ))

    return n, m

def zernikeGen(coeff,ZernPoly36, **kwargs):

    result = coeff[:,np.newaxis, np.newaxis] * ZernPoly36

    return result

def genZernPoly(index, x_grid, y_grid):
    """
    This function simply

    :param index:
    :param x_grid:
    :param y_grid:
    :return:
    """
    n,m = nollToZernInd(index)
    radial = radialZernike(x_grid, y_grid, (n,m))
    #print(n,m)
    if m < 0:
        return np.multiply(radial, np.sin(-m * np.arctan2(y_grid, x_grid)))
    else:
        return np.multiply(radial, np.cos(m * np.arctan2(y_grid, x_grid)))

def radialZernike(x_grid, y_grid, z_ind):
    rho = np.sqrt(x_grid ** 2 + y_grid ** 2)
    radial = np.zeros(rho.shape)
    n = z_ind[0]
    m = np.abs(z_ind[1])

    for k in range(int((n - m)/2 + 1)):
        #print(k)
        temp = (-1) ** k * np.math.factorial(n - k) / (np.math.factorial(k) * np.math.factorial((n + m) / 2 - k)
                                                       * np.math.factorial((n - m) / 2 - k))
        radial += temp * rho ** (n - 2*k)

    # radial = rho ** np.reshape(np.asarray([range(int((n - m)/2 + 1))]), (int((n - m)/2 + 1), 1, 1))

    return radial


def ComputOTF(a,Z):
    Zernike_stack = zernikeGen(a, Z);
    Fai = np.sum(Zernike_stack, axis=0)

    # print(type(Fai),type(mask))

    # 根据傅里叶变换得到对偶定理，F(F(w()))计算OTF
    wave = np.exp(1j * 2 * np.pi * Fai);

    # wave进行归一化,离散Parseval定理np.sum(PSF)/(128**2)=np.sum(np.abs(wave)**2)
    p = np.sum(np.abs(wave) ** 2)
    wave = wave * (((1 / 128 ** 2) / p) ** 0.5)

    cor = correlate(wave, wave, mode='same') * N ** 2
    OTF = cor[::-1, ::-1]

    return OTF


if __name__ == "__main__":
    wvl = 0.525e-6;  # wavelength
    L = 7000;  # propagation distance
    D = 0.305;  # observation aperture
    k = 2 * np.pi / wvl; # wave vector

    # Sampling grid size
    N = 128;
    # Sampling spacing of the target
    delta0 = L * wvl / (2 * D);

    # aperture function
    x_grid, y_grid = np.meshgrid(np.linspace(-1, 1, N, endpoint=True), np.linspace(-1, 1, N, endpoint=True))
    mask = np.sqrt(x_grid ** 2 + y_grid ** 2) <= 1

    # get current working directory
    current_directory = os.getcwd()

    # Zernike Basis Functions (128×128 Size)
    file_name = "36—128ZernPoly.npy";

    # Splicing file paths
    file_path = os.path.join(current_directory, file_name)

    # The optimization here uses genZernPoly(index, x_grid, y_grid).
    # ZernPoly36=mask*genZernPoly(36, x_grid, y_grid)
    # np.save(file_path, ZernPoly36)

    ZernPoly36 = np.load(file_path)
    ZernPoly36 = np.transpose(ZernPoly36, (2, 0, 1))
    ZernPoly36=ZernPoly36[3:,:,:]


    Cn2 = 15 * (1e-16);  # Turbulence intensity,m^-2/3
    # Define symbolic variables
    z = sp.symbols('z')
    # Define integral expressions
    expression = (z / L) ** (5 / 3);
    r0 = ((0.423 * (2 * np.pi / wvl) ** 2) * Cn2 * sp.integrate(expression, (z, 0, L))) ** (-3 / 5);

    # The scaling factor of the coefficient with respect to turbulence intensity
    kappa = ((D / r0) ** (5 / 3) / (2 ** (5 / 3)) * (2 * wvl / (np.pi * D)) ** 2 * 2 * np.pi) ** (0.5) * L / delta0;
#########################################################################################################################
    # Generate 50,000 data points for PCA fitting

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Create a list containing 50,000 instances of the number 36
    input_list = [36] * 50000

    # Use map to call the genZernikeCoeff function 50,000 times and collect the results into a list named results
    results = pool.map(genZernikeCoeff, input_list)

    # Close the process pool and wait for all child processes to complete
    pool.close()
    pool.join()

    results=np.array(results,dtype=np.float64)
    print(results.shape)
    results=results*kappa;

    #"/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_aj.npy":Please change the path to the one that corresponds to your environment
    np.save("/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_aj.npy",results)

###################################### Visualize the results ###########################################
    # a=np.load("/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_aj.npy",allow_pickle=True)
    # a=np.array(a,dtype=np.float64)
    # a=a[3256,:]
    #
    # # Calculate the phase
    # Zernike_stack = zernikeGen(a, ZernPoly36);
    # Fai=np.sum(Zernike_stack,axis=0)
    #
    # print(type(Fai),type(mask))
    #
    # # According to the duality theorem derived from the Fourier transform, compute the Optical Transfer Function (OTF) using (F(F(w())).
    # wave =  np.exp(1j * 2 * np.pi * Fai);
    # wave =mask *wave
    # PSF=fftshift(fft2(wave))
    # plt.imshow(np.abs(PSF)**2)
    # plt.show()

########################################################################################################
####################################    Generate the Optical Transfer Function (OTF).       ##################################################
    a=np.load("/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_aj.npy",allow_pickle=True)
    a=np.array(a,dtype=np.float64)

    with Pool() as pool:
        # Use a list to collect the results.
        # Use pool.starmap to pass multiple arguments to the process_element function.
        results = pool.starmap(ComputOTF, [(e, ZernPoly36) for e in a])

    # Convert the results back to a NumPy array.
    OTF = np.array(results)

    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_OTF.npy',OTF)

###########################################################################################################
##################################  Visualize the Optical Transfer Function (OTF) ##############################################################
    # OTF=np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_OTF.npy')
    # OTF = np.array(OTF, dtype=complex)
    #
    # plt.imshow(np.abs(OTF[4200,:,:]))
    # plt.show()

############################################################################################################
#################################  Fit the data to obtain a PCA model  #####################################
    Zoom_crop_self_convolution = np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_OTF.npy')
    Zoom_crop_self_convolution = Zoom_crop_self_convolution[:10000, :, :]

    num_matrix = Zoom_crop_self_convolution.shape[0]
    matrix_size = Zoom_crop_self_convolution.shape[1]

    Zoom_crop_self_convolution = Zoom_crop_self_convolution.reshape(num_matrix, matrix_size, matrix_size)

    # Convert the complex matrix into a real-valued matrix
    real_matrix = np.hstack((Zoom_crop_self_convolution.real, Zoom_crop_self_convolution.imag)).reshape(num_matrix, 2 *
                                                                                                        matrix_size * matrix_size)

    # print(real_matrix.shape)
    # Apply Kernel PCA for dimensionality reduction
    n_componets = 70  # The dimensionality after dimensionality reduction

    pca = PCA(n_components=n_componets)
    reduced_matrices = pca.fit_transform(real_matrix)

    # Save the model parameters
    model_filename = '/home/aiofm/PycharmProjects/My_JuaSuSuanFa/15e-16Cn2Data/pca_model-70.pkl'
    joblib.dump(pca, model_filename)


##############################################################################################################
########################################## Perform PCA decomposition on the OTF #################################################################
    OTF = np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_OTF.npy')
    OTF = np.array(OTF, dtype=complex)

    num_matrix = OTF.shape[0]
    matrix_size = OTF.shape[1]

    # Convert the complex matrix to a real matrix
    real_matrix = np.hstack((OTF.real, OTF.imag)).reshape(num_matrix, 2 *matrix_size * matrix_size)

    pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/pca_model-70.pkl')
    reduced_matrices = pca.transform(real_matrix)

    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_PCA70.npy',reduced_matrices)
#################################################################################################################
############################################## Display the reconstruction results ################################################
    # PCA_70=np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_PCA70.npy')
    # PCA_70=np.array(PCA_70,dtype=np.float64)
    # matrix_size=128
    #
    # pca = joblib.load('/home/aiofm/PycharmProjects/MyKANNet/15e-16Cn2Data/pca_model-70.pkl')
    # approx_complex_matrix=pca.inverse_transform(PCA_70[332:333])
    # approx_real_part=approx_complex_matrix[:,:matrix_size*matrix_size].reshape(matrix_size,matrix_size)
    # approx_imag_part=approx_complex_matrix[:,matrix_size*matrix_size:].reshape(matrix_size,matrix_size)
    # approx_complex_matrix = approx_real_part + 1j * approx_imag_part
    # plt.imshow(np.abs(approx_complex_matrix))
    # plt.show()

#########################################################################################################
########################################## Split the dataset #####################################################
    A_j=np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_aj.npy',allow_pickle=True)
    PCA70=np.load('/media/aiofm/F/20240830_50000_A_OTF_PCA/50000_PCA70.npy',allow_pickle=True)

    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Train/Train_input.npy', A_j[0:45000,:])
    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Test/Test_input.npy', A_j[45000:48000, :])
    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Val/Val_input.npy', A_j[48000:50000, :])


    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Train/Train.npy',PCA70[0:45000,:])
    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Test/Test.npy',PCA70[45000:48000, :])
    np.save('/media/aiofm/F/20240830_50000_A_OTF_PCA/Val/Val.npy',PCA70[48000:50000, :])

##########################################################################################################
