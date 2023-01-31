#!/home/user/miniconda3/envs/egs/bin/python
import numpy as np
import pickle
import pandas as pd
import sys
#\!/~/miniconda3/envs/skmod/bin/python

#from ProcessInput import ProcessInput
from ProcessCSV import FullProcessSingle, CompareImages
from ProcessLabelData import ProcessDistData

def Chi2_i(prediction_i, label_dist):
    #Cowan p. 104
    # The quantity (y_i - f(x_i, w)) / s_i is a measure of deviation between the 
    # ith measurement y_i and the function f(s_i, w), so Chi^2 is a measure of 
    # the total agreement between ovserved data and hypothesis. It can be shown
    # that if

    # 1) y_i, i=1,...,N are independent Gaussian random variables [...]
    #       > mean of distribution and expectation of the mean are expected to be 
    #         the same.
    #       > !!! wikipedia says this should be calculated about a sample mean
    #             (not population mean)
    y_i = np.mean(label_dist)

    # [...] with known variances s_i^2
    #s_i = np.var(label_dist)
    population_variance = np.var(label_dist)
    population_variance_of_the_mean = population_variance / len(label_dist)

    chi2_i = (y_i - prediction_i)**2 / population_variance_of_the_mean
    return chi2_i

def Chi2Test(input_data, label):
    from LoadModel import LoadModel, InferNPad, GetExpandedShape
    mod = LoadModel()

    chi2 = 0
    chi2_none = 0
    chi2_compare = 0

    # number of data points = f*f
    N_pixels = len(input_data)

    # Number of independent parameters
    m = len(input_data[0])

    N_sigma = 1.0
    for ind in range(N_pixels):
        # slice down the set of samples (ind refers to pixel)
        label_i = label[ind, :]
        
        # Select one of (duplicated) scattering object for input to mod
        input_data_i = input_data[ind]
        input_data_shape = GetExpandedShape(input_data_i)
        input_data_i = input_data_i.reshape((input_data_shape))

        # Get model prediction
        pred_i = mod.predict(input_data_i)
        pred_i = pred_i.item()
        
        # Aggregate chi^2 for model
        chi2 += Chi2_i(pred_i, label_i)

        # Negative control / comparison
        test_none = np.mean(label_i)
        chi2_none += Chi2_i(test_none, label_i)

        #test = np.mean(label_i)
        #test += N_sigma * np.random.random() * np.std(label_i)
        #test += N_sigma * np.std(label_i)
       
        # Positive control / comparison. Normally distributed around mean w/ N_sigma
        # std.
        test_std = N_sigma * np.std(label_i) / np.sqrt(len(label_i))
        test = np.random.normal(loc=np.mean(label_i), scale=test_std)
       
        # aggregate chi2 values for positive control
        chi2_compare += Chi2_i(test, label_i)


    # divide by number of degress of freedom
    degrees_of_freedom = N_pixels - m
    print(f"degrees of freedom, chi2_func: {degrees_of_freedom}")
    #chi2 /= degrees_of_freedom 
    #chi2_compare /= degrees_of_freedom
    #chi2_none /= degrees_of_freedom

    # Get probability of sample
    #mod_prob = sts.chi2.cdf(chi2_mod, loc=)
    return chi2, chi2_none, chi2_compare, N_sigma, degrees_of_freedom


#def CompareToDist(rn):
if True:
    # Load Scatter Input
    try:
        rn = sys.argv[1]
    
    except IndexError:
        rn = "62783361878388904048055595911667886675"
        #rn = "252785838956249757503353786711590552658"
    #    rn = "334800398373121594589956989"
    
    input_object_path = "".join((
                        f"./{rn}_-_detector_results/",
                        f"scatter-learn_-_num_0-{rn}_-_scatter_input.csv"))
    
    from LoadModel import LoadModel, InferNPad, GetExpandedShape
    mod = LoadModel()
    labr_min = mod.labr_min
    labr_minmax = mod.labr_minmax
    
    # Calculate npad from model
    npad = InferNPad(mod)
    
    # Get test object input.
    input_data = FullProcessSingle(input_object_path, npad)
    prediction = mod.predict(input_data)
    
    distribution_path = "".join((
                        f"./{rn}_-_detector_results/",
                        "column_3.csv"))
    
    # (f*f pixels, N experiments)
    label = pd.read_csv(distribution_path, header=None)
    label = label.to_numpy()
    label_og = label.copy()

    import matplotlib.pyplot as plt

    #fig = plt.figure()
    #plt.imshow(label_og[:,3:][:,1].reshape((64,64)))
    #fig.show()

    # (N experiments, f*f pixels)
    label = ProcessDistData(label, lv_min=labr_min, lv_minmax=labr_minmax)

    #fig2=plt.figure()
    #plt.imshow(label[0].reshape((8,8)))
    #fig2.show()

    label_shape = GetExpandedShape(label)
    label = label.reshape(label_shape)

    label = label.squeeze()
    
    
    
    
    
    
    
    chi2, chi2_none, chi2_compare, N_sigma, degrees_of_freedom  = Chi2Test(input_data.copy(), label.copy())#[0])
    print(f"chi2: {chi2}") 
    print(f"chi2_none ({N_sigma} sigma): {chi2_none}") 
    print(f"chi2_compare ({N_sigma} sigma): {chi2_compare}") 
    print(f"df: {degrees_of_freedom}")
    
    
    mean_image = np.mean(label, axis=0)
    mean_image = mean_image.reshape((8,8))
    pred = prediction.reshape((8,8))
    

    from scipy import stats as sts
    prob = sts.chi2.cdf(np.infty, df=64) - sts.chi2.cdf(chi2, df=64) 
    print(f"prob: {prob}")

    fig = CompareImages(pred, mean_image)
#    return chi2
#   
#
#try:
#    import sys
#    rn = sys.argv[1]
#    CompareToDist(rn)
#
#except:
#    rn = 97161475716309467815031936870382432174
#    CompareToDist(rn)
#    pass
