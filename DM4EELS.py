# MIT License
#
# Copyright (c) 2023 Nicolette Shaw
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ncempy.io import dm
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline

def TotalEELSAreaAcrossScan(ds, start=0, show=True):
    # Calculates the area of the spectrum at each point, area will be lower than the rest of the dataset when there is no EELS signal.
    areas = []
    for i in range(start, len(ds['data'])):
        areas.append(simps(ds['data'][i]))
    if show:
        plt.plot(areas)
        plt.title(ds['filename'])
        plt.show()
    return areas

def gauss(x, A, mu, sigma):
    "f(x; A, \mu, \sigma) = \frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}"
    return A / (sigma * math.sqrt(2 * math.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

def gaussianFit(ydata, plotfit=False):
    xdata = range(0, len(ydata))

    def gauss_fit(x, y):
        mean = sum(x * y) / sum(y)
        sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
        popt, pcov = curve_fit(gauss, x, y, p0=[max(y), mean, sigma])
        return popt

    A, mu, sigma = gauss_fit(xdata, ydata)

    if plotfit:
        # plt.axhline(A/(sigma*math.sqrt(2*math.pi)))
        plt.plot(xdata, ydata, 'ko', label='data')
        plt.plot(xdata, gauss(xdata, *gauss_fit(xdata, ydata)), '--r', label='fit')
        plt.legend()
        plt.title('Gaussian fit,  $f(x; A, \mu, \sigma) = \\frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}"$')
        plt.show()

    return A, mu, sigma


class getEELSAreas:
    def __init__(self, ds, n):
        self.spectrum = ds['data'][n]

    def showEELS(self):
        plt.plot(self.spectrum)
        plt.show()

    # Get index of 2 highest peaks in array of Y values.
    def getPeakInts(self):
        peak_indices, peak_dict = find_peaks(self.spectrum, height=0.3, distance=20, width=2)
        peak_heights = peak_dict['peak_heights']
        highest_peak_index = peak_indices[np.argmax(peak_heights)]
        second_highest_peak_index = peak_indices[np.argpartition(peak_heights, -2)[-2]]
        return highest_peak_index, second_highest_peak_index

    # Use the min between 2 peaks to get the width of the ZLP.
    def getSimpleBounds(self, highest_peak_index, second_highest_peak_index):
        Ysubset_peak2peak = self.spectrum[highest_peak_index:second_highest_peak_index]
        if len(Ysubset_peak2peak)==0:
            plt.plot(self.spectrum)
            plt.show()
            raise "spectrum does not have 2 well-defined peaks."
        Yval_min = min(Ysubset_peak2peak)
        index_min_in_subset = np.where(Ysubset_peak2peak == Yval_min)[0]
        if len(index_min_in_subset) > 1:
            raise "Multiple occurrences of minimum value found."
        X2index = index_min_in_subset[0] + highest_peak_index
        index_step = X2index - highest_peak_index
        X1index = highest_peak_index - index_step
        return X1index, X2index


    def getT_over_IMFP(self, method='simple'):
        '''

        :param method: simple, or gaussianFit
        :return:
        '''
        highest_peak_index, second_highest_peak_index = self.getPeakInts()
        X1index, X2index = self.getSimpleBounds(highest_peak_index, second_highest_peak_index)
        if method == 'simple':
            ZLP = simps(self.spectrum[X1index:X2index], dx=1)
            # print('simple ZLP area= ' + str(ZLP))
        elif method == 'gaussianFit':
            A, mu, sigma = gaussianFit(self.spectrum[X1index:X2index])
            # sigma is standard deviation, calculate area analytically.
            ZLP = A/(sigma*math.sqrt(2*math.pi)) * sigma * math.sqrt(2*math.pi)
            # print('gaussianFit ZLP areas= ' + str(ZLP))
        else:
            raise 'getT_over_IMFP did not receive a valid method.'
        Total = simps(self.spectrum)
        return np.log(Total/ZLP)

def getlnI_I0OverPoints(ds, method, skippoints=1, plot: bool = False):
    # Increase skippoints to lower the number of plots processed.
    thicknesses = []
    for i in range(0, len(ds['data']), skippoints):
        I_I0 = getEELSAreas(ds, i).getT_over_IMFP(method=method)
        thicknesses.append(I_I0)
    if plot:
        plt.plot(thicknesses)
        plt.show()
    return thicknesses

def getThicknessOverCoords(empty_cell, full_cell, method, skippoints=10, IMFP_H20=117.95150547696309):
    avg_SiNx_lnI_I0 = np.mean(getlnI_I0OverPoints(empty_cell, method))

    water_lnI_I0 = getlnI_I0OverPoints(full_cell, method)

    thicknessWater = []
    for val in water_lnI_I0:
        water_T_over_IMFP = val - avg_SiNx_lnI_I0
        thicknessWater.append(water_T_over_IMFP * IMFP_H20)

    loc = full_cell['pixelUnit'].index('µm')
    pixelsize = full_cell['pixelSize'][loc] * skippoints
    #print(full_cell['pixelUnit'], full_cell['pixelSize'])

    return thicknessWater, pixelsize

def getDistAxis(ds, datarange=('min','max')):
    # Converts pixel coordinates to distances.
    # TODO the scale seems off, fix this
    if datarange[0] == 'min':
        minim = 0
    else:
        minim = datarange[0]
    if datarange[1] == 'max':
        maxim = len(ds['coords'][0])
    else:
        maxim = datarange[1]

    distAxis = []
    zero = ds['coords'][0][minim]
    scale = ds['pixelSize'][0]

    if window5nm['pixelUnit'][0] == 'µm':
        scale = scale * 1000
    elif window5nm['pixelUnit'][0] == 'nm':
        pass
    else:
        sys.exit("pixelUnit value is ambiguous.")

    for val in ds['coords'][0][minim:maxim]:
        distAxis.append((val-zero) * scale)

    return distAxis, 'nm'

def downsample(xdata, ydata, reduce):
    newx =[]
    newy = []
    totalDataPoints = len(xdata)
    for i in range(0, totalDataPoints, int(max(1, reduce))):
        newx.append(xdata[i])
        newy.append(ydata[i])
    return newx, newy

def smoothData(xlist, ylist, totalpoints: bool = 300):
    # Smooth data.
    xlist_smooth = np.linspace(min(xlist), max(xlist), totalpoints)
    spl = make_interp_spline(xlist, ylist, k=2)  # type: BSpline
    ylist_smooth = spl(xlist_smooth)
    return xlist_smooth, ylist_smooth

class InelasticTransmission:
    def __init__(self, beamE, collection_angle):
        self.beamE = beamE
        self.collection_angle = collection_angle

    def inelMFPMalis(self, Zeff, AEL=False):
        if AEL == False:
            AEL = 7.6 * (Zeff ** 0.36)
            print("The average energy loss (Em, eV) was calculated to be ", str(AEL))
        F = (1 + self.beamE/1022)/((1 + self.beamE/511)**2)
        inelMFP = (106 * F * self.beamE) / (self.beamE * np.log(2 * self.collection_angle * self.beamE / AEL))
        print("The mean free path of electric scattering was calculated (using Malis et al.) to be ", str(inelMFP))
        return inelMFP

    def inelMFPIakoubovski(self, density, CSA=False, sat_fact=20):
        F = (1 + self.beamE/1022)/((1 + self.beamE/511)**2)
        if CSA == False:
            CSA = 5.5 * ((density ** 0.3) / (F * self.beamE))
        inelMFP = ((200 * F * self.beamE) / (11 * (density ** 0.3))) / np.log((1 + self.collection_angle ** 2 / CSA ** 2) / (1 + self.collection_angle ** 2 / sat_fact ** 2))
        print("The mean free path of electric scattering was calculated (using Iakoubovski et al.) to be ", str(inelMFP))
        return inelMFP


if __name__ == '__main__':
    ds = dm.dmReader("data/empty_5nmcell_linescan_Feb10_alignedZLP/EELS Spectrum Image aligned.dm4")
    # print(ds['pixelUnit'][0])
    # print(TotalEELSAreaAcrossScan(ds, start=150))
    # plt.plot(ds['data'][85])
    # plt.show()



    beamE = 300  # keV
    collection_angle = 53.4375  # mrad
    Zeff_SiN = 10.36  # Si3N4
    density_SiN = 3.17  # g/cm3

    iTransmission = InelasticTransmission(beamE=beamE, collection_angle=collection_angle)
    IMFPSiN_malis = iTransmission.inelMFPMalis(Zeff=Zeff_SiN)
    IMFPSiN_iakoubovski = iTransmission.inelMFPIakoubovski(density=density_SiN)
    # interpolated from riley = 72.67

    # Trim data to usable spectra (inside window).
    cutoff = 163
    ds['data'] = ds['data'][cutoff:]

    # Get thickness data.
    lnI_I0O = pd.DataFrame(getlnI_I0OverPoints(ds, method='simple'))
    t_malis = lnI_I0O * IMFPSiN_malis
    t_iakoubovski = lnI_I0O * IMFPSiN_iakoubovski
    t_riley = lnI_I0O * 72.67

    # Get scan distance data.
    distance, unit = getDistAxis(ds, (cutoff,'max'))

    print(distance)

    plt.plot(distance, t_malis, label='malais')
    plt.plot(distance, t_iakoubovski, label='iakoubovski')
    plt.plot(distance, t_riley, label='riley')
    plt.ylabel('Thickness (nm)')
    plt.xlabel('Distance from window edge (' + str(unit) + ')')
    plt.legend()
    plt.show()

    sys.exit(0)

    # Talos_empty_cell_ds1 = dm.dmReader("data/LiquidThicknessPaper/Talos 200/Empty_assembly/Linescan1/BL-STEM SI.dm4", dSetNum=2)
    # Talos_empty_cell_ds2 = dm.dmReader("data/LiquidThicknessPaper/Talos 200/Empty_assembly/Linescan2/TR-STEM SI.dm4", dSetNum=2)
    # Talos_water99nm_cell_ds = dm.dmReader("data/LiquidThicknessPaper/Talos 200/99nm_spacer/Linescan1_STEM_SI.dm4", dSetNum=2)
    # Talos_water550nm_cell_ds = dm.dmReader("data/LiquidThicknessPaper/Talos 200/550nm_spacer/Linescan1_STEM_SI.dm4", dSetNum=2)
    #
    # Titan_empty_cell_ds = dm.dmReader("data/LiquidThicknessPaper/Titan_HB/Empty_assembly/Linescan1/Linescan1 EELS Spectrum Image.dm4")
    # Titan_water181nm_cell_ds = dm.dmReader("data/LiquidThicknessPaper/Titan_HB/181nm_spacer/Linescan1EELS Spectrum Image.dm4")
    # Titan_water632nm_cell_ds = dm.dmReader("data/LiquidThicknessPaper/Titan_HB/632nm_spacer/EELS Spectrum Image.dm4")
    #
    # # Print TotalEELSArea as a function of coodinate (to see if window was reached at (0,0)).
    # Talos_empty_cell_ds1['data'] = Talos_empty_cell_ds1['data'][58:]
    # Talos_empty_cell_ds2['data'] = Talos_empty_cell_ds2['data'][55:]
    # Titan_empty_cell_ds['data'] = np.delete(Titan_empty_cell_ds['data'], (6143), axis=0)
    #
    # TotalEELSAreaAcrossScan(Talos_empty_cell_ds1)
    #
    # talos = np.mean(getlnI_I0OverPoints(Talos_empty_cell_ds1, 'gaussianFit'))
    # titan = np.mean(getlnI_I0OverPoints(Titan_empty_cell_ds, 'gaussianFit'))
    # print(talos)
    # print(titan)
    #
    # spacer99 , size99 = getThicknessOverCoords(Talos_empty_cell_ds1, Talos_water99nm_cell_ds, method = 'gaussianFit', IMFP_H20=144.97125724587494)
    # spacer550, size550 = getThicknessOverCoords(Talos_empty_cell_ds1, Talos_water550nm_cell_ds, method = 'gaussianFit', IMFP_H20=144.97125724587494)
    # spacer181, size181 = getThicknessOverCoords(Titan_empty_cell_ds, Titan_water181nm_cell_ds, method = 'gaussianFit', IMFP_H20=167.36301257265185)
    # spacer632, size632 = getThicknessOverCoords(Titan_empty_cell_ds, Titan_water632nm_cell_ds, method = 'gaussianFit', IMFP_H20=167.36301257265185)
    #
    # newDistspacer181, newspacer181 = downsample(getDistAxis(spacer181, size181), spacer181, 10)
    # newDistspacer632, newspacer632 = downsample(getDistAxis(spacer632, size632), spacer632, 10)
    #
    # plt.plot(newDistspacer181, newspacer181, label = 'spacer181')
    # plt.plot(newDistspacer632, newspacer632, label = 'spacer632')
    # plt.plot(getDistAxis(spacer99 , size99 ), spacer99 , label = 'spacer99')
    # plt.plot(getDistAxis(spacer550, size550), spacer550, label = 'spacer550')
    # plt.ylabel('Thickness (nm)')
    # plt.xlabel('Distance from window edge (microns)')
    # plt.legend()
    # plt.show()