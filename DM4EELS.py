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
from bokeh.plotting import figure, show
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ncempy.io import dm
from scipy.integrate import simps
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline, BSpline
from scipy import arange


def find_index_closest(arr, val):
    return np.abs(arr - val).argmin()


def alignZLPbyHighestPeak(ds, rangeVals: list):
    # x = ds['xdata'][0]
    # y = ds['data'][0]

    x_step = ds['pixelSize'][1]
    n_data_points = len(ds['xdata'])
    for i in range(0, len(ds['data'])):
        # convert range to indexes.

        index_range = [find_index_closest(ds['xdata'][i], rangeVals[0]),
                       find_index_closest(ds['xdata'][i], rangeVals[1])]

        # print(index_range)
        # Get index for peak in range in y data.
        arr = ds['data'][i][index_range[0]: index_range[1]]
        peak_index = np.asarray(arr).argmax() + index_range[0]

        # Shift x data so peak appears at x = 0.
        if ds['xdata'][i][peak_index] == 0:
            continue
        else:
            # print(ds['xdata'][i])
            starting_x = - peak_index * x_step + x_step
            newX = []
            for i in range(n_data_points):
                newX.append(starting_x)
                starting_x = starting_x + x_step
            ds['xdata'][i] = newX

    return ds


def TotalEELSAreaAcrossScan(ds, start=0, end=-1, show=False):
    # Calculates the area of the spectrum at each point, area will be lower than the rest of the dataset when there is no EELS signal.
    areas = []
    if end == -1 or end > len(ds['data']):
        end = len(ds['data'])
    else:
        pass

    for i in range(start, end):
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
        plt.title(
            'Gaussian fit,  $f(x; A, \mu, \sigma) = \\frac{A}{\sigma\sqrt{2\pi}} e^{[{-{(x-\mu)^2}/{{2\sigma}^2}}]}"$')
        plt.show()

    return A, mu, sigma


def showEELS(ds, n):
    plt.plot(ds['xdata'][n], ds['data'][n])
    plt.show()


class getEELSAreas:
    def __init__(self, ds, n):
        self.spectrum = ds['data'][n]
        self.coords = ds['coords'][1]  # what is coords 0?

    def showEELSInteractive(self):

        x = self.coords
        y = self.spectrum

        tools = ["box_select", "hover", "reset", "box_zoom", "wheel_zoom"]
        p = figure(title="Simple line example", x_axis_label='x', y_axis_label='y', tools=tools)
        p.line(x, y, legend_label="Temp.", line_width=2)
        show(p)

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
        if len(Ysubset_peak2peak) == 0:
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

    def getT_over_IMFP(self, method='simple', bounds=[-1, -1]):
        '''

        :param method: simple, or gaussianFit
        :return:
        '''
        highest_peak_index, second_highest_peak_index = self.getPeakInts()
        # If no bounds for ZLP are given, use getSimpleBounds.
        if bounds == [-1, -1]:
            X1index, X2index = self.getSimpleBounds(highest_peak_index, second_highest_peak_index)
        else:
            X1index = bounds[0]
            X2index = bounds[1]
        if method == 'simple':
            ZLP = simps(self.spectrum[X1index:X2index], dx=1)
            # print('simple ZLP area= ' + str(ZLP))
        elif method == 'gaussianFit':
            A, mu, sigma = gaussianFit(self.spectrum[X1index:X2index])
            # sigma is standard deviation, calculate area analytically.
            ZLP = A / (sigma * math.sqrt(2 * math.pi)) * sigma * math.sqrt(2 * math.pi)
            # print('gaussianFit ZLP areas= ' + str(ZLP))
        else:
            raise 'getT_over_IMFP did not receive a valid method.'
        Total = simps(self.spectrum)
        return np.log(Total / ZLP)


def getlnI_I0OverPoints(ds, method, bounds=[-1, -1], skippoints=1, plot: bool = False):
    # Increase skippoints to lower the number of plots processed.
    thicknesses = []
    for i in range(0, len(ds['data']), skippoints):
        I_I0 = getEELSAreas(ds, i).getT_over_IMFP(method=method, bounds=bounds)
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
    # print(full_cell['pixelUnit'], full_cell['pixelSize'])

    return thicknessWater, pixelsize


def getDistAxis(ds, datarange=('min', 'max')):
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

    if ds['pixelUnit'][0] == 'µm':
        scale = scale * 1000
    elif ds['pixelUnit'][0] == 'nm':
        pass
    else:
        sys.exit("pixelUnit value is ambiguous.")

    for val in ds['coords'][0][minim:maxim]:
        distAxis.append((val - zero) * scale)

    return distAxis, 'nm'


def downsample(xdata, ydata, reduce):
    newx = []
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
            # print("The average energy loss (Em, eV) was calculated to be ", str(AEL))
        F = (1 + self.beamE / 1022) / ((1 + self.beamE / 511) ** 2)
        inelMFP = (106 * F * self.beamE) / (AEL * np.log(2 * self.collection_angle * self.beamE / AEL))
        print("The mean free path of electric scattering was calculated (using Malis et al.) to be ", str(inelMFP))
        return inelMFP

    def inelMFPIakoubovski(self, density, sat_fact, CSA=False):
        F = (1 + self.beamE / 1022) / ((1 + self.beamE / 511) ** 2)
        if CSA == False:
            CSA = 5.5 * ((density ** 0.3) / (F * self.beamE))
        inelMFP = ((200 * F * self.beamE) / (11 * (density ** 0.3))) / np.log(
            (1 + self.collection_angle ** 2 / CSA ** 2) / (1 + self.collection_angle ** 2 / sat_fact ** 2))
        print("The mean free path of electric scattering was calculated (using Iakoubovski et al.) to be ",
              str(inelMFP))
        return inelMFP


def plotWaterThickness(ds, windowThickness, IMFPSiN, IMFPH2O, start=0, end=-1, bounds=[-1, -1]):
    if end == -1 or end > len(ds['data']):
        end = len(ds['data'])
    else:
        pass

    # Trim data to usable spectra (inside window).
    ds['data'] = ds['data'][start:end]

    # Get thickness data.
    lnI_I0O = pd.DataFrame(getlnI_I0OverPoints(ds, method='simple', bounds=bounds))

    t_water = (lnI_I0O - windowThickness / IMFPSiN) * IMFPH2O

    # Get scan distance data.
    distance, unit = getDistAxis(ds, (start, end))

    plt.plot(distance, t_water)
    plt.ylabel('Thickness (nm)')
    plt.xlabel('Distance from window edge (' + str(unit) + ')')
    plt.legend()
    plt.show()


def getZLParea(data, bounds=[-1, -1]):
    # If no bounds for ZLP are given, use getSimpleBounds.

    if bounds == [-1, -1]:
        peak_indices, peak_dict = find_peaks(data, height=0.3, distance=20, width=2)
        peak_heights = peak_dict['peak_heights']
        highest_peak_index = peak_indices[np.argmax(peak_heights)]
        second_highest_peak_index = peak_indices[np.argpartition(peak_heights, -2)[-2]]

        Ysubset_peak2peak = data[highest_peak_index:second_highest_peak_index]
        Yval_min = min(Ysubset_peak2peak)
        index_min_in_subset = np.where(Ysubset_peak2peak == Yval_min)[0]
        X2index = index_min_in_subset[0] + highest_peak_index
        index_step = X2index - highest_peak_index
        X1index = highest_peak_index - index_step
    else:
        X1index = bounds[0]
        X2index = bounds[1]
    ZLP = simps(data[X1index:X2index], dx=0.5)  # TODO define dispersion

    return ZLP


if __name__ == '__main__':
    ds = dm.dmReader(
        "data/Spectra_March28_2023/60kV-17nmNanorod 50meVch 2e-4s 57kx 4nm/SI data (6)/EELS Spectrum Image.dm4")
    # [x on energy loss axis][y on 2d scan][x on 2d scan] is count when x,y is 0
    x = 10
    y = 10
    eelsdata = []
    for i in range(ds['data'].shape[0]):
        eelsdata.append(ds['data'][i][x][y])
    plt.plot(eelsdata)
    plt.show()

    print(getZLParea(eelsdata))
    print(simps(eelsdata))

    # ds['xdata'] = []
    # for i in range(0, len(ds['data'])):
    #     ds['xdata'].append(ds['coords'][1])
    #
    # TotalEELSAreaAcrossScan(ds, start=150)
    # ds['data'] = ds['data'][150:]
    # ds['xdata'] = ds['xdata'][150:]
    # alignZLPbyHighestPeak(ds, [250, 265])
    # showEELS(ds, 0)

    # beamE = 300  # keV
    # collection_angle = 53.4375  # mrad
    # Zeff_SiN = 10.36  # Si3N4
    # density_SiN = 3.17  # g/cm3
    # IMFP_H20 = 117.95150547696309
    # getEELSAreas(ds, 50).showEELSInteractive()
    # bounds = [179,191]

    # TODO align ZLP before chosing peak
    # IMFPSiN_malis = InelasticTransmission(beamE=beamE, collection_angle=collection_angle).inelMFPMalis(Zeff=Zeff_SiN)
    # plotWaterThickness(ds, windowThickness=5, IMFPSiN=IMFPSiN_malis, IMFPH2O=IMFP_H20, start=0, end=270, bounds=bounds)

    # print(len(ds['data'][1]))
    # plt.plot(ds['coords'][1], ds['data'][1])
    # plt.xlim(150, 187)
    # plt.ylim(0, 0.05*10**6)

    # plt.ylabel('Intensity (counts)')
    # plt.xlabel('Energy loss (keV)')
    # plt.savefig('data/EELS_spectrum.pdf')
    # plt.show()

    # print(getlnI_I0OverPoints(ds, method='gaussianFit', bounds=[175,187])[0])

    # ds = dm.dmReader("data/empty_5nmcell_linescan_Feb10_alignedZLP/EELS Spectrum Image aligned.dm4")
    # print(ds['pixelUnit'][0])
    # print(TotalEELSAreaAcrossScan(ds, start=150))
    # plt.plot(ds['data'][-10])
    # plt.xlim(200, 300)
    # plt.ylim(0, 20000)
    # plt.ylabel('Intensity (counts)')
    # plt.xlabel('Energy loss (keV)')
    # plt.savefig('data/EELS_spectrum.pdf')
    # plt.show()
    #
    # '''# Tyler's code for Titan
    # set_collection_angle = 55
    # Zeff_Si3N4 = 10.36
    # Em_Si3N4_eV = (7.6 * Zeff_Si3N4 ** 0.36)
    # E0_keV = 300
    # F_rel_factor = (1 + E0_keV / 1022) / (1 + E0_keV / 511) ** 2
    # collection_angle_mrad = set_collection_angle  # Talos 200: 23 mrad # Titan HB: 55 mrad
    # IMFP = (106 * F_rel_factor * E0_keV) / (Em_Si3N4_eV * np.log((2 * collection_angle_mrad * E0_keV) / Em_Si3N4_eV))
    # print(IMFP)'''
    # # Test Titan calc with my code.
    # # testc = InelasticTransmission(beamE=300, collection_angle=55)
    # # testmfpm = testc.inelMFPMalis(Zeff=10.36)
    #
    #
    # beamE = 300  # keV
    # collection_angle = 53.4375  # mrad
    # Zeff_SiN = 10.36  # Si3N4
    # density_SiN = 3.17  # g/cm3
    #
    # IMFPSiN_malis = InelasticTransmission(beamE=beamE, collection_angle=collection_angle).inelMFPMalis(Zeff=Zeff_SiN)
    #
    # # Trim data to usable spectra (inside window).
    # cutoff = 250 #163
    # ds['data'] = ds['data'][cutoff:]
    #
    # # Get thickness data.
    # lnI_I0O = pd.DataFrame(getlnI_I0OverPoints(ds, method='simple'))
    # t_malis = lnI_I0O * IMFPSiN_malis
    # print(np.average(lnI_I0O))
    #
    # # Get scan distance data.
    # distance, unit = getDistAxis(ds, (cutoff,'max'))
    #
    # plt.plot(distance, t_malis, label='malais')
    # print(np.average(t_malis))
    # plt.ylabel('Thickness (nm)')
    # plt.xlabel('Distance from window edge (' + str(unit) + ')')
    # plt.legend()
    # # plt.show()
    #
    # sys.exit(0)

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
