import sys

import matplotlib.pyplot as plt
from collections.abc import MutableMapping
import cv2
import h5py
import numpy as np
import io
import os
import glob
import csv
from PIL import Image
import ujson
import pandas as pd
import datetime as dt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo


def removeFilesinDir(folderpath):
    files = glob.glob(str(folderpath) + '*')
    for f in files:
        os.remove(f)


class GUI:
    @staticmethod
    def select_files():
        root = tk.Tk()
        paths = fd.askopenfilenames(filetypes=[("EMD files", ".emd")], title="Choose .emd file(s).")
        paths = root.tk.splitlist(paths)
        root.destroy()
        return paths

    @staticmethod
    def save_file(filetype, initialfilename, windowtext, filetypeexclude=False):
        root = tk.Tk()
        types = []
        if filetype == "csv":
            types.append(("CSV file", ".csv"))
        if filetype == "jpeg" or filetype == "jpg":
            types.append(("JPEG file", ".jpg"))
        if filetype == "tif" or filetype == "tiff":
            types.append(("TIFF file", ".tif"))
        path = fd.asksaveasfilename(filetypes=types, initialfile=initialfilename, title=windowtext)
        if filetypeexclude:
            if "." + str(filetype) in path:
                ext = "." + str(filetype)
                path = path.replace(ext, '')
        else:
            if "." + str(filetype) not in path:
                path = path + "." + str(filetype)
        root.destroy()
        return path

    @staticmethod
    def ProcessAsk(question: str):
        root = tk.Tk()
        decision = tk.messagebox.askquestion('', question, icon='question')
        root.destroy()
        return decision


class navigate:

    @staticmethod
    def getGroupsNames(group):
        items = []
        for item in group:
            if group.get(item, getclass=True) == h5py._hl.group.Group:
                items.append(group.get(item).name)
        print(items)

    @staticmethod
    def getGroup(group, item):
        if group.get(item, getclass=True) == h5py._hl.group.Group:
            return group.get(item)

    @staticmethod
    def getSubGroup(group, path):
        return group[path]

    @staticmethod
    def getDirectoryMap(group):
        for item in group:
            # check if group
            if group.get(item, getclass=True) == h5py._hl.group.Group:
                item = group.get(item)
                # check if emd_group_type
                # if 'emd_group_type' in item.attrs:
                print('found a group emd at: {}'.format(item.name))
                # process subgroups
                if type(item) is h5py._hl.group.Group:
                    navigate.getDirectoryMap(item)
                else:
                    print('found an emd at: {}'.format(item))
                    # print(type(item))

    @staticmethod
    def getMemberName(group, path):
        members = list(group[path].keys())
        if len(members) == 1:
            return str(members[0])
        else:
            return members


class EMDreader:
    """
    Unpacks emd data files with the h5py package, by navigating subdirectories.

    Parameters
    ----------
    path : str
        Path to file, NOT a list of paths.
    """

    def __init__(self, singlePath: str):
        self.path = singlePath
        self.singleH5pyObject = h5py.File(singlePath, 'r', driver='core')

    @staticmethod
    def convertASCII(transposed_meta, frame):
        ascii_meta = transposed_meta[frame]
        metadata_text = ''.join(chr(i) for i in ascii_meta)
        ASCii = metadata_text.replace("\0", '')
        return ujson.loads(ASCii)

    def unpackMetadata(self):

        # TODO add implementation to search subfolders if the format is not the Velox default.

        try:
            metadata = self.singleH5pyObject[
                'Data/Image/' + navigate.getMemberName(self.singleH5pyObject, '/Data/Image/') + '/Metadata']
            transposed_meta = [list(i) for i in zip(*(metadata[:]))]
        except:
            raise ValueError("Metadata was not able to be read, see unpackMetadata function.")

        # Unpack metadata dictionaries.
        meta = {}
        nframes = metadata.shape[-1]
        for i in range(nframes):
            meta[i] = self.convertASCII(transposed_meta, i)

        return meta

    def unpackData(self):

        # TODO add implementation to search subfolders if the format is not the Velox default.

        try:
            data = self.singleH5pyObject[
                'Data/Image/' + navigate.getMemberName(self.singleH5pyObject, '/Data/Image/') + '/Data']
            data = np.array(data)
            return data

        except:
            raise ValueError("File was not able to be read, see unpackData function.")

    def parseEMDdata(self):

        data = self.unpackData()
        # Shape single frame data by removing final channel.
        if data.shape[-1] == 1:
            data = data.reshape(data.shape[0], data.shape[1])
            return data, False
        # Shape multiple frame data with transpose.
        else:
            # data = data[...].transpose()
            return data, True


class frameExporter:
    '''
    fullFilePath = full path of original file.
    multi = True for multiple frame data, false for single frame data.
    '''

    def __init__(self, fullFilePath: str, multi: bool):
        self.fileName = os.path.splitext(os.path.basename(fullFilePath))[0]
        self.filePath = os.path.dirname(fullFilePath)
        self.multi = multi

    def get_output_path(self):
        if self.multi:
            targetFolderPath = os.path.join(self.filePath, self.fileName)
            if os.path.exists(targetFolderPath):
                # if GUI.ProcessAsk('Existing folder found, for '+str(self.fileName)+' would you like to overwrite data?'):
                #     print('Clearing existing folder.')
                #     removeFilesinDir(targetFolderPath)
                # else:
                #     print('Skipping file.')
                return None
            else:
                print('No existing folder found.')
                os.makedirs(targetFolderPath)
            return targetFolderPath
        else:
            return self.filePath

    def save_frames(self, data, dpi=300, filetype='jpg', show=False, nosave=False):
        outputPath = self.get_output_path()
        if outputPath == None:
            return
        print('Output path is: ' + str(outputPath))

        if self.multi:
            for i in range(data.shape[-1]):
                plt.imshow(data[:, :, i], cmap=plt.cm.gray)
                plt.axis('off')
                pathname = os.path.join(outputPath, self.fileName + "_frame_" + str(i) + "." + str(filetype))
                if not nosave:
                    plt.savefig(pathname, dpi=dpi, bbox_inches='tight', pad_inches=0)
                if show:
                    plt.show(bbox_inches='tight', pad_inches=0)
                else:
                    plt.clf()
        else:
            data = np.array(data)
            pathname = os.path.join(outputPath, self.fileName + '.' + filetype)
            plt.imshow(data, cmap=plt.cm.gray)
            plt.axis('off')
            if not nosave:
                plt.savefig(pathname, dpi=dpi, bbox_inches='tight', pad_inches=0)
            if show:
                plt.show(bbox_inches='tight', pad_inches=0)
            else:
                plt.clf()


class videoViewer:

    @staticmethod
    def playVideo(video, speed):
        while (video.isOpened()):
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            if cv2.waitKey(speed) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

    @staticmethod
    def showFrame(video, frame_number):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        _, frame = video.read()
        greyframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame ' + str(frame_number), greyframe)
        cv2.waitKey(5)

    @staticmethod
    def showGreyImage(img):
        plt.imshow(img, cmap='gray')
        plt.show()

    @staticmethod
    def showRGBImage(img):
        col_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.imshow(col_img)
        plt.show()


class metadata:

    def __init__(self, fullFilePath: str):
        self.fileName = os.path.splitext(os.path.basename(fullFilePath))[0]
        self.filePath = os.path.dirname(fullFilePath)

        self.singleH5pyObject = h5py.File(fullFilePath, 'r', driver='core')
        metalocation = navigate.getMemberName(self.singleH5pyObject, '/Data/Image/')  # CAUTION, may break.
        self.meta = self.singleH5pyObject['/Data/Image/' + str(metalocation) + '/Metadata']
        self.nframes = self.meta.shape[1]
        self.transposed_meta = [list(i) for i in zip(*(self.meta[:]))]

    def convertASCII(self, frame):
        ascii_meta = self.transposed_meta[frame]
        metadata_text = ''.join(chr(i) for i in ascii_meta)
        ASCii = metadata_text.replace("\0", '')
        return ujson.loads(ASCii)

    @staticmethod
    def flattenAndCollect(jsondict, items):
        for _, v in jsondict.items():
            if isinstance(v, MutableMapping):
                metadata.flattenAndCollect(v, items)
            else:
                items.append(v)

    def getCSVmetadata(self, transpose=False):
        pathname = os.path.join(self.filePath, self.fileName + '.csv')

        out = []
        cols = list(pd.json_normalize(self.convertASCII(0)).columns.values)

        for i in range(self.nframes):
            jsondict = self.convertASCII(i)
            items = []
            self.flattenAndCollect(jsondict, items)
            out.append(items)

        df = pd.DataFrame(out, columns=cols)

        if transpose:
            df = df.transpose()

        df.to_csv(pathname)


# Import file with tkinter selection.
fullFilePaths = GUI.select_files()

for file in fullFilePaths:
    metadata(file).getCSVmetadata(transpose=True)

for file in fullFilePaths:
    data, multi = EMDreader(file).parseEMDdata()
    frameExporter(file, multi).save_frames(data)
