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
    def select_file():
        root = tk.Tk()
        path = fd.askopenfilename(filetypes=[("EMD files", ".emd")], title="Choose .emd file to analyze")
        print('Opening "' + str(path) + '"')
        root.destroy()
        return path

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
        print("Saving " + str(path))
        root.destroy()
        return path

    @staticmethod
    def autoProcessAsk():
        root = tk.Tk()
        decision = tk.messagebox.askquestion('Auto Process', 'Would you like to perform auto-processing?',
                                             icon='question')
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

    @staticmethod
    def parseFileName(file):
        return str(file).split("/")[-1].split(".")[0]


class frameExporter:

    @staticmethod
    def checkPath(path):
        # Check whether the specified path exists or not
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)

    @staticmethod
    def saveFrame(h5pyfile, name, path, filetype="jpg", framenum=0):
        frameExporter.checkPath(path)
        data = h5pyfile['Data/Image/' + navigate.getMemberName(h5pyfile, '/Data/Image/')]
        frame = np.array(data['Data'][:, :, framenum]).astype('uint8')
        rgbImage = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        pathname = path + '/' + name + '_frame' + str(frame) + '.' + filetype
        cv2.imwrite(pathname, rgbImage)

    @staticmethod
    def saveAllFrames(h5pyfile, originalfilename, filetype="jpg", auto=False):
        if auto:
            # Make folder for frame images.
            folderpath = originalfilename + "/"
            os.makedirs(folderpath)
            path = str(folderpath) + navigate.parseFileName(originalfilename)
        else:
            path = GUI.save_file("jpg", navigate.parseFileName(originalfilename), "Choose folder to save images frames",
                                 filetypeexclude=True)
        print("Saveframes path: " + str(path))
        # Save files
        data = h5pyfile['Data/Image/' + navigate.getMemberName(h5pyfile, '/Data/Image/')]

        for i in range(len(data['Data'][0][0])):
            frame = np.array(data['Data'][:, :, i]).astype('uint8')
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            outname = path + "_frame" + str(i) + "." + str(filetype)
            cv2.imwrite(outname, rgbImage)

    @staticmethod
    def getPath(originalfilename, multi, auto):
        if multi:
            # Create a folder with the same name as the original file.
            folderpath = originalfilename + "/"
            # Remove items in folder
            if os.path.exists(folderpath):
                removeFilesinDir(folderpath)
            else:
                os.makedirs(folderpath)
            path = str(folderpath) + navigate.parseFileName(originalfilename)
        else:
            # Keep the same path.
            path = originalfilename

        # Return file names without file type.
        if auto:
            return path
        else:
            return GUI.save_file("jpg", navigate.parseFileName(originalfilename), "Choose folder to save images frames",
                                 filetypeexclude=True)

    @staticmethod
    def saveMultiFrames(data, path, filetype='jpg'):
        data = np.array(data)
        for i in range(data.shape[-1]):
            plt.imshow(data[:, :, i], cmap=plt.cm.gray)
            plt.axis('off')
            pathname = path + "_frame" + str(i) + "." + str(filetype)
            # plt.savefig(pathname, dpi=300, bbox_inches='tight', pad_inches=0)
            plt.show()

    @staticmethod
    def saveOnlyFrame(data, path, filetype="jpg"):
        data = np.array(data)
        pathname = path + '.' + filetype
        plt.imshow(data, cmap=plt.cm.gray)
        plt.axis('off')
        # plt.savefig(pathname, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.show()

    @staticmethod
    def frameOutput(h5pyfile, originalfilename, auto):

        data = h5pyfile['Data/Image/' + navigate.getMemberName(h5pyfile, '/Data/Image/') + '/Data']

        nframes = h5pyfile['Data/Image/' + navigate.getMemberName(h5pyfile, '/Data/Image/') + '/Data'].shape[-1]
        # Check if multiple frames (video) and save.
        if nframes > 1:
            # Get output path, folder for multiple images.
            path = frameExporter.getPath(originalfilename, multi=True, auto=auto)
            frameExporter.saveMultiFrames(data, path)
            print('multiple frames')
        elif nframes == 1:
            # Get output path, for one images.
            path = frameExporter.getPath(originalfilename, multi=False, auto=auto)
            print('only one frame')
            frameExporter.saveOnlyFrame(data, path)
        else:
            sys.exit("In frameOutput, len(data['Data'][0][0]) is " + str(len(data['Data'][0][0])))


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

    def __init__(self, h5pyfile):
        metalocation = navigate.getMemberName(h5pyfile, '/Data/Image/')  # CAUTION, may break.
        self.meta = h5pyfile['/Data/Image/' + str(metalocation) + '/Metadata']
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

    def getCSVmetadata(self, originalfilename, filter=None, auto=False):
        print("Parsing metadata.")
        if auto:
            pathname = str(originalfilename) + ".csv"
        else:
            pathname = GUI.save_file("csv", navigate.parseFileName(originalfilename),
                                     "Choose place to save metadata file")
        print("Saving metadata path: " + str(pathname))
        out = []
        cols = list(pd.json_normalize(self.convertASCII(0)).columns.values)

        for i in range(self.nframes):
            jsondict = self.convertASCII(i)
            print(jsondict)
            items = []
            self.flattenAndCollect(jsondict, items)
            out.append(items)

        df = pd.DataFrame(out, columns=cols)
        if filter is None:
            print("No filter, outputting all metadata.")
            df.to_csv(pathname)
        else:
            print("Filtering metadata.")
            newdf = df[filter]
            newdf.to_csv(pathname)

    def getMetaAllFrames(self, query, printoption):
        out = []
        m = self.convertASCII(0)
        if printoption:
            for i in m:
                print(i)
                for g in m[i]:
                    print("--- " + str(g))

        for i in range(self.nframes):
            meta = self.convertASCII(i)
            if query == 'mag':
                out.append(meta['CustomProperties']['StemMagnification']['value'])
            elif query == 'sclbr':
                out.append(meta['BinaryResult']['PixelSize']['width'])  # x and y should be the same
                out.append(meta['BinaryResult']['PixelUnitX'])
            else:
                out.append('NA')
        return out


# Need to fix these names to be more general/pick columns.
filter = ["Optics.Apertures.Aperture-1.Diameter", "Optics.Apertures.Aperture-2.Diameter",
          "BinaryResult.ImageSize.width", "BinaryResult.ImageSize.height",
          "BinaryResult.PixelSize.width", "BinaryResult.PixelSize.height", "BinaryResult.PixelUnitX",
          "BinaryResult.PixelUnitY",
          "CustomProperties.Detectors[SuperXG22].IncidentAngle.value"]

filter = None

# Import file with tkinter selection.
file = GUI.select_file()
f = h5py.File(file, 'r')

plainpathname = str(file.replace('.emd', ''))
print(plainpathname)

# Auto processing ask.
if GUI.autoProcessAsk() == "yes":
    auto = True
else:
    auto = False

frameExporter.frameOutput(f, originalfilename=plainpathname, auto=auto)
# metadata(f).getCSVmetadata(originalfilename=plainpathname, filter=filter, auto=True)

# if GUI.autoProcessAsk() == "yes":
#
#     # Export all frames to folder.
#     frameExporter.frameOutput(f, originalfilename=plainpathname)
#
#     # Export important metadata to csv.
#     # metadata(f).getCSVmetadata(originalfilename=plainpathname, filter=filter, auto=True)
# else:
#     # Export all frames to folder.
#     frameExporter.saveAllFrames(f, originalfilename=plainpathname)
#
#     # Export important metadata to csv.
#     # metadata(f).getCSVmetadata(originalfilename=plainpathname)
