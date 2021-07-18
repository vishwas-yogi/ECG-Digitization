#importing libraries
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import argparse

"""Defining Syntactic Sugar"""
#defining decorator
def decorator1(func):
    def inner(*args):
        image, title = func(*args)
        cv.imshow(title, image)
        cv.waitKey(0)
    return inner

class digitizer:
    def __init__(self, image_path):
        """ Constructor only requires image path"""
        self.image_path = image_path

    @decorator1
    def read_gray_image(self):
        """
        Reads the image and converts it into grayscale
        """

        image = cv.imread(self.image_path)
        self.image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        return self.image_gray, 'GrayImage'
      

    def histogram(self):
        """
        Plots histogram,
        plotted histogram is used to determine threshold value (using hit and trial),
        choose a value which removes the graphical grid from background
        """

        img_hist = cv.calcHist([self.image_gray], [0], None, [256], [0, 256])
        plt.figure('Grayscale Histogram')
        plt.xlabel('Bins')
        plt.ylabel("Number of pixels")
        plt.xlim([0, 257])
        plt.plot(img_hist)
        plt.show()
        
    @decorator1
    def binarizing_image(self, threshold= 185):
        """
        Binarizing image by thresholding
        Changing background color to black as morphological operations in opencv expects
        white foreground and black background
        """

        threshold, self.thresh = cv.threshold(self.image_gray, threshold, 255, cv.THRESH_BINARY)
        self.thresh = cv.bitwise_not(self.thresh)
        return self.thresh, f'Threshold image (threshold = {threshold})'


    def image_denoise(self, kernel= 1):
        """
        Applying Median Filtering
        params:
        kernel = shape of structuring element
        """

        self.thresh = cv.medianBlur(self.thresh, kernel)
        return self.thresh

    def morph_image(self, iterations = 4):
        """
        Applying morphological operations
        params:
        iterations = number of iterations of dilation operation
        """

        self.image_morph = cv.morphologyEx(self.thresh, cv.MORPH_CLOSE, (23, 23))
        self.image_morph = cv.dilate(self.image_morph, (1, 1), iterations= iterations)
        return self.image_morph

    def connected_components_plot(self):
        """
        Labels different connected component using different colors and plots them.
        It also prints the number of connected component in the image.
        """

        num_labels, labels = cv.connectedComponents(self.image_morph)
        print(f'Number of connected components are: {num_labels}')
        
        @decorator1
        def show_connected_components(self, labels):
            label_hue = np.uint8(179*labels/np.max(labels))
            blank_ch = 255*np.ones_like(label_hue)
            labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
            
            labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
            
            # set bg label to black
            labeled_img[label_hue==0] = 0

            return labeled_img, 'labeled image'

        show_connected_components(self, labels)


    def show_connected_component(self):
        """
        Plots and shows every connected component individually.
        """

        self.output = cv.connectedComponentsWithStats(self.image_morph, connectivity= 8, ltype = cv.CV_32S )
        (numLabels, labels, stats, centroids) = self.output
        for i in range(0, numLabels):
            
            if(i==0):
                text = 'examining component {}/{} (background)'.format(i+1, numLabels)
            
            else:
                text = 'examining component {}/{}'.format(i+1, numLabels)
                
            print('[INFO] {}'.format(text))
            
            output = self.opening.copy()
            componentmask = (labels==i).astype('uint8')*255
            cv.imshow("Output", output)
            cv.imshow('Connected Components', componentmask)
            cv.waitKey(0)

    @decorator1
    def largest_connected_component(self):
        """
        Finds the largest connected component, Plots it and prints the image matrix.
        """

        self.output = cv.connectedComponentsWithStats(self.image_morph, connectivity= 8, ltype = cv.CV_32S )
        (numLabels, labels, stats, centroids) = self.output
        
        largest_label = np.bincount(labels.flatten()[labels.flatten()!=0]).argmax()
        largest = (labels== largest_label).astype('uint8')*255
        print('Image Matrix is:')
        print(largest)

        return largest, 'largest_component'



"""construct the argument parse and parse the arguments"""
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required= True, help= 'Path to image')
args = vars(ap.parse_args())


#calling functions
a = digitizer(vars['path'])
a.read_gray_image()
a.histogram()
a.binarizing_image()
a.image_denoise()
a.morph_image()
a.connected_components_plot()
a.largest_connected_component()
