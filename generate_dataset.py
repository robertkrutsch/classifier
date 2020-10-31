import numpy as np
from PIL import Image
import os
import random
import csv

class GenData(object):

    def __init__(self, path, size, overwrite):
        """
        Init the dataset.
        :param path: Path of the dataset
        :param size: Total number of files to be generated (currently 0.8 train, 0.1 validation and 0.1 test)
        :param overwrite: should we overwrite the files , but this needs to be 1 to write anything
        """
        self.width = 100
        self.height = 100
        self.path = path
        self.overwrite = overwrite
        self.size = size
        self.imgarray = np.zeros((self.width, self.height,3))

    def __create_dir(self, dirname):
        """
        This is a function that is used internally to create the dataset directory structure.
        :param dirName: directory to be created
        """
        try:
            # Create target Directory
            os.mkdir(dirname)
            print("Directory ", dirname, " Created ")
        except FileExistsError:
            print("Directory ", dirname, " already exists")

    def __create_random_circle(self):
        """
        Create a circle at a random location in the image.
        """
        cx = min(int(np.random.random() * self.width), self.width - 1)
        cy = min(int(np.random.random() * self.height), self.height - 1)
        radius = 5 + int(np.random.random() * 20)

        starty = max(cy - radius,0)
        stopy  = min(cy + radius, self.height-1)
        startx = max(cx - radius, 0)
        stopx  = min(cx + radius, self.width - 1)

        r = min(int(np.random.random() * 255), 255)
        g = min(int(np.random.random() * 255), 255)
        b = min(int(np.random.random() * 255), 255)

        for y in range(starty, stopy + 1, 1):
            for x in range(startx, stopx + 1, 1):
                incircle = (cx-x)**2 + (cy-y)**2
                if incircle**0.5 <= radius:
                    self.imgarray[y, x, 0] = r
                    self.imgarray[y, x, 1] = g
                    self.imgarray[y, x, 2] = b

    def __create_random_rectangle(self):
        """
        Create a rectangle at a random location in the image.
        """
        # generate the top and bottom of a rectangle, we should not go over the boudaries
        topx = min(int(np.random.random() * self.width), self.width - 1)
        topy = min(int(random.random() * self.height), self.height - 1)
        botx = min(int(random.random() * self.width), self.width - 1)
        boty = min(int(random.random() * self.height), self.height - 1)

        r = min(int(np.random.random() * 255), 255)
        g = min(int(np.random.random() * 255), 255)
        b = min(int(np.random.random() * 255), 255)
        # swap coordinates such that top has lower coordinates vs bottom
        if topx > botx:
            tmp = topx
            topx = botx
            botx = tmp

        if topy > boty:
            tmp = topy
            topy = boty
            boty = tmp

        for y in range(topy, boty + 1, 1):
            for x in range(topx, botx + 1, 1):
                self.imgarray[y, x, 0] = r
                self.imgarray[y, x, 1] = g
                self.imgarray[y, x, 2] = b

    def __create_images(self, dirName, percent, start):
        """
        Generate the image files and the csv file that list the file and the label. The images can have rectangles, circles or nothing.
        :param dirName: Path name where to create the image.
        :param percent: percentage from the size. This gives how many files will be created.
        :param start: Start index from where we beggin writing.
        """
        file = open(self.path + '/' + dirName+'.csv', 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(["id", "FileName", "Type"])

        for i in range(int(self.size * percent)):

            r = min(int(np.random.random() * 255), 255)
            g = min(int(np.random.random() * 255), 255)
            b = min(int(np.random.random() * 255), 255)

            self.imgarray = np.zeros((self.width, self.height,3)) # cleanup the image
            self.imgarray[:, :, 0] = r
            self.imgarray[:, :, 1] = g
            self.imgarray[:, :, 2] = b

            rnd_tmp = random.random()

            if rnd_tmp < 0.33:
                self.__create_random_rectangle()
                writer.writerow([i + 1, dirName + '_' + str(start+i) + '.png', "1"])
            elif rnd_tmp > 0.66:
                self.__create_random_circle()
                writer.writerow([i + 1, dirName + '_' + str(start+i) + '.png', "0"])
            else:
                writer.writerow([i + 1, dirName + '_' + str(start + i) + '.png', "2"])


            im = Image.fromarray(self.imgarray.astype(np.uint8))
            im.save(self.path + '/' + dirName + '/' + dirName + '_' + str(start+i) + '.png')

        file.close()

    def create_dataset(self, start_train, start_valid,start_test):
        """
        Create the dataset.
        :param start_train: file id from where to start in training dataset
        :param start_valid:file id from where to start in valid dataset
        :param start_test: file id from where to start in test dataset
        """
        if self.overwrite == 1:
            '''
             Create directory structure.
            '''
            self.__create_dir(self.path)
            self.__create_dir(self.path + '/train')
            self.__create_dir(self.path + '/valid')
            self.__create_dir(self.path + '/test')
            '''
            Create images.
            '''
            self.__create_images('train', 0.8, start_train)
            self.__create_images('valid', 0.1, start_valid)
            self.__create_images('test' , 0.1, start_test)


'''
    Create the dataset 
'''

#a = GenData(path='/media/robert/80C2-37E4/multi_task/dataset', size=10000, overwrite=1)
#a.create_dataset(0, 0, 0)
#print('Done')








