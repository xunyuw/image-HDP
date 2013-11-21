from os.path import exists, isdir, basename, join, splitext
import cv2, sys
import math
from glob import glob
import scipy.cluster.vq as vq
from cPickle import dump, HIGHEST_PROTOCOL
import numpy as np
import subprocess
import operator
#from gensim import corpora, models, similarities
#from hdp import *

EXTENSIONS = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
K_THRESH = 1  # early stopping threshold for kmeans originally at 1e-5, increased for speedup
CODE_BOOK = 'code_book.file'
TEST_DATA_PATH = './HDPData/test.dat'
OUTPUT_PATH = './HDPData/'
CLUSTERS_NUM = 1500
FLANN_INDEX_KDTREE = 0

#HDP para
ITERS_NUM = '300'
GAMMA_B = '0.1'
ALPHA_B = '1.0'


class imageHDP():
    def __init__(self, path):
        self.img_matrix = []
        self.img_path = path
        self.img_files = []
        self.code_book = []
        self.img_descriptor = []
        self.all_word_histograms = {}
        self.img_word_list = []
        self.file_name_matrix = []
        self.test_data_path = TEST_DATA_PATH
        self.data_path = ''

    def get_img_files(self, path):
        self.img_files.extend(
            [join(path, basename(file_name)) for file_name in glob(path + "/*") if
             splitext(file_name)[-1].lower() in EXTENSIONS])

    def gen_code_book(self):
    #print self.imgPath
        if not self.img_path:
            raise "None data path set!"

        self.get_img_files(self.img_path)

        #print self.imgFiles
        if not self.img_files:
            raise "None Image in the data path!"

        all_features_array = np.zeros((1, 128))

        # SIFT find descriptor
        for img_file in self.img_files:

            img = cv2.imread(img_file)

            #SIFT
            sift = cv2.SIFT()

            kp_img, des_img = sift.detectAndCompute(np.asarray(img), None)
            tmp_i = 0
            if des_img is not None:
                #print desImg.shape
                #print kpImg[0].pt
                #img3 = cv2.drawKeypoints(np.asarray(img),kpImg)

                #Normalize the descriptor
                for i in range(0, len(des_img)):
                    des_img[i] = des_img[i] / np.sum(des_img[i])
                self.img_descriptor.append(des_img)
                self.file_name_matrix.append(img_file)
                tmp_i += 1
                all_features_array = np.vstack((all_features_array, des_img))
                #allFeaturesAarray = desImg.copy()

        #print all_features_array


        # k-mean
        n_features = all_features_array.shape[0]
        print "Num of features:" + str(n_features)
        #nclusters = int(np.sqrt(n_features))
        #print nclusters
        print "Num of Clusters:" + str(CLUSTERS_NUM)
        n_clusters = CLUSTERS_NUM
        code_book, distortion = vq.kmeans(all_features_array,
                                          n_clusters,
                                          thresh=K_THRESH)
        self.code_book = code_book

        #Save codeBook into file
        with open(CODE_BOOK, 'wb') as f:
            dump(code_book, f, protocol=HIGHEST_PROTOCOL)


    def gen_img_word_list(self):
        """
        The input file is put to ./HDPData/test.dat
        """
        for (ii, img) in enumerate(self.img_descriptor):
            word_list = []
            for img_des in img:
                tmp_distance = 0
                found_word = 0
                for (i, code) in enumerate(self.code_book):
                    diff_mat = code - img_des
                    sq_diff_mat = diff_mat ** 2
                    distance = sq_diff_mat.sum()
                    if tmp_distance <= distance:
                        tmp_distance = distance
                        found_word = i

                word_list.append(found_word)

            self.img_word_list.append(word_list)

        # Generate HDP data

        f = open(self.test_data_path, 'w')
        for word_list in self.img_word_list:

            #print wordList
            y = np.bincount(word_list)
            ii = np.nonzero(y)[0]
            tmp_str = str(len(ii)) + ' '
            for i in range(0, len(ii)):
                tmp_str += str(ii[i]) + ':' + str(y[ii[i]]) + ' '
            print >> f, tmp_str
        f.close

        return self.img_word_list

    def run_HDP(self):
        # Read the training data.
        c_train_filename = self.test_data_path
        p = subprocess.Popen("./hdp/hdp --algorithm train --data " + c_train_filename +
                             " --directory " + OUTPUT_PATH +
                             " --max_iter " + ITERS_NUM +
                             " --save_lag -1 "
                             " --gamma_b " + GAMMA_B +
                             " --alpha_b " + ALPHA_B,
                             stdout=subprocess.PIPE, shell=True)
        p.wait()
        print p.communicate()[0]

    def print_topic(self):

        topic_doc_file = OUTPUT_PATH + 'mode-word-assignments.dat'

        doc_topics = {}

        topics = {}
        doc_id = 0
        line_id = 0

        fread = open(topic_doc_file, 'r')

        for line in fread:
            if line_id > 0:
                data = line.strip().split()

                new_doc_id = data[0]

                if (doc_id != 0) and (doc_id != new_doc_id):
                    doc_topics[doc_id] = topics
                    topics = {}

                doc_id = new_doc_id
                topic_id = int(data[2])

                if topic_id not in topics:
                    topics[topic_id] = 0
                topics[topic_id] += 1


            line_id += 1

        fread.close()
        print doc_topics


if __name__ == "__main__":
# Pic folder
    pic = imageHDP('./thumbnails/')

    # Generate codebook
    codeBook = pic.gen_code_book()

    # Generate Wordlist
    ImgWordList = pic.gen_img_word_list()

    # Run HDP
    pic.run_HDP()

    # Print Topic
    pic.print_topic()

