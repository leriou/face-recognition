# -*- coding: UTF-8 -*-

import sys,os,dlib,glob,numpy
from skimage import io


class FaceRecongnition:

    def __init__(self):
        self.predictor_path = "1.dat"
        self.face_rec_model_path = "2.dat"
        self.faces_folder_path = "./candidate-faces"
        
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(self.predictor_path)
        self.facerec = dlib.face_recognition_model_v1(self.face_rec_model_path)
        self.descriptors = []
        self.dist = []

        self.candidate = ['feifei','feifei2','Shishi','gakki','shishi2','bingbing','bingbing2']
    
    def get_train_sets(self):
        for f in glob.glob(os.path.join(self.faces_folder_path, "*.jpeg")):
            print("Processing file: {}".format(f))
            self.img = io.imread(f)
            self.dets = self.detector(self.img, 1)
            print("Number of faces detected: {}".format(len(self.dets)))
            for k, d in enumerate(self.dets):  
                shape = self.sp(self.img, d)
                face_descriptor = self.facerec.compute_face_descriptor(self.img, shape)
                v = numpy.array(face_descriptor)  
                self.descriptors.append(v)

    def get_test_sets(self):
        self.img = io.imread(self.img_path)
        self.dets = self.detector(self.img, 1)
        for k, d in enumerate(self.dets):
            shape = self.sp(self.img, d)
            face_descriptor = self.facerec.compute_face_descriptor(self.img, shape)
            d_test = numpy.array(face_descriptor) 

            # 计算欧式距离
            for i in self.descriptors:
                dist_ = numpy.linalg.norm(i-d_test)
                self.dist.append(dist_)
    
    def recongnition(self):
        c_d = dict(zip(self.candidate,self.dist))
        cd_sorted = sorted(c_d.iteritems(), key=lambda d:d[1])
        print "\n The person is: ",cd_sorted[0][0]  
        dlib.hit_enter_to_continue()
    
    def run(self):
        self.img_path = sys.argv[1]
        self.get_train_sets()
        self.get_test_sets()
        self.recongnition()
        
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "请检查参数是否正确,需要3个参数 python app.py  xxx.jpg"
        exit()
    m = FaceRecongnition()
    m.run()
