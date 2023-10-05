import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import time
from pathlib import Path
import math
#import gi                       # for Jetson Nano
#gi.require_version('Gtk','2.0') # for Jetson Nano
import cv2
import os
import json 
import pandas as pd
import torch
from utils.sort import *
import torch.backends.cudnn as cudnn
from numpy import random
# import os
import speech_recognition as sr
from sklearn.metrics import mean_squared_error
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from moviepy.editor import VideoFileClip
import shutil
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import mediapipe as mp
global net,smile_cascade, person_scores
 
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)



class Face_smile_detect:
    def __init__(self,net,smile_cascade):
        self.net=net
        self.smile_cascade=smile_cascade
        
    def detect(self,image,smile=True):
        # fake_img=image.copy()
        try:
            frameHeight,frameWidth,_=image.shape
            frameOpencvDnn=cv2.resize(image,(300,300))
            
            conf_threshold=0.4
            
            blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
            
            self.net.setInput(blob)
            detections = self.net.forward()
            # bboxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * frameWidth)
                    y1 = int(detections[0, 0, i, 4] * frameHeight)
                    x2 = int(detections[0, 0, i, 5] * frameWidth)
                    y2 = int(detections[0, 0, i, 6] * frameHeight)
                    # print(x1,y1,x2,y2)
                    cv2.rectangle(image,(x1, y1), (x2, y2), (0,255,0), lineType=cv2.LINE_AA)
                    face=image[y1:y2, x1:x2]
                    
                    # fake_img[y1:y2, x1:x2]=face*0
                    if smile:
                        
                        face=cv2.resize(face,(0,0),fx=2,fy=2)
                        roi_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY )
                        # cv2.imshow('ss',roi_gray)
                        smiles = self.smile_cascade.detectMultiScale(roi_gray)
                        # print(smiles)
                        for (sx, sy, sw, sh) in smiles:
                            # print(sx, sy, sw, sh)
                            cv2.rectangle(image, (x1+int(sx/2), y1+int(sy/2)), (x1+(int(sx/2) + int(sw/2)), y1+(int(sy/2) + int(sh/2))), (255, 0, 0))
                            break
        except:
            pass
        return image
             
        

def mp4tomp3(mp4file,mp3file):
    videoclip=VideoFileClip(mp4file)
    audioclip=videoclip.audio
    audioclip.write_audiofile(mp3file)
    audioclip.close()
    videoclip.close()
    # return audioclip

def class_detect(chunks_data):
    
    classes={'Seatbelt':'Each chair has',
             'Safty_Instruction':'information please refer',
             'Emergency_Exits':'located in the',
             'Emergency_Exits2':'are located along',
             'Life_Vest':'is located',
             'OXYGEN_Mask':'located in a compartment'
             }
    real_class={}
    for chunk in list(chunks_data.keys()):
        text=chunks_data[chunk][1]
        for clas in list(classes.keys()):
            if classes[clas] in text:
                real_class[clas]=chunks_data[chunk][0][0]
                if clas=='Emergency_Exits':
                    for clas in list(classes.keys()):
                        if classes[clas] in text:
                            if clas=='Emergency_Exits':
                                continue
                            if clas=='Emergency_Exits2':
                                real_class[clas]=chunks_data[chunk][0][0]
    
    
    return real_class



def fps_counter(real_class, fps, total_frames):
    
    
    class_time={'Seatbelt': 21 ,
             'Safty_Instruction': 11,
             'Emergency_Exits': 23,
             'Emergency_Exits2': 21,
             'Life_Vest': 35,
             'OXYGEN_Mask':28
             }
    class_start_end_frame={}
    for clas in list(real_class.keys()):
        start=real_class[clas]
        frame_start=round(fps*start)
        frame_end=frame_start+(class_time[clas]*fps)
        frame_end=round(frame_end)
        if total_frames<frame_end:
            frame_end=total_frames
        class_start_end_frame[clas]=[frame_start,frame_end]
        
    return class_start_end_frame
        


def video_audio_to_text(video_files,folder_path):
    folder_name = os.path.join(folder_path,"audios")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    audio_path=os.path.join(folder_name,"audio.wav")
    mp4tomp3(video_files,audio_path)
    r = sr.Recognizer()
    
    sound = AudioSegment.from_wav(audio_path)  
    nonsilent_data = detect_nonsilent(sound, min_silence_len=1000, 
                                      silence_thresh=sound.dBFS-14,
                                      seek_step=1)
    chunk_time=[]
    for chunks in nonsilent_data:
        chunk_time.append([chunk/1000 for chunk in chunks])
    chunks = split_on_silence(sound,
        min_silence_len = 1000,
        silence_thresh = sound.dBFS-14,
        keep_silence=1000,
    )
    whole_text = []
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")        
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            try:
                text = r.recognize_google(audio_listened, language='en-CA')
            except sr.UnknownValueError:
                pass
            else:
                text = f"{text.capitalize()}. "
                whole_text.append(text)
    chunks_data={}
    for i in range(len(whole_text)):
        chunks_data['chunks_'+str(i)]=[chunk_time[i],whole_text[i]]
    shutil.rmtree(folder_name)
    return chunks_data

class PoseDetectors:

    def __init__(self,mode = False, upBody = True, smooth=True, detectionCon = 0.5, trackCon = 0.5):
        self.lmList= {}
        self.poses_id={0:'nose',        1:'left_eye_inner',
                  2:'lef_eye',          3:"left_eye_outer",
                  4:'right_eye_inner',  5:'right_eye',
                  6:"right_eye_outer",  7:"left_eear",
                  8:'right_ear',        9:'mouth_left',
                  10:'mouth_right',     11:'left_shoulder',
                  12:'right_shoulder',  13:'left_elbow',
                  14:'right_elbow',     15:'left_wrist',
                  16:'right_wrist',     17:'left_pinky',
                  18:'right_pinky',     19:'left_index',
                  20:'right_index',     21:'left_thumb',
                  22:'right_thumb',     23:'left_hip',
                  24:'right_hip',       25:'left_knee',
                  26:'right_knee',      27:'left_ankle',
                  28:'right_ankle',     29:'left_heel',
                  30:'right_heel',      31:'left_foot_index',
                  32:'right_foot_index'
            }
        self.angles_joints={'right_arm':        [24,12,14],
                            'left_arm':         [23,11,13],
                            'left_leg':         [23,25,27],
                            'right_leg':        [24,26,28],
                            'left_fore_arm':    [11,13,15],
                            'right_fore_arm':   [12,14,16],
                            'left_waist':       [11,23,25],
                            'right_Waist':      [12,24,26]
            }
        
        self.classes_frames={'Seatbelt':0,
                  'Safty_Instruction':0,
                  'Emergency_Exits':0,
                  'Emergency_Exits2':0,
                  'Life_Vest':0,
                  'OXYGEN_Mask':0,
                  }
        
        # self.frame_class=frame_class
        self.clas_data={}
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(False,self.mode, self.upBody, self.smooth, self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
    def angle2d(self,a,b,c):
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang
    
    def angle3d(self,a,b,c):
        
        a,b,c= np.array(a),np.array(b),np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return round(np.degrees(angle),2)
    
    
    def angles_find(self):
        all_angles={}
        for joints in list(self.angles_joints.keys()):
          if  all(self.poses_id[joint] in list(self.lmList.keys())  for joint in self.angles_joints[joints]):           
              # print(self.lmList)
              a=self.lmList[self.poses_id[self.angles_joints[joints][0]]]
              b=self.lmList[self.poses_id[self.angles_joints[joints][1]]]
              c=self.lmList[self.poses_id[self.angles_joints[joints][2]]]
              all_angles[joints]=self.angle3d(a,b,c)
    
    
        return all_angles
    
    
    def getPosition(self, img, draw=True):
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(lm)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * abs(h-w))
                # print(cx, cy, cz)
                self.lmList[self.poses_id[id]]=[cx,cy,cz]
        
        p_connections=self.mpPose.POSE_CONNECTIONS
        if draw:
            for con in p_connections:
                # print(con)
                if con[0]>10 and con[1]>10:
                    pt1=self.lmList[self.poses_id[con[0]]]
                    pt2=self.lmList[self.poses_id[con[1]]]
                    # print(pt1[:1])
                    cv2.line(img,tuple(pt1[:2]),tuple(pt2[:2]),(255,255,255),2)
                    
                    cv2.circle(img, (pt1[0], pt1[1]), 4, (255, 0, 0), cv2.FILLED)
                    cv2.circle(img, (pt2[0], pt2[1]), 4, (255, 0, 0), cv2.FILLED)
            

        all_angles=self.angles_find()
        
       
        return all_angles


class error_find:
    def __init__(self):
        self.classes_frames={'Seatbelt':0,
                  'Safty_Instruction':0,
                  'Emergency_Exits':0,
                  'Emergency_Exits2':0,
                  'Life_Vest':0,
                  'OXYGEN_Mask':0,
                  }

        self.clas_data={'Seatbelt':pd.read_csv(os.path.join('models','Seatbelt.csv')),
                  'Safty_Instruction':pd.read_csv(os.path.join('models','Safty_Instruction.csv')),
                  'Emergency_Exits':pd.read_csv(os.path.join('models','Emergency_Exits.csv')),
                  'Emergency_Exits2':pd.read_csv(os.path.join('models','Emergency_Exits2.csv')),
                  'Life_Vest':pd.read_csv(os.path.join('models','Life_Vest.csv')),
                  'OXYGEN_Mask':pd.read_csv(os.path.join('models','OXYGEN_Mask.csv')),
                  }
    def score_get(self,frame_class,all_angles):
        
        score_error=0
        try:
            if not frame_class ==None:
                for clas__ in list(self.clas_data.keys()):
                    if frame_class == clas__:
                        refrence_angles=self.clas_data[clas__].iloc[self.classes_frames[clas__]]
                        self.classes_frames[clas__]=int(self.classes_frames[clas__])+1
                        all_angle_values_list=list(all_angles.values())
                        if len(list(refrence_angles))!=len(all_angle_values_list):
                            mini=min([len(list(refrence_angles)),len(all_angle_values_list)])
                            refrence_angles=refrence_angles[:mini]
                            all_angle_values_list=all_angle_values_list[:mini]
    
                        score_error=round(mean_squared_error(np.array(list(refrence_angles))/100,np.array(list(all_angle_values_list))/100),2)
                        score_error=round(score_error/self.classes_frames[clas__],3)
        except:
            pass
        return score_error
    def img_scores_get(self,all_angles):
        
        bow={'right_arm': 45.45, 'left_arm': 15.96, 
                  'left_leg': 178.53, 'right_leg': 132.85, 
                  'left_fore_arm': 76.77, 'right_fore_arm': 111.15, 'waist': 130.275}
        
        greeting={'right_arm': 45.45, 'left_arm': 15.96, 
                  'left_leg': 178.53, 'right_leg': 132.85, 
                  'left_fore_arm': 76.77, 'right_fore_arm': 111.15, 'waist': 155.275}
        try:
            refrence_angles=list(bow.values())
            all_angle_values_list=list(all_angles.values())
            if len(list(refrence_angles))!=len(all_angle_values_list):
                mini=min([len(list(refrence_angles)),len(all_angle_values_list)])
                refrence_angles=refrence_angles[:mini]
                all_angle_values_list=all_angle_values_list[:mini]
    
            bow_error=round(mean_squared_error(np.array(list(refrence_angles))/100,np.array(list(all_angle_values_list))/100),2)
            
            refrence_angles=list(greeting.values())
            all_angle_values_list=list(all_angles.values())
            if len(list(refrence_angles))!=len(all_angle_values_list):
                mini=min([len(list(refrence_angles)),len(all_angle_values_list)])
                refrence_angles=refrence_angles[:mini]
                all_angle_values_list=all_angle_values_list[:mini]
    
            greeting_error=round(mean_squared_error(np.array(list(refrence_angles))/100,np.array(list(all_angle_values_list))/100),2)
            
            # print(greeting_error , bow_error)
            if greeting_error < bow_error:
                return 'greet', 100-greeting_error
            else:
                return 'bow', 100-bow_error
        except:
            return '', 0
        
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # print(x)
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    startX = c1[0]
    startY = c1[1]
    endX = c2[0]
    endY = c2[1]
    # print("X  ",startX,endX)
    # print("Y  ",startY,endY)
    crop_img = img[startY:endY, startX:endX]
    face_smile=Face_smile_detect(net,smile_cascade)
    crop_img=face_smile.detect(crop_img,smile=True)
    detector=PoseDetectors()
    all_angles=None
    
    try:
        p_landmarks, p_connections = detector.findPose(crop_img, False)
        # mp.solutions.drawing_utils.draw_landmarks(crop_img, p_landmarks, p_connections)
        all_angles=detector.getPosition(crop_img)
        waist_angle=(all_angles['right_Waist']+all_angles['left_waist'])/2
        all_angles.pop('right_Waist')
        all_angles.pop('left_waist')
        all_angles['waist']=waist_angle
        # print(all_angles)
        print('Found...')
    except:
        print('Nothing found...')
    # detector.getPosition(crop_img)
    
    img[startY:endY, startX:endX] = crop_img
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        # tf = max(tl - 1, 1)  # font thickness
        # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 9
        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
       
        
        listss=[1,2,3]
        tl=1
        for i, line in enumerate(reversed(label.split('\n'))):
            # print('tl',tl)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(line, 0, fontScale=tl / 3, thickness=tf)[0]
            # print(t_size)
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, line, (c1[0], c1[1] - listss[i]), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
            c1=(c1[0],c2[1])
        # cv2.putText(img, label, (c1[0], c1[1] - 6), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    return all_angles
    
def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    mot_tracker = Sort()
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # print('yeahi',save_dir)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
    
    if any(ext in source for ext in vid_formats):
        # print('ssss')
        chunks_data=video_audio_to_text(source,save_dir)
        # print('chunks_data',chunks_data)
        real_class=class_detect(chunks_data)
        # print('real_class',real_class)
        
        path, img, im0s, vid_cap, nframes =next(iter(dataset))
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        class_start_end_frame=fps_counter(real_class, fps, nframes)
        print('Class_fames',class_start_end_frame)
    # raise Exception("A error occured!")
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    frame=0

    refrence_angles=[90,90,180,180,180,180]
    wrong_action=0
    person_scores={}
    wrong_action_time=0
    error_score=error_find()
    
    
    # person_number =None
    
    # all_angles_videos=pd.read_csv('OXYGEN_Mask.csv')
    # all_angles_videos=pd.DataFrame()
    
    for path, img, im0s, vid_cap ,nframes in dataset:
        
        frame=frame+1
        
        # if frame<1400:
        #     continue
        # if frame>0+100:
        #     break

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()
        
        new_tensor=pred[0].clone().detach()
        # new_tensor = torch.tensor(pred[0], device = 'cpu')
        persons_123=[]

        for p in range(len(new_tensor)):
            if int(new_tensor[p][5])==0:
                persons_123.append(new_tensor[p])
        
        
        
        # # Apply Classifier
        # if classify:
        #     pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        # for i, det in enumerate(pred):  # detections per image
        if webcam:  # batch_size >= 1
            path=path[0]
            im0s=im0s[0]
            
        p, s, im0, frame = Path(path), '', im0s, getattr(dataset, 'frame', 0)
        
        if not len(persons_123)==0:    

            save_path = str(save_dir / p.name)
            tracked_objects = mot_tracker.update(persons_123)
            tracked_objects[:, :4] = scale_coords(img.shape[2:], tracked_objects[:, :4], im0.shape)#.round()
            # print(tracked_objects)
            # people_coords = []
            conf_number=0
            for x1, y1, x2, y2, obj_id, cls_pred in reversed(tracked_objects):
                # conf=persons_123[conf_number][4]
                conf_number=conf_number+1
                xyxy=(x1,y1,x2,y2)
                # c1=(int(x1),int(y1))
                # c2=(int(x2),int(y2))
                # line_thickness=2
                # tl =  line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
                color = [random.randint(0, 255) for _ in range(3)]
                label = f'{names[int(cls_pred)]}'
                label='person'
                # if label=="person":
                color=[90,85,68]
                # label = label+'_'+str(int(obj_id))+str(float(conf*100))
                label = label+'_'+str(int(obj_id))
                # print(label)
                if label:
                    frame_class=None
                    if any(ext in source for ext in vid_formats):
                        for clas_ in list(class_start_end_frame.keys()):
                            times=class_start_end_frame[clas_]
                            # total_frames=times[0],times[1]
                            if frame in range(times[0],times[1]+1):
                                frame_class=clas_
                    if label in list(person_scores.keys()):
                        try:
                            label= label+'\n'+'Angle_score: '+str(person_scores[label][frame_class][0])+'\n'+'Time score: '+str(person_scores[label][frame_class][1])
                        except:
                            pass
                    # print(label)
                    # print('aaa',frame_class)
                    # print(class_start_end_frame)
                    all_angles=plot_one_box(xyxy, im0, label=label, color=color, line_thickness=2)
                    
                    # if frame_class=='OXYGEN_Mask':
                    #     all_angles=plot_one_box(xyxy, im0, label=label, color=color, line_thickness=2) 
                    #     if person_number == None:
                    #         person_number=obj_id
                    #     if obj_id == person_number:
                    #         all_angles_videos=all_angles_videos.append(all_angles,ignore_index = True)
                   
                    
                    if not  any(ext in source for ext in vid_formats):
                        class_detected, score_error=error_score.img_scores_get(all_angles)
                        if not int(obj_id) in list(person_scores.keys()):
                            person_scores[label]={class_detected:[score_error]}
                    # print(person_scores)
                    if any(ext in source for ext in vid_formats):
                        score_error=error_score.score_get(frame_class,all_angles)
                        if not int(obj_id) in list(person_scores.keys()):
                            person_scores[label]={'Seatbelt':[100,100],
                                          'Safty_Instruction ':[100,100],
                                          'Emergency_Exits':[100,100],
                                          'Emergency_Exits2':[100,100],
                                          'Life_Vest':[100,100],
                                          'OXYGEN_Mask':[100,100],
                                          }
                            
                        total_score_error=person_scores[label]
                        total_score_error[frame_class][0]=total_score_error[frame_class][0]-score_error
                        if frame_class == None:
                            score_errorss=round(mean_squared_error(np.array(list(refrence_angles))/100,np.array(list(all_angles.values()))/100),2)
                            if score_errorss>20:
                                person_scores[label][1]=person_scores[label][1]-(1/fps)
            
            # all_angles_videos.to_csv('OXYGEN_Mask.csv', index=False)   
            
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if False:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        sss=os.path.splitext(vid_path)
                        # vid_path=sss[0]+'.avi'
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(sss[0]+'.avi', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
                    
    # for persons_names in list(person_scores.keys()):
    #     persons=person_scores[persons_names]
    #     for countss in list(persons.keys()):
    #         if len(persons[countss])>1:
    #             if persons[countss][1] > 0:
    #                 persons[countss][1]=persons[countss][1]/fps
    #     person_scores[persons_names]=persons
    
    json_path=os.path.join(save_dir,'score.json')
    with open(json_path, "w") as outfile: 
        json.dump(person_scores, outfile) 
    
    print(person_scores)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='C:/AI_Projects/Pose_Estimation/Blurring/videos/My Video.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',default=False, help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs=1, type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='Model_Output', help='save results to project/name')
    parser.add_argument('--name', default='project', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
                

