import cv2
import numpy as np
import time
import copy
fgmask=[]
%matplotlib inline
import matplotlib.pyplot as plt

    def correlation(img1,img2):#<class 'numpy.ndarray'>
        cor=(((img1-img1.mean())/(img1.std(ddof=0)))*((img2-img2.mean())/(img2.std(ddof=0)))).mean() 
        return cor
    def drawCostTime(processTime,totalframenum,outframenum):
        plt.plot(processTime)
        plt.title("CostTime,totalframes:"+str(totalframenum)+"|saveframes:"+str(outframenum)+"|meantime:"+str(np.array(processTime).mean())+"|totaltime"+str(np.array(processTime).sum()))
        plt.show()
    def init(history):
        bs = cv2.createBackgroundSubtractorKNN(history=history,detectShadows=True)
        es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return bs,es

def video_pre_process(camera,history=10,threshold=0.99,saveVideo=True,saveFileName='camera_test.avi',
                      showVideo=True,showCostTime=True,isprint=True):
    outframes = []
    outframenum = 0
    processTime = []
    firstImg = []
    secondImg = []
    firstnum = 0
    secondnum = 0
    totalframenum = 0
    frames = 0
    output1=[]
    output2=[]
    output3=[]
    output4=[]
    output5=[]
    output6=[]
    count=0
    bs,es=init(history)
    size=0
    while True:
        totalframenum+=1
        grabbed, frame_lwpCV = camera.read()
        frame=copy.deepcopy(frame_lwpCV)
        if not grabbed:
            break
        fgmask = bs.apply(frame_lwpCV) # 背景分割器，该函数计算了前景掩码
        if frames<history:
            frames+=1
            firstImg=frame;
            firstnum=totalframenum
            continue
        
        # 二值化阈值处理，前景掩码含有前景的白色值以及阴影的灰色值，在阈值化图像中，将非纯白色（244~255）的所有像素都设为0，而不是255
        th1 = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
        th2=cv2.erode(th1,es,iterations=2)
        
        # 下面就跟基本运动检测中方法相同，识别目标，检测轮廓，在原始帧上绘制检测结果
        dilated = cv2.dilate(th2, es, iterations=2) # 形态学膨胀
        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w<h and cv2.contourArea(c) > 100:
                cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (255, 255, 0), 2)
                
        if showVideo:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            show=False
            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 该函数计算一幅图像中目标的轮廓
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                if w<h and cv2.contourArea(c) > 100:
                    show=True
                    cv2.rectangle(dilated, (x, y), (x + w, y + h), (255, 255, 0), 2)
            if not show:
                print("Remove: ",totalframenum)
            
            #cv2.imshow('Origin', frame)
            output1.append(frame)
            output2.append(th1)
            output3.append(frame_lwpCV)
            output4.append(fgmask)
            output5.append(th2)
            output6.append(dilated)
            
            if show:
                """
                output1.append(frame)
                output2.append(th1)
                output3.append(frame_lwpCV)
                output4.append(fgmask)
                output5.append(th2)
                output6.append(dilated)
                """
                secondImg=frame
                secondnum=totalframenum
                
                start=time.time()
                cor=correlation(firstImg,secondImg)
                processTime.append(time.time()-start)
                
                
                if cor<threshold:
                    outframenum+=1
                    if saveVideo:
                        outframes.append(frame)
                        
                    if isprint:
                        print(firstnum,secondnum,cor)
                    cv2.imshow('dilated', dilated)
                    cv2.imshow('detection', frame_lwpCV)
                    firstImg=frame
                else:
                    print("--Remove: ",totalframenum)
                firstnum=totalframenum
        else:
            secondImg=frame
            secondnum=totalframenum
            start=time.time()
            cor=correlation(firstImg,secondImg)
            processTime.append(time.time()-start)
            
            if cor<threshold:
                firstImg=frame
                outframenum+=1
            firstnum=totalframenum
            
    if saveVideo:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(saveFileName, fourcc,10.0, size)
        if isprint:
            print()
            print(totalframenum,outframenum) 
        for f in outframes:
            count+=1
            out.write(f)

        out.release()
        
    #if showCostTime:
     #   drawCostTime(processTime,totalframenum,outframenum)
        
    camera.release()  
    cv2.destroyAllWindows()

    return [output1,output2,output3,output4,output5,output6,size,processTime,totalframenum,outframenum]


#C:\Users\hisangke\Desktop\PP

import os
root="C:\\Users\\hisangke\\Desktop\\PP\\Olympic-sport"   #"""数据集绝对路径"""
totalInformation=[]
count=0
for file in os.listdir(root):
    if file=="desktop.ini":
        continue
    print("------------",count,"------------")
    count+=1
    dir_file_path = os.path.join(root,file)
    print(dir_file_path)
    camera = cv2.VideoCapture(dir_file_path) 
    r1,r2,r3,r4,r5,r6,size,processTime,totalframenum,outframenum=video_pre_process(camera,isprint=False)
    print("CostTime,totalframes:"+str(totalframenum)+"|saveframes:"+str(outframenum)+"|meantime:"+str(np.array(processTime).mean())+"|totaltime"+str(np.array(processTime).sum()),str(totalframenum-outframenum))
    information=[]
    information.append(dir_file_path)
    information.append(file)
    information.append(totalframenum)
    information.append(outframenum)
    information.append(np.array(processTime).mean())
    information.append(np.array(processTime).sum())
    information.append(totalframenum-outframenum)
    totalInformation.append(information)

f=open("Olympic-sporttotalInformation_0.99.csv","w")
f.write("NO.,Path,Filename,Totalframenum,Outframenum,Mean Time,Total Time,Removed\n")
nu=-1
for first in totalInformation:
    nu+=1
    temp=str(nu)+","
    for second in first:
        temp+=str(second)+","
    f.write(temp)
    f.write("\n")
    print(temp)
f.close()