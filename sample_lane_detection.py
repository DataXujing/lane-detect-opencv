'''
xujing
2020-06-20

车道线检测 opencv

1、CCD视频摄像机校准
2、读视频，转成按每一帧读取，图像预处理
3、图像灰度化
4、高斯平滑，减少边缘干扰
5、利用canny算子，进行边缘检测
6、设定感兴趣区域，减少运算量
7、利用hough变换，进行直线检测
8、将检测成功的直线和原图像融合


'''

import cv2
import numpy as np
import time


#Canny算子或Sobel算子进行边缘检测
def canny_func(blur_gray,canny_lthreshold=150,canny_hthreshold=250):
    canny_lthreshold = canny_lthreshold
    canny_hthreshold = canny_hthreshold
    edges = cv2.Canny(blur_gray,canny_lthreshold,canny_hthreshold)

    return edges


#设置ROI区域,定义一个和输入图像同样大小的全黑图像mask
def roi_mask(img,vertics):
    mask = np.zeros_like(img)
    #根据输入图像的通道数，忽略的像素点是多通道的白色，还是单通道的白色
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        mask_color = (255,)*channel_count
    else:
        mask_color = 255
    cv2.fillPoly(mask,[vertics],mask_color)
    masked_img = cv2.bitwise_and(img,mask)
    return masked_img

#Hough变换
# https://blog.csdn.net/yuyuntan/article/details/80141392

def hough_func(roi_image,rho=1,theta=np.pi/180,threshold=15,min_line_lenght=40,max_line_gap=20):
    rho = rho
    theta = theta
    threshold = threshold
    min_line_lenght = min_line_lenght
    max_line_gap = max_line_gap
    # line_img = hough_lines(roi_image,rho,theta,threshold,min_line_lenght,max_line_gap)
    line_img = cv2.HoughLinesP(roi_image,rho,theta,threshold,min_line_lenght,max_line_gap)

    return line_img


# def draw_lines(img,lines,color = [0,0,255],thickness = 2):
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img,(x1,y1),(x2,y2),color,thickness)


# line_image = np.copy(img) # 复制一份原图，将线段绘制在这幅图上
# draw_lines(line_image, lines, [255, 0, 0], 6)



# 计算左右车道线的直线方程
# 根据每个线段在图像坐标系下的斜率，判断线段为左车道线还是右车道线，
# 并存于不同的变量中。随后对所有左车道线上的点、所有右车道线上的点做一次最小二乘直线拟合，
# 得到的即为最终的左、右车道线的直线方程。
# 最小二乘拟合讲解可参考：https://blog.csdn.net/nienelong3319/article/details/80894621
# np.polyfit(X, Y, 1) #一次多项式拟合，相当于线性拟合

# 计算左右车道线的上下边界
# 考虑到现实世界中左右车道线一般都是平行的，所以可以认为左右车道线上最上和最下的点对应的y值，
# 就是左右车道线的边界。
def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    left_lines_x = []
    left_lines_y = []
    right_lines_x = []
    right_lines_y = []
    line_y_max = 0
    line_y_min = 999

    try:
        for line in lines:
            for x1,y1,x2,y2 in line:
                if y1 > line_y_max:
                    line_y_max = y1
                if y2 > line_y_max:
                    line_y_max = y2
                if y1 < line_y_min:
                    line_y_min = y1
                if y2 < line_y_min:
                    line_y_min = y2
                
                k = (y2 - y1)/(x2 - x1)
            
                if k < -0.3:
                    left_lines_x.append(x1)
                    left_lines_y.append(y1)
                    left_lines_x.append(x2)
                    left_lines_y.append(y2)
                elif k > 0.3:
                    right_lines_x.append(x1)
                    right_lines_y.append(y1)
                    right_lines_x.append(x2)
                    right_lines_y.append(y2)
        #最小二乘直线拟合
        left_line_k, left_line_b = np.polyfit(left_lines_x, left_lines_y, 1)
        right_line_k, right_line_b = np.polyfit(right_lines_x, right_lines_y, 1)
     
        #根据直线方程和最大、最小的y值反算对应的x
        cv2.line(img,
                 (int((line_y_max - left_line_b)/left_line_k), line_y_max),
                 (int((line_y_min - left_line_b)/left_line_k), line_y_min),
                 color, thickness)
        cv2.line(img,
                 (int((line_y_max - right_line_b)/right_line_k), line_y_max),
                 (int((line_y_min - right_line_b)/right_line_k), line_y_min),
                 color, thickness)
        # plot polygon
        zero_img = np.zeros((img.shape), dtype=np.uint8)
        polygon = np.array([
            [int((line_y_max - left_line_b)/left_line_k), line_y_max], 
            [int((line_y_max - right_line_b)/right_line_k), line_y_max], 
            [int((line_y_min - right_line_b)/right_line_k), line_y_min], 
            [int((line_y_min - left_line_b)/left_line_k), line_y_min]
            ])
        # 用1填充多边形
        cv2.fillConvexPoly(zero_img, polygon, color=(0, 255, 0))
        # zero_mask = cv2.rectangle(zero_img, (int((line_y_max - left_line_b)/left_line_k), line_y_max), 
        #     (int((line_y_min - right_line_b)/right_line_k), line_y_min),
        #     color=(0, 255, 0), thickness=-1)

        alpha = 1
        # beta 为第二张图片的透明度
        beta = 0.2
        gamma = 0
        # cv2.addWeighted 将原始图片与 mask 融合
        img = cv2.addWeighted(img, alpha, zero_img, beta, gamma)

    except Exception as e:
        print(str(e))
        print("NO detection")

    return img



#--------------------------------------------------------------------------

#  开始检测
video_file = "./test.mp4"
cap=cv2.VideoCapture(video_file)#通过VideoCapture函数对视频进行读取操作 打开视频文件
if cap.isOpened():
     print("正确打开视频文件")
else:
     print("没有正确打开视频文件")
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# save video 
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("./output.mp4", fourcc, fps, (width, height), 1)


while cap.isOpened():

    try:
        ret,img = cap.read()#读取每一帧图片 flag表示是否读取成功 frame是图片


        start = time.time()
        #取图灰度化
        grap = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blur_grap = cv2.GaussianBlur(grap,(3,3),0)
        canny_image = canny_func(blur_grap,canny_lthreshold=150,canny_hthreshold=250)

        #图像像素行数 rows = canny_image.shape[0]  720行
        #图像像素列数 cols = canny_image.shape[1]  1280列
        left_bottom = [0, canny_image.shape[0]]
        right_bottom = [canny_image.shape[1], canny_image.shape[0]]
        left_top = [canny_image.shape[1]/3,canny_image.shape[0]/1.5]
        right_top = [canny_image.shape[1]/3*2,canny_image.shape[0]/1.5]
        # apex = [canny_image.shape[1]/2, 290]
        # vertices = np.array([ left_bottom, right_bottom, apex ], np.int32)
        vertices = np.array([ left_top,right_top, right_bottom, left_bottom], np.int32)

        roi_image = roi_mask(canny_image, vertices)
        # roi_image = roi_mask(img, vertices)


        line_img = hough_func(roi_image,rho=1,theta=np.pi/180,threshold=15,min_line_lenght=40,max_line_gap=20)
        img = draw_lines(img,line_img)
        end = time.time()

        detect_fps = round(1.0/(end-start+0.00001),2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(img, 'Lane detect v1.0 | Xu Jing | FPS: {}'.format(detect_fps), 
            (40, 40), font, 0.7, (0,0,0), 2)

        writer.write(img)
    

        cv2.imshow('lane_detect', img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            break
    except Exception as e:
        print(str(e))
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

cap.release()
writer.release()
cv2.destroyAllWindows()

     
       

