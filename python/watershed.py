import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import cProfile
#기존 알고리즘보다 6배 느림 

################################################################################
def gradient(arr):#grad 크기 반환
    img_y=np.gradient(arr,axis=0)#y차원 기울기
    img_x=np.gradient(arr,axis=1)#x차원 기울기
    return np.sqrt(np.square(img_x)+np.square(img_y)).astype(np.uint8)#기울기크기(int)
################################################################################
img=cv2.imread('C:/my_source/tttt.jpg')
#img=cv2.imread('C:/my_source/Chvrches.png')
img=cv2.fastNlMeansDenoising(img,img,10,7,21)#denoising은 확실하게 성능을 개선
img_copy=img.copy()
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_grad=gradient(img_gray)
grad_max=np.max(img_grad)+1
#_,img_grad=cv2.cv2.threshold(img_grad,40,41,cv2.THRESH_TRUNC)
cv2.imshow('input',img)
cv2.imshow('grad',img_grad)
height=img_grad.shape[0]
width=img_grad.shape[1]
################################################################################폐곡선으로 이루어진 마킹을 만들고 싶음(contour?)
def nothing(x):#trackbar콜백함수
    pass

cv2.createTrackbar('thickness', 'input', 1, 20, nothing)

mouse_pressed=False
state=1
color=[]
r=np.random.randint(256)
g=np.random.randint(256)
b=np.random.randint(256)
color.append((b,g,r))
print(state,'번째 마커')
points=[]
line=[]
mask=np.zeros_like(img_gray)
def onMouse(event,x,y,flags,param):#마우스 콜백 함수
    global mouse_pressed,img,state,b,g,r,SortedList,tag,thickness,mask#글로벌 변수 선언
            
    if event==cv2.EVENT_LBUTTONDOWN:#점 그리기 
        mouse_pressed=True
        cv2.circle(img,(x,y),thickness,(b,g,r),-1)
        cv2.circle(mask,(x,y),thickness,255,-1)
        line.append((x,y))        
    
    if event==cv2.EVENT_MOUSEMOVE:#선 그리기
        if mouse_pressed:
            #cv2.circle(img,(x,y),1,(b,g,r),-1)
            #cv2.imshow('input',img)
            #tag[y,x]=state
            line.append((x,y))
            if(len(line))>=2:
                cv2.line(img,line[-1],line[-2],(b,g,r),thickness)
                cv2.line(mask,line[-1],line[-2],255,thickness)
            
    if event==cv2.EVENT_LBUTTONUP:
        mouse_pressed=False
    
    
    if event==cv2.EVENT_RBUTTONDOWN:
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for _,cnt in enumerate(contours):
            
            for j in range(len(cnt)):
                points.append((cnt[j][0][0],cnt[j][0][1],state))
                
            
        state=state+1
        mask=np.zeros_like(img_gray)
        r=np.random.randint(256)
        g=np.random.randint(256)
        b=np.random.randint(256)
        color.append((b,g,r))
        print(state,'번째 마커')
    cv2.imshow('input',img)    

################################################################################ 
#addleft_total=0
def addleft(S, marker, x, y, img_grad, tag):
#    global addleft_total
#    start=time.time()
    x=x-1
    if (0 <= x):  # 주변부 좌표가 원하는 위치
        if tag[y,x] == 0:#미정이면 확장    
            S[img_grad[y,x]].append([x,y, marker])
#    addleft_total=addleft_total+time.time()-start
    
#addright_total=0
def addright(S, marker, x, y, img_grad, tag):
    x=x+1
#    global addright_total
#    start=time.time()
    if (x < width):  # 주변부 좌표가 원하는 위치
        if tag[y,x] == 0:#미정이면 확장        
            S[img_grad[y,x]].append([x,y, marker])
#    addright_total=addright_total+time.time()-start
    
#adddown_total=0   
def adddown(S, marker, x, y, img_grad, tag):
#    global adddown_total
#    start=time.time()
    y=y-1
    if (0 <= y):  # 주변부 좌표가 원하는 위치
        if tag[y,x] == 0:#미정이면 확장
            S[img_grad[y,x]].append([x,y, marker])
#    adddown_total=addright_total+time.time()-start
    
#addup_total=0
def addup(S, marker, x, y, img_grad, tag):
#    global addup_total
#    start=time.time()
    y=y+1
    if (y <height):  # 주변부 좌표가 원하는 위치
        if tag[y,x] == 0:#미정이면 확장
            S[img_grad[y,x]].append([x,y, marker])
#    addup_total=addup_total+time.time()-start

################################################################################

tag = np.zeros(img_grad.shape, dtype= 'int')#img의 state
S = [list() for i in range(grad_max)]# 0 smallest, -1 largest=>dist 순으로 정렬

while True:#마킹
    cv2.setMouseCallback('input',onMouse)
    thickness=cv2.getTrackbarPos('thickness','input')#trackbar로 굵기 설정
    if thickness<1:#굵기가 0이면 오류 발생하므로 최소값으로 설정
        thickness=1
    key=cv2.waitKey(1)
    
    if key==27:#esc=>종료          
        cv2.destroyAllWindows()
        #plt.imshow(tag,'gray')
        #plt.show()
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for _,cnt in enumerate(contours):
            for j in range(len(cnt)):
                points.append((cnt[j][0][0],cnt[j][0][1],state))
        for i in range(len(points)):#마커를 S 채워넣기
            addleft(S,points[i][2],points[i][0],points[i][1],img_grad,tag)
            addright(S,points[i][2],points[i][0],points[i][1],img_grad,tag)
            addup(S,points[i][2],points[i][0],points[i][1],img_grad,tag)
            adddown(S,points[i][2],points[i][0],points[i][1],img_grad,tag)
        break


#checktime=0    
def checkEmptyList(List):
#    global checktime
#    start=time.time()
    if grad_max>0:
        IN=0
        for i in range(grad_max):
            if List[i]:
                break
            else:
                IN=IN+1
                if IN==grad_max:IN=-1
#    checktime=checktime+time.time()-start
    else: grad_max=-1
    return IN
        

################################################################################  
def watershed():
    while True:#S에 있는 주변부 확장
        i=checkEmptyList(S)
        
        if i==-1:break
        c = S[i].pop(0) #candidate형  
        x = c[0]
        y = c[1]
        marker = c[2]
        if tag[y,x] == 0: #미정 tag marker로 마킹
            tag[y,x] = marker
            addleft(S, marker, x, y, img_grad, tag)
            addright(S, marker, x, y, img_grad, tag)
            adddown(S, marker, x, y, img_grad, tag)
            addup(S, marker, x, y, img_grad, tag)



if __name__=="__main__":

    arr=[]
    start=time.time()

    #watershed()
    cProfile.run("watershed()")
#    watertime=time.time()-start
    #print('watershed time:',time.time()-start)
    tagImg = np.zeros_like(img)
    if len(color)>0:
        for i in range(state):
            tagImg[tag == i+1] = color[i]

    out=cv2.addWeighted(img_copy,0.5,tagImg,0.5,0)
    images=[img_copy[:,:,::-1],img_gray,img_grad,img[:,:,::-1],tagImg[:,:,::-1],out[:,:,::-1]]
    titles=['input','gray','grad','marking','watershed','output']
    print("time:", time.time()-start)
    print('ncalls: 호출 수, tottime:함수에서 소비된 총 시간,percall:tottime//ncalls')
    print('cumtime:함수와 서브 함수에서 소요된 누적 시간,percall:cumtime//프리미티브 호출 횟수')
    print('pixel 수:', width*height)
#    addtime=addleft_total+addright_total+adddown_total+addup_total
#    print('add time',addtime)
#    print('check time',checktime)
#    print('extend time',watertime-addtime-checktime)
    for i in range(len(images)):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray'),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

    
    plt.show()

