import cv2

def main():
    cv2.namedWindow('show',0)
    cv2.resizeWindow('show',640,360)

    vc = cv2.VideoCapture(0) #웹캠 읽기    
    #vc = cv2.VideoCapture('./images/video2.mp4') #원하는 동영상을 읽기
    
    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) #동영상이 갖고 있는 정보를 vc의 get() 함수로 읽기
    print (vlen) # 웹캠은 video length 가 0 입니다.

    while True:
        ret, img = vc.read() #vc 객체에서 read() 함수로 img 읽기 
                             #ret 은 read() 함수에서 이미지가 반환되면 True, 반대의 경우 False를 받기
        if ret == False:
            break           

        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전     

        # preprocess
        #img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
        # detector
        #img_rgb_vga = cv2.resize(img_rgb, (640, 360))

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)

        cv2.imshow('show', img)
        key = cv2.waitKey(1)
        if key == 27:
            break        

if __name__ == '__main__':
    main()