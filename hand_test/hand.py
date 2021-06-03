import cv2
import mediapipe as mp
import numpy as np
import time


# hand initialize 설정
def hand_init() :
    max_num_hands = 2
    gesture = {
        0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
        6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
    }
    #rps_gesture = {0:'rock', 5:'paper', 9:'scissors'}
    hand_gesture = {0:'stop', 5:'move'}

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=max_num_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Gesture recognition model
    file = np.genfromtxt('data/gesture_train.csv', delimiter=',')
    angle = file[:,:-1].astype(np.float32)
    label = file[:, -1].astype(np.float32)
    knn = cv2.ml.KNearest_create()   # OpenCV KNN model로 학습
    knn.train(angle, cv2.ml.ROW_SAMPLE, label)
        
    return knn, angle, mp_hands, mp_drawing, hands


# hand detection start
#def hand_detection(cap, x_min, y_min) :
def hand_detection() :
    time_chk=0
    cur_time = 0.0
    hand_status = False
    
    action = ''
    hand_gesture = {0:'stop', 5:'move', 10:'ok'}
    cap = cv2.VideoCapture(-1)
    #_, frame = cap.read()
    #H,W = frame.shape[:2]
    hand_x, hand_y, hand_w, hand_h =0,0,0,0

    knn, angle, mp_hands, mp_drawing, hands = hand_init()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            action_result = []

            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 3))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z]

                # Compute angles between joints
                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                v = v2 - v1 # [20,3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                angle = np.degrees(angle) # Convert radian to degree

                # Inference gesture
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                # Draw gesture result
                if idx in hand_gesture.keys():
                    # 손바닥 가장 아래 : 0번 landmark , 중지 가장 윗부분 : 12번 landmark
                    org = ( int(res.landmark[0].x * frame.shape[1]), int(res.landmark[0].y * frame.shape[0])
                        , int(res.landmark[12].x * frame.shape[1]), int(res.landmark[12].y * frame.shape[0])
                    )

                    cv2.putText(frame, text=hand_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    action_result.append({
                        'action': hand_gesture[idx],
                        'org': org
                    })

                mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)
                
                init_ =0
                # 2개의 손이 보엿을때
                if len(action_result) == 2:
                    print("동작함")
                    if not hand_status :
                        time_chk = time.time()
                        hand_status = True
                    cur_time = round(time.time() - time_chk,2)
                    print(f"cur_time : ", cur_time)
                    print(f"cur_time 진행중  ")
                    if action_result[0]['action'] == 'stop' : 
                        # cv2.putText  적용위치  [ text 시작점  ]
                        #init_ =0
                        if action_result[1]['action'] =='stop' :
                            action = 'stop'
                            #init_ =1
                        elif action_result[1]['action'] !='stop' :
                            # move와 stop을 주는 gesture를 동시에하는 경우
                            # 한쪽손은 보자기, 다른 손은 주먹을 쥐는 경우 올바르지 않은 action으로 인식
                            action =''
                            #cur_time =0  # 시간 다시 초기화
                            time_chk = time.time()
                            cur_time = round(time.time() - time_chk,2)
                        # another action 
                        #
                        #
                    elif action_result[0]['action'] == 'move' :
                        # cv2.putText  적용위치  [ text 시작점  ]
                        #init_ = 0
                        if action_result[1]['action'] == 'move' :
                            action ='move'
                            #init_ = 1
                        elif action_result[1]['action'] !='move' :
                            # move와 stop을 주는 gesture를 동시에하는 경우
                            # 한쪽손은 보자기, 다른 손은 주먹을 쥐는 경우 올바르지 않은 action으로 인식
                            action =''
                            #cur_time =0  # 시간 다시 초기화
                            time_chk = time.time()
                            cur_time = round(time.time() - time_chk,2)
                            
                        # another action
                        #
                        #
                
                    if action != '' and len(action_result) >0 :
                        print(f"action test {action}, {action_result[init_]['org']}")
                        #pass
                        # org[0] : 손바닥 가장 아래부분 x   , org[1] : 손바닥 가장 아래 부분 y
                        #  org[2] : 중지의 가장 윗부분 x , org[2] : 중지의 가장 윗부분 y
                        text_x,text_y = (action_result[init_]['org'][2] , action_result[init_]['org'][3] )
                        if  not  text_y - 50  < 0  : 
                            text_y -= 50
                        #cv2.putText(frame, text=f"action : {action}" ,org=( action_result[init_]['org'][0] ,  action_result[init_]['org'][1]  ) , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3) 
                        cv2.putText(frame, text=f"action : {action}" ,org=( text_x, text_y ) , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3) 
                        hand_x = min(action_result[0]['org'][0], action_result[1]['org'][0])
                        hand_y = min(action_result[0]['org'][1], action_result[1]['org'][1])
                        hand_w = max(action_result[0]['org'][0], action_result[1]['org'][0]) - hand_x
                        hand_h = max(action_result[0]['org'][1], action_result[1]['org'][1]) - hand_y
                        #print( f"  (hand_x, hand_y, hand_w, hand_h)  : ", (hand_x, hand_y, hand_w, hand_h) )
                        print( f"  action_result  : ", action_result )
                # 2개의 손이 아닌 값이 보엿을때
                # 모든 값 초기화
                #elif len(action_result) != 2:

                # else :
                #     print("두 손이 아님")
                #     #print(f"len(action_result) : {len(action_result)} ")
                #     print(f"action_result : {action_result} ")
                #     action =''
                #     cur_time =0  # 시간 다시 초기화
                #     time_chk = time.time()


        print(f"curtime : {cur_time}  , time_chk : {time_chk}")
        #cv2.rectangle(frame, (x_min, y_min), (hand_x + hand_w, hand_y + hand_h),(255, 0, 120), 2)
	#cv2.rectangle(frame, (hand_x, hand_y), (hand_x + hand_w, hand_y + hand_h),(255, 0, 120), 2)
        cv2.imshow('hand_gesture', frame)
        #if action == '' : 
        #    continue
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27  or cur_time > 2.0:
            #cv2.destroyAllWindows()
            #cap.release()
            break


    return action,  (hand_x, hand_y, hand_w, hand_h)

if __name__ == '__main__':
    hand_detection()
