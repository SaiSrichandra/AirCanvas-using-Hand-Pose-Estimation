import math
import cv2 as cv
import mediapipe as mp
import numpy as np

def nothing(x):
	pass
	pass

def render_lines(x, y):
	global List,cList
	b,g,r = [0,0,0]
	List[-1].append([x, y])
	cList[-1].append([int(b),int(g),int(r)])

def clear(event, x, y, flags, params):
	global List,cList
	if event == cv.EVENT_FLAG_LBUTTON:
		List = [[]]
		cList = [[]]

List = [ [ ] ]
cList = [[]]
THRESHOLD = 35
CONTOUR_MASK_SIZE = 10

isDrawing = False
prev_point = [0,0]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv.VideoCapture(0)

cv.namedWindow('result')
cv.namedWindow('mask')
cv.setMouseCallback('result', clear)


with mp_hands.Hands (min_detection_confidence=0.8,min_tracking_confidence=0.5, max_num_hands = 1) as hands:
    while(cap.isOpened()):
        if cv.waitKey(1) & 0xFF == 27:
            break
        ret,frame = cap.read()
        frame = cv.cvtColor(cv.flip(frame, 1) , cv.COLOR_BGR2RGB)
        frame.flags.writeable = False
        
        results = hands.process(frame)
        
        frame.flags.writeable = True
        
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks :
            
            index_tip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            

            index_center = (int(index_tip.x* frame.shape[1]), int(index_tip.y* frame.shape[0]))
            thumb_center = (int(thumb.x* frame.shape[1]), int(thumb.y* frame.shape[0]))

           

            dist = math.sqrt(sum(pow(a-b,2) for a, b in zip(index_center, thumb_center)))

            rows,cols,chan = frame.shape

            temp = frame.copy()
            mask = np.zeros((rows,cols), np.uint8)
            mask[index_center[1]-CONTOUR_MASK_SIZE : index_center[1]+CONTOUR_MASK_SIZE, index_center[0]-CONTOUR_MASK_SIZE : index_center[0]+CONTOUR_MASK_SIZE] += 255
        
            res = cv.bitwise_and(temp, temp, mask=mask)
            cv.imshow('mask',res)
            res2gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
            median = cv.medianBlur(res2gray, 23)
            contours, hierarchy = cv.findContours(median, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            if (len(contours) > 0) and (contours is not None) :
                cnt = contours[0]
                # cv.drawContours(frame,cnt, -1, (128, 200, 100), 2 )
                (a, b), r = cv.minEnclosingCircle(cnt)
                index_center = (int(a), int(b))
                radius = int(r)
                

                if dist > THRESHOLD :
                    isDrawing = True
                    avg_pt =tuple(np.add(index_center, prev_point) / 2)
                    cv.circle(frame, index_center, radius, (0, 128 ,128), 4)
                    render_lines(index_center[0], index_center[1])
                    prev_point = index_center
                   

                else:
                    if isDrawing:
                        prev_point = [0,0]
                        List.append([])
                        cList.append([])
                    isDrawing = False

        else:
            if isDrawing:
                prev_point = [0,0]
                List.append([])
                cList.append([])
            isDrawing = False

        
        white_arr = np.zeros(frame.shape, np.uint8)
        white_arr += 255


        for i,j in zip(List,cList):
            if j !=[]:
                r,g,b = j[0]
                cv.polylines(frame,[np.array(i, dtype=np.int32)], False, (r,g,b), 4, cv.LINE_AA)
                
                cv.polylines(white_arr,[np.array(i, dtype=np.int32)], False, (r,g,b), 4, cv.LINE_AA)
        
        disp_arr = np.vstack((white_arr, frame))
        
        cv.imshow('result', disp_arr)
        
        

cv.destroyAllWindows()
cap.release()