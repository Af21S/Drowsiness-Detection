from keras.models import load_model # type: ignore
import numpy as np
import cv2

model=load_model('bestModel.keras')

cap=cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
        retur,frame=cap.read()
        if retur:
            frame2=frame
            eyes=eye_cascade.detectMultiScale(frame,scaleFactor=1.3,minNeighbors=5)
            if len(eyes):
                (x,y,w,h)=eyes[0]
            else:
                # print("no eyes")
                continue
            
            try:
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                frame=frame.astype(np.uint8)
                frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                aoi=frame[y:y+h,x:x+h]
                aoi=cv2.resize(aoi,(64,64))
                reshaped=np.reshape(aoi,(1,64,64,1))
                reshaped=np.array(reshaped)
                reshaped=reshaped/255.0
                result = model.predict(reshaped)
                if result > 0.5:
                    k='Open'
                else:
                    k="Closed"
                cv2.putText(frame2,k,(70,70),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,0,0),2)
                cv2.imshow("Actual",frame2)
                cv2.imshow("test",aoi)
                # cv2.imshow("",frame)
                c=cv2.waitKey(1)
                if c==ord('q'):
                    cv2.destroyAllWindows()
                    break
            except Exception as e:
                # print(e)
                continue
        else:
            break
