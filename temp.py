from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import face_recognition_models
import numpy as np # مكتبة معالجة البيانات numpy
import argparse
import imutils
import time
import dlib
import cv2
 
 
def eye_aspect_ratio(eye):
    # إحداثيات علامة العين الرأسية (X ، Y)
    A = dist.euclidean(eye[1], eye[5])# احسب المسافة الإقليدية بين مجموعتين
    B = dist.euclidean(eye[2], eye[4])
    # احسب المسافة الإقليدية بين المستويات
    # إحداثيات علامة العين الأفقية (X ، Y)
    C = dist.euclidean(eye[0], eye[3])
    # حساب نسبة أبعاد العين
    ear = (A + B) / (2.0 * C)
    # أعد نسبة العرض إلى الارتفاع للعينين
    return ear
 
 
# تحديد ثابتين
# نسبة أبعاد العين
# عتبة الوميض
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3
# تهيئة عداد الإطار وإجمالي عدد الومضات
COUNTER = 0
TOTAL = 0
 
# قم بتهيئة جهاز كشف الوجه DLOG (HOG) ، ثم قم بإنشاء تنبؤ بارز للوجه
print("[INFO] loading facial landmark predictor...")
# الخطوة 1: استخدم dlib.get_frontal_face_detector () للحصول على كاشف موضع الوجه
detector = dlib.get_frontal_face_detector()
# الخطوة 2: استخدم dlib.shape_predictor للحصول على كاشف موضع ميزة الوجه
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
 
# الخطوة 3: الحصول على مؤشر علامات الوجه بالعين اليسرى واليمنى بشكل منفصل
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# الخطوة الرابعة: افتح كاميرا cv2 المحلية
cap = cv2.VideoCapture(0)

# إطارات حلقة من دفق الفيديو
while True:
    # الخطوة الخامسة: حلقة ، قراءة الصورة ، توسيع أبعاد الصورة ، وتدرج الرمادي
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=720)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # الخطوة 6: استخدم الكاشف (الرمادي ، 0) لاكتشاف موضع الوجه
    rects = detector(gray, 0)
    
    # الخطوة 7: قم بتكرار معلومات موضع الوجه واستخدم أداة التنبأ (الرمادي ، المستقيم) للحصول على معلومات موضع ميزة الوجه
    for rect in rects:
        shape = predictor(gray, rect)
        
        # الخطوة 8: تحويل معلومات ميزات الوجه إلى تنسيق الصفيف
        shape = face_utils.shape_to_np(shape)
        
        # الخطوة 9: استخرج إحداثيات العين اليسرى واليمنى
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # الخطوة 10: يقوم المُنشئ بحساب قيمة EAR للعين اليسرى واليمنى ، باستخدام متوسط ​​القيمة مثل EAR النهائي
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
 
        # الخطوة الحادية عشرة: استخدم cv2.convexHull للحصول على موضع بدن محدب ، استخدم drawContours لرسم موضع المخطط التفصيلي لعملية الرسم
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
 
        # الخطوة الثانية عشرة: إجراء عمليات الرسم ، وضع علامة على الوجه بإطار مستطيل
        left = rect.left()
        top = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)    
 
        '''
                         احسب متوسط ​​درجة العين اليسرى والعين اليمنى بشكل منفصل كدرجة نهائية. إذا كانت أقل من العتبة ، أضف 1 ؛ إذا كانت أقل من الحد الأدنى لثلاث مرات متتالية ، فهذا يعني أنه تم تنفيذ حدث وميض.
        '''
        # الخطوة الثالثة عشرة: حلقة ، إذا تم استيفاء الشرط ، فإن عدد الومضات +1
        if ear < EYE_AR_THRESH:# نسبة أبعاد العين: 0.2
            COUNTER += 1
           
        else:
            # إذا كانت أقل من العتبة لمدة 3 مرات متتالية ، فهذا يعني أنه تم تنفيذ نشاط وامض
            if COUNTER >= EYE_AR_CONSEC_FRAMES:# العتبة: 3
                TOTAL += 1
            # إعادة تعيين عداد إطار العين
            COUNTER = 0
            
        # الخطوة 14: تنفيذ عمليات الرسم ، 68 تحديد نقطة الميزة
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
        # الخطوة 15: تنفيذ عمليات الرسم ، واستخدام cv2.putText لعرض عدد الومضات
        cv2.putText(frame, "Faces: {}".format(len(rects)), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (150, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "COUNTER: {}".format(COUNTER), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (450, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

 
    print("نسبة العرض إلى الارتفاع في الوقت الفعلي للعيون: {: .2f}".format(ear))
    if TOTAL >= 50:
        cv2.putText(frame, "SLEEP!!!", (200, 200),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, "Press 'q': Quit", (20, 500),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (84, 255, 159), 2)
    # عرض نافذة عرض مع opencv
    cv2.imshow("Frame", frame)
    
    # if the `q` key was pressed, break from the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# الافراج عن الكاميرا الافراج عن الكاميرا
cap.release()
# do a bit of cleanup
cv2.destroyAllWindows()
