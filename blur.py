import cv2


def main():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        #事前学習した結果ファイルを利用して正面の顔を認識

    while True:
        _,source = cap.read() #ビデオキャプチャ
     
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        #先程読み込んだ学習ないようを使って顔検出

        for (x, y, w, h) in faces: #facesの座標を利用して動画にモザイクをかける
            roi_face = source[y:y+h, x:x+w] #顔の領域　roi(Region Of Interest)
            blur = cv2.blur(roi_face,(10,10))

            source[y:y+h, x:x+w] = blur

        cv2.imshow('result', source)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
    