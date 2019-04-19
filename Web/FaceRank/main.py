import cv2
import FaceRank.predict as predict
import FaceRank.face_detection as fd


if __name__ == '__main__':
    img = cv2.imread("girls.jpg")
    img_drawed = fd.draw_faces(img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    faces, coordinates = fd.get_face_image(img)
    print(faces)
    for i in range(len(faces)):
        score = predict.predict_cv_img(faces[i])
        AQ = str(predict.get_AQ(score[0][0]))
        print("AQ: ",AQ)
        cv2.putText(img_drawed, AQ, coordinates[i], font, 0.8, (255, 0, 0), 2)
    fd.show(img_drawed)
