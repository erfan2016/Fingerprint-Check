import cv2
import glob

class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name

finger = cv2.imread("120.jpg", 0)

bf = cv2.BFMatcher()
ratio = 0.6
bestMatches = []

finalImg = None
final_kp = None
finalPath = None

sift = cv2.SIFT_create()
(s_kp, s_des) = sift.detectAndCompute(finger, None)
for i, imagePath in enumerate(glob.glob("database" + "/*.jpg")):
    img = cv2.imread(imagePath, 0)
    matched = []
    (kp, des) = sift.detectAndCompute(img, None)
    matches = bf.knnMatch(s_des, des, k = 2)

    for m, n in matches:
        if m.distance < ratio * n.distance:matched.append([m])
    if(len(matched) > len(bestMatches)):
        bestMatches = matched
        finalImg = img.copy()
        finalPath = imagePath
        number = i
        final_kp = kp
    
if len(bestMatches) >= 10:
    print("Maximum points for best matcheing is ", len(bestMatches))
    print("Fingerprint number in data set is:", number)
    result = cv2.drawMatchesKnn(finger, s_kp, finalImg, final_kp, bestMatches,
                                None, matchColor=(0, 255, 0), matchesMask=None,
                              singlePointColor=(255, 0, 0), flags=0)
    get_name = MyImage(finalPath)
    cv2.imshow(str(get_name), result)
else:
    print("Fingerprint not find")
cv2.waitKey(0)
cv2.destroyAllWindows()
