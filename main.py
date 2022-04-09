import math

import cv2
import glob
import numpy as np
from mnist import MNIST

def main():



    # or
    # images, labels = mndata.load_testing()

    Hough()

    # quit = False
    # while(not quit):
    #     imageFound = False
    #     keepImage = True
    #     while(not imageFound):
    #         print('\nBonjour, voici les images disponnibles dans le répertoire de travail')
    #         imgs = findAllImagesNames()
    #         for image in imgs:
    #             print(image)
    #         imageToRead = input("Avec quelle image voulez-vous travailler?:")
    #
    #         if imageToRead not in imgs:
    #             print ("L'image " + imageToRead + " n'existe pas, merci de réessayer")
    #         else:
    #             imageNameTable = []
    #             imageNameTable.append(imageToRead)
    #             img = returnActualImages(imageNameTable)[0]
    #             imageFound = True
    #     while(keepImage):
    #         print("Que voulez-vous faire?\n1 - Voir l'image sélectionnée\n2 - Afficher les zones d'intérêt de l'image"
    #               "\n3 - Choisir une nouvelle image\n4 - quitter")
    #         choix = input("Entrez votre choix: ")
    #         imageTable = []
    #         imageTable.append(img)
    #         if choix == '1':
    #             displayImages(imageTable)
    #         elif choix == '2':
    #             contours = returnContours(imageTable)[0]
    #             imageContours = returnActualImages(imageNameTable)[0]
    #             cv2.drawContours(imageContours, contours, -1, (0,255,0), 3)
    #             contoursTable = []
    #             contoursTable.append(imageContours)
    #             displayImages(contoursTable)
    #             yesOrNo = False
    #             while not yesOrNo:
    #                 saveYesOrNo = input("Voulez-vous enregistrer l'image avec les contours? Y/N: ")
    #                 if saveYesOrNo == 'y' or saveYesOrNo == 'Y':
    #                     cv2.imwrite(input("S'il-vous-plait entrez le nom de l'image à "
    #                                       "enregistrer sans l'extension: ") + ".jpg", imageContours)
    #                     yesOrNo = True
    #                 elif saveYesOrNo == 'n' or saveYesOrNo == 'N':
    #                     yesOrNo = True
    #         elif choix == '3':
    #             keepImage = False
    #         elif choix == '4':
    #             keepImage = False
    #             quit = True
    #         else:
    #             print("Votre choix ne correspond à aucune option, veuillez recommencer\n")

#reads all images in working folder and return their names
def findAllImagesNames():
    imgs = []
    for item in glob.glob("*.jpg"):
        imgs.append(item)
    for item in glob.glob("*.jpeg"):
        imgs.append(item)
    for item in glob.glob("*.png"):
        imgs.append(item)
    return imgs

#Reads all images in param table and return them
#For now we only read one image at the time, but we can do more if needed
def returnActualImages(imgs):
    imagesToReturn = []
    for image in imgs:
        imagesToReturn.append(cv2.imread(image, -1))
    return imagesToReturn

#Display all images in param table
#For now we only display one image at the time, but we can do more if needed
def displayImages(imgs):
    for image in imgs:
        cv2.imshow('image', image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

#Returns the ROI of pictures in param table
#Return a table of tables as each ROI is on a sepratate image
def returnContours(imgs):
    contoursTableTable = []
    for image in imgs:
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imgray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) [-2:]
        contoursTableTable.append(contours)
    return contoursTableTable



def test_knn():
    listOfImages = []


    myimg = cv2.imread("digits.png", -1)
    imageFaitParTom = cv2.imread("test-nb-2.png", -1)
    imgray = cv2.cvtColor(imageFaitParTom, cv2.COLOR_BGR2GRAY)

    imgTest = np.array(imgray).flatten()
    cells = [np.hsplit(row, 100) for row in np.vsplit(myimg, 50)]

    # Make it into a Numpy array: its size will be (50,100,20,20)
    imgTable = np.array(cells)

    # 5000 images of different written numbers. An image is 20x20, so 400 pixels
    # 5000 labels to identify each written numbers.
    # each contains 10 values representing % chance of representing this value (0 to 9)

    train_img = np.zeros((5000, 400))
    train_labels = np.zeros(5000)

    test_img = np.zeros((5000, 400))
    test_labels = np.zeros(5000)

    pixelIdx = 0
    imgIdx = -1
    imgPerNumber = 0
    for idx, firstD in np.ndenumerate(imgTable):
        if pixelIdx == 0:
            imgIdx += 1

            # if the number is 3: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # if the number is 9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            train_labels[imgIdx] = math.floor(idx[0] / 5)

            test_labels[imgIdx] =math.floor(idx[0] / 5)
            imgPerNumber += 1



        train_img[imgIdx, pixelIdx] = imgTable[idx[0]][idx[1]][idx[2]][idx[3]]
        test_img[imgIdx, pixelIdx] = imgTable[idx[0]][idx[1]][idx[2]][idx[3]]

        pixelIdx += 1
        if pixelIdx > 399:
            pixelIdx = 0


    imgIdx = 0
    for img in train_img:
        features = []
        for pixels in img:
            features.append(pixels)

        listOfImages.append([train_labels[imgIdx], features])

        imgIdx += 1

    neighbourgs = findKNearestNeigbours(listOfImages, 27, imgTest)

    countByNumber = {}
    for node in neighbourgs:
        if countByNumber.__contains__(node[0]):
            countByNumber[node[0]] += 1
        else:
            countByNumber[node[0]] = 1

    maxCount = 0
    maxLabel = -1
    for count in countByNumber:
        if countByNumber[count] > maxCount:
            maxCount = countByNumber[count]
            maxLabel = count

    print(neighbourgs)
    print("image est un " + maxLabel.__str__())


def knn2():
    mndata = MNIST('samples')
    listOfImages = []

    #60 000 images of 784 pixels (28x28)
    #60 000 labels
    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    imgIdx = 0
    for img in images:
        listOfImages.append([labels[imgIdx], img])

        imgIdx += 1

    # imageFaitParTom = cv2.imread("test-nb-2.png", -1)
    # imgray = cv2.cvtColor(imageFaitParTom, cv2.COLOR_BGR2GRAY)
    # imgTest = np.array(imgray).flatten()


    testAccuraty = 0
    imgCount = 0
    for imgToTest in images_test:
        neighbourgs = findKNearestNeigbours(listOfImages, 77, imgToTest, 784)

        countByNumber = {}
        for node in neighbourgs:
            if countByNumber.__contains__(node[0]):
                countByNumber[node[0]] += 1
            else:
                countByNumber[node[0]] = 1

        maxCount = 0
        maxLabel = -1
        for count in countByNumber:
            if countByNumber[count] > maxCount:
                maxCount = countByNumber[count]
                maxLabel = count

        print(neighbourgs)
        print("image est un " + maxLabel.__str__())
        print(imgCount.__str__() + " sur 200")

        if imgCount > 2000:
            break

        if maxLabel == labels_test[imgCount]:
            testAccuraty += 1

        imgCount += 1

    print("accuracy over " + imgCount.__str__() + " testing images:")
    print((testAccuraty / imgCount) * 100)

def findKNearestNeigbours(listOfKnownImg, k, imgToTest, imgSize = 400):

    distByNumber = []
    for img in listOfKnownImg:
        dist = distBetweenImages(img[1], imgToTest, imgSize)
        distByNumber.append([img[0], dist])

    distByNumber.sort(key=sortSecond)

    neighbourgsToReturn = []
    for x in range(k):
        neighbourgsToReturn.append(distByNumber[x])

    return neighbourgsToReturn


def sortSecond(val):
    return val[1]

def distBetweenImages(img1, img2, imgSize = 400):
    dimensionDistancesSum = 0
    for x in range(imgSize):
        dimensionDistancesSum += math.pow((img1[x] - img2[x]) , 2)

    return math.sqrt(dimensionDistancesSum)


def Hough():
    listOfImages = []

    myimg = cv2.imread("digits.png", -1)
    imageFaitParTom = cv2.imread("test-nb-2.png", -1)
    imgray = cv2.cvtColor(imageFaitParTom, cv2.COLOR_BGR2GRAY)

    imgTest = np.array(imgray).flatten()
    cells = [np.hsplit(row, 100) for row in np.vsplit(myimg, 50)]

    # Make it into a Numpy array: its size will be (50,100,20,20)
    imgTable = np.array(cells)


    for img in imgTable[44][0:4]:
        strong_lines = []
        hough_circles = []

        img = (255 - img)
        img = cv2.resize(img, (150, 150))

        #circle...
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=20, minRadius=10, maxRadius=90)

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            # circle outline
            radius = i[2]
            hough_circles.append(HoughCircle(center, radius))


        #lines...
        edges = cv2.Canny(img, 100, 200)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 14)

        for r_teta in lines:
            rho, teta = r_teta[0]
            strong = True
            for houghLine in strong_lines:
                if abs(houghLine.rho - rho) < 20 and abs(teta - houghLine.theta) < 6:
                    strong = False
                    break

            if strong:
                strong_lines.append(HoughStrong(rho, teta))



        # draw lines and circles
        for circle in hough_circles:
            cv2.circle(img, circle.center, circle.r, (80, 0, 0), 2)

        for line in strong_lines:
            a = np.cos(line.theta)
            b = np.sin(line.theta)
            x0 = a * line.rho
            y0 = b * line.rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img, (x1, y1), (x2, y2), (80, 0, 0), 1)



        cv2.imshow('image', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

class HoughStrong:
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta

class HoughCircle:
    def __init__(self, center, r):
        self.r = r
        self.center = center


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


