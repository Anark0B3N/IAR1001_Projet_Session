import math

import cv2
import glob
import numpy as np

def main():
    test_knn()

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

def findKNearestNeigbours(listOfKnownImg, k, imgToTest):

    distByNumber = []
    for img in listOfKnownImg:
        dist = distBetweenImages(img[1], imgToTest)
        distByNumber.append([img[0], dist])

    distByNumber.sort(key=sortSecond)

    neighbourgsToReturn = []
    for x in range(k):
        neighbourgsToReturn.append(distByNumber[x])

    return neighbourgsToReturn


def sortSecond(val):
    return val[1]

def distBetweenImages(img1, img2):
    dimensionDistancesSum = 0
    for x in range(400):
        dimensionDistancesSum += math.pow((img1[x] - img2[x]) , 2)

    return math.sqrt(dimensionDistancesSum)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()


