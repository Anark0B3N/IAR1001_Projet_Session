import cv2
import glob
import numpy as np

def main():
    quit = False
    while(not quit):
        imageFound = False
        keepImage = True
        while(not imageFound):
            print('\nBonjour, voici les images disponnibles dans le répertoire de travail')
            imgs = findAllImagesNames()
            for image in imgs:
                print(image)
            imageToRead = input("Avec quelle image voulez-vous travailler?:")

            if imageToRead not in imgs:
                print ("L'image " + imageToRead + " n'existe pas, merci de réessayer")
            else:
                imageNameTable = []
                imageNameTable.append(imageToRead)
                img = returnActualImages(imageNameTable)[0]
                imageFound = True
        while(keepImage):
            print("Que voulez-vous faire?\n1 - Voir l'image sélectionnée\n2 - Afficher les zones d'intérêt de l'image"
                  "\n3 - Choisir une nouvelle image\n4 - quitter")
            choix = input("Entrez votre choix: ")
            imageTable = []
            imageTable.append(img)
            if choix == '1':
                displayImages(imageTable)
            elif choix == '2':
                contours = returnContours(imageTable)[0]
                imageContours = returnActualImages(imageNameTable)[0]
                cv2.drawContours(imageContours, contours, -1, (0,255,0), 3)
                contoursTable = []
                contoursTable.append(imageContours)
                displayImages(contoursTable)
                yesOrNo = False
                while not yesOrNo:
                    saveYesOrNo = input("Voulez-vous enregistrer l'image avec les contours? Y/N: ")
                    if saveYesOrNo == 'y' or saveYesOrNo == 'Y':
                        cv2.imwrite(input("S'il-vous-plait entrez le nom de l'image à "
                                          "enregistrer sans l'extension: ") + ".jpg", imageContours)
                        yesOrNo = True
                    elif saveYesOrNo == 'n' or saveYesOrNo == 'N':
                        yesOrNo = True
            elif choix == '3':
                keepImage = False
            elif choix == '4':
                keepImage = False
                quit = True
            else:
                print("Votre choix ne correspond à aucune option, veuillez recommencer\n")

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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
