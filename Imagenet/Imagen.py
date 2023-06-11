from idDict import idDictionary
class Imagen:
    def __init__(self, imageName, imageData, imageSize, imageId, attackName='', epsilon=''):
        self.name = imageName
        self.data = imageData
        self.size = imageSize #Indica el tamaño de la imagen (Ej:(224, 224))
        self.id = imageId #Indica el ID (n--------) de la clase real del objeto
        self.idName = idDictionary[imageId] #Indica el nombre de la clase real del objeto
        self.predictionId = ''
        self.predictionName = ''
        self.attackName = attackName #Indica el metodo de ataque con el cual se ha generado la imagen
        self.epsilon = epsilon #Indica el epsilon del metodo de ataque con el cual se ha generado la imagen

    def addPrediction(self, predID):
        self.predictionId = predID
        self.predictionName = idDictionary[predID]

    def modifyData(self, imageData):
        self.data = imageData

    def addAdversarialData(self, attackName, epsilon):
        self.attackName = attackName
        self.epsilon = epsilon

    def copyImage(self):
        img = Imagen(self.name, self.data, self.size, self.id)
        return img