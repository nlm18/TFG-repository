from idDict import idDictionary
class Imagen:
    def __init__(self, imageName, imageData, imageSize, imageId, networkName, attackName='', epsilon='', isAdvNatural=False):
        self.name = imageName
        self.data = imageData
        self.heatmap = 0
        self.size = imageSize #Indica el tamaño de la imagen (Ej:(224, 224))
        self.id = imageId #Indica el ID (n--------) de la clase real del objeto
        self.idName = idDictionary[imageId] #Indica el nombre de la clase real del objeto
        self.predictionId = ''
        self.predictionName = ''
        self.attackName = attackName #Indica el metodo de ataque con el cual se ha generado la imagen
        self.epsilon = epsilon #Indica el epsilon del metodo de ataque con el cual se ha generado la imagen
        self.advNatural = isAdvNatural #Indica si es un adversario natural
        self.networkModelName = networkName #Indica el nombre de la red neuronal con el que se hizo la prediccion de los datos

    def addPrediction(self, predID):
        self.predictionId = predID
        self.predictionName = idDictionary[predID]

    def modifyData(self, imageData):
        self.data = imageData

    def addAdversarialData(self, attackName, epsilon):
        self.attackName = attackName
        self.epsilon = epsilon

    def copyImage(self):
        img = Imagen(self.name, self.data, self.size, self.id, self.networkModelName)
        return img

    def addHeatmap(self, heatmapData):
        self.heatmap = heatmapData

    def addAdvNatural(self, isAdvNatural):
        self.advNatural = isAdvNatural

    def addNetworkModelName(self, networkName):
        self.networkModelName = networkName