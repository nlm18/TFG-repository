'''
Clase donde se agrupan los datos de las metricas de las 500 imagenes para cada tipo
'''
class MetricsData:
    def __init__(self, imageTypeID, networkName, imageId):
        self.imageType = imageTypeID #Indica si es original, adv natural o que tipo de adv artificial
        self.networkModelName = networkName #Indica el nombre de la red neuronal con el que se hizo la prediccion de los datos
        self.idName = imageId #Indica el nombre de la clase real del objeto
        self.MediaIntensidadPixeles = []
        self.MediaIntensidadPixelesNormalizada = []
        self.Mediana = []
        self.VarianzaPixeles = []
        self.DesviacionTipicaPixeles = []
        self.CentroideMax = []
        self.DistanciaCentroideMax = []
        self.CentroideMin = []
        self.DistanciaCentroideMin = []
        self.DifMedias = []
        self.NormaMascara = []
        self.NormaImagen = []
        self.MSE = []
        self.PSNR = []
        self.SSIM = []

    def addMetricsValue(self, metricsValue):
        self.MediaIntensidadPixeles.append(metricsValue[4])
        self.MediaIntensidadPixelesNormalizada.append(metricsValue[5])
        self.Mediana.append(metricsValue[6])
        self.VarianzaPixeles.append(metricsValue[7])
        self.DesviacionTipicaPixeles.append(metricsValue[8])
        self.CentroideMax.append(metricsValue[9])
        self.DistanciaCentroideMax.append(metricsValue[10])
        self.CentroideMin.append(metricsValue[11])
        self.DistanciaCentroideMin.append(metricsValue[12])
        self.DifMedias.append(metricsValue[13])
        self.NormaMascara.append(metricsValue[14])
        self.NormaImagen.append(metricsValue[15])
        self.MSE.append(metricsValue[16])
        self.PSNR.append(metricsValue[17])
        self.SSIM.append(metricsValue[18])
