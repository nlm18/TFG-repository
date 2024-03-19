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
        self.NormaMascara = []
        self.NormaImagen = []
        self.DifMedias = []
        self.DifNormaMascara = []
        self.DifNormaImagen = []
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
        self.NormaMascara.append(metricsValue[13])
        self.NormaImagen.append(metricsValue[14])
        self.DifMedias.append(metricsValue[15])
        self.DifNormaMascara.append(metricsValue[16])
        self.DifNormaImagen.append(metricsValue[17])
        self.MSE.append(metricsValue[18])
        self.PSNR.append(metricsValue[19])
        self.SSIM.append(metricsValue[20])
