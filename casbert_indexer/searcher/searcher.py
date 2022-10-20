from .indexVariable import IndexVariable
from ..general import CURRENT_PATH, RESOURCE_DIR, RS_CLUSTERER, RS_UNIT, RS_MATH, RS_SEDML, RS_VARIABLE
from ..general import RS_COMPONENT, RS_WORKSPACE, RS_CELLML, RS_IMAGE, SEDML_IMG_DIR, IMG_EXT
from ..general import PMR_SERVER, WORKSPACE_DIR, ALG_BM25, loadJson
from ..colls.unit import Units
from ..colls.equation import Maths
from ..colls.variable import Variables
from ..colls.cellml import Cellmls
from ..colls.sedml import Sedmls
from ..colls.workspace import Workspaces
from ..colls.component import Components
from ..colls.image import Images
import os


class Searcher:
    def __init__(self, algorithm=ALG_BM25, idxVarFile='invIdxVar-obo-lemma-low'):
        """Initialise ...

        Parameters
        ----------
        idxVarFile : str
            The name of the variavle file index.
        idxCellmlFile : str
            The name of the cellml file index.
        idxSedmlFile : str
            The name of the sedml file index.
        idxWorkspaceFile : str
            The name of the workspace file index.
        idxWorkspaceFile : str
            The name of the workspace file index.
        ....

        Returns
        -------
        list
            a list of strings used that are the header columns
        """
        self.algorithm = algorithm

        self.idxVar = IndexVariable(idxVarFile)

        self.clusterer = loadJson(RESOURCE_DIR, RS_CLUSTERER)

        self.sysUnits = Units(RESOURCE_DIR, RS_UNIT)
        self.sysMaths = Maths(RESOURCE_DIR, RS_MATH)
        self.sysSedmls = Sedmls(RESOURCE_DIR, RS_SEDML)
        self.sysVars = Variables(self.sysMaths, RESOURCE_DIR, RS_VARIABLE)
        self.sysComps = Components(RESOURCE_DIR, RS_COMPONENT)
        self.sysWks = Workspaces(RESOURCE_DIR, RS_WORKSPACE)
        self.sysCellmls = Cellmls(RESOURCE_DIR, RS_CELLML)
        self.sysImages = Images(RESOURCE_DIR, RS_IMAGE)

    def searchSedmls(self, query, top=10, page=1):
        # classify the query vertically
        # queryTypes = self.__classify(query)

        # get result based on the classification results
        # temporarily just variable search :)
        # the sedml search temporarily is modified variable search

        return self.__searchSedmls(query, top, page)

        # return self.__getVariables(query, top)

    def searchVariables(self, query, top=10, page=1):
        # classify the query vertically
        # queryTypes = self.__classify(query)

        # get result based on the classification results
        # temporarily just variable search :)
        # the sedml search temporarily is modified variable search

        # return self.__searchSedmls(query, top, page)

        return self.__getVariables(query, top)

    def __getVariables(self, query, top, page):
        resultVars = self.idxVar.getResults(
            query, top, page, algorithm=self.algorithm)

        for varId in resultVars['candidates']:
            varData = {}
            # get main information
            varData['name'] = self.sysVars.getName(varId)
            varData['init'] = self.sysVars.getInit(varId)
            varData['type'] = self.sysVars.getType(varId)
            if varData['type'] == 'state':
                varData['rate'] = self.sysVars.getRate(varId)
            # get units
            varUnit = self.sysVars.getUnit(varId)
            varData['unit'] = {'name': self.sysUnits.getNames(
                varUnit)[0], 'text': self.sysUnits.getText(varUnit)}
            # get math
            varData['math'] = self.sysVars.getMaths(varId)
            # get maths' dependents
            varData['dependent'] = {}
            self.sysVars.getDependents(varId, varDep=varData['dependent'])
            # get leaves
            varData['rdfLeaves'] = self.sysVars.getObjLeaves(varId)
            # get sedmls and plot
            varData['plot'] = [os.path.join(
                CURRENT_PATH, RESOURCE_DIR, SEDML_IMG_DIR, plot+IMG_EXT) for plot in self.sysVars.getPlots(varId)]
            varData['sedml'] = [os.path.join(PMR_SERVER, self.sysSedmls.getUrl(
                plot.split('.')[0])) for plot in self.sysVars.getPlots(varId)]
            #get components
            varData['component'] = self.sysComps.getName(
                self.sysVars.getCompId(varId))
            varData['compLeaves'] = self.sysComps.getObjLeaves(
                self.sysVars.getCompId(varId))
            # get cellml, workspace, images
            cellmlId = self.sysComps.getCellml(self.sysVars.getCompId(varId))
            varData['cellmlUrl'] = PMR_SERVER + \
                self.sysCellmls.getUrl(id=cellmlId)
            varData['cellmlCaption'] = self.sysCellmls.getCaption(id=cellmlId)
            # varData['cellmlImages'] = [varData['cellmlUrl'][:varData['cellmlUrl'].rfind('/')+1]+self.sysImages.getPath(id) for id in self.sysCellmls.getImages(id=cellmlId)]
            varData['workspaceUrl'] = PMR_SERVER + \
                self.sysCellmls.getWorkspace(id=cellmlId)

            # get exposures
            exposures = self.sysWks.getExposures(
                url=self.sysCellmls.getWorkspace(id=cellmlId))
            varData['exposures'] = [PMR_SERVER +
                                    exposure for exposure in exposures.keys()]
            # get similar cellml using cluster
            similarCellmls = self.__getOtherCellms(cellmlId)
            varData['similarCellmls'] = [
                PMR_SERVER + url for url in similarCellmls]
            varData['cellmlImages'] = self.__getImagePaths(
                cellmlId, similarCellmls)

            resultVars['candidates'][varId] = varData

        return resultVars

    def __searchSedmls(self, query, top, page):
        """
            Temporarily, the use of top in this function and passing to __getResults
            is not relevant. searchSedml is based on searchVariables which in many cases
            variables do not related to sedml. So, after get the variable results, this
            searchSedml will check the availability. The strategy is get all variables,
            chek their sedml and then selectting based on top and page
        """
        def getVarDataForPlot(varId):
            varData = {}
            varData['name'] = self.sysVars.getName(varId)
            varData['init'] = self.sysVars.getInit(varId)
            varData['type'] = self.sysVars.getType(varId)
            if varData['type'] == 'state':
                varData['rate'] = self.sysVars.getRate(varId)
            varUnit = self.sysVars.getUnit(varId)
            varData['unit'] = {'name': self.sysUnits.getNames(
                varUnit)[0], 'text': self.sysUnits.getText(varUnit)}
            varData['math'] = self.sysVars.getMaths(varId)
            varData['dependent'] = {}
            self.sysVars.getDependents(varId, varDep=varData['dependent'])
            varData['rdfLeaves'] = self.sysVars.getObjLeaves(varId)
            return varData

        resultVars = self.idxVar.getResults(query, algorithm=self.algorithm)
        resultPlots = {}

        # identify plots as an indication that variable has sedml
        left = (page-1)*top if page > 0 else 0
        right = left + top
        curPlotPos = -1
        for varId in resultVars['candidates']:
            varData = {}
            plots = self.sysVars.getPlots(varId)
            for plot in plots:
                if plot not in resultPlots:
                    resultPlots[plot] = {}
        lenResultPlots = len(resultPlots)
        if len(resultPlots) <= left:
            resultPlots = {}
        elif len(resultPlots) < right:
            resultPlots = {plot: {}
                           for plot in list(resultPlots.keys())[left:]}
        else:
            resultPlots = {plot: {}
                           for plot in list(resultPlots.keys())[left:right]}

        # for each variable having sedml plot, identify additional data and store in a resultPlots
        for plot, value in resultPlots.items():
            variable = {}
            sedmlId, plotId = plot.split('.')
            varIdsPlot = self.sysSedmls.getVariables(sedmlId, plot=plotId)
            for varIdPlot in varIdsPlot:
                if varIdPlot not in variable:
                    variable[varIdPlot] = getVarDataForPlot(varIdPlot)
            value['variable'] = variable
            value['path'] = os.path.join(
                CURRENT_PATH, RESOURCE_DIR, SEDML_IMG_DIR, plot+IMG_EXT)
            value['url'] = PMR_SERVER+self.sysSedmls.getUrl(sedmlId)
            value['workspaceUrl'] = PMR_SERVER + \
                self.sysSedmls.getWorkspace(sedmlId)
            cellmlId = self.sysSedmls.getCellmlId(sedmlId)
            value['cellmlUrl'] = PMR_SERVER + \
                self.sysCellmls.getUrl(id=cellmlId)
            # get exposures
            exposures = self.sysWks.getExposures(
                url=self.sysCellmls.getWorkspace(id=cellmlId))
            value['exposures'] = [PMR_SERVER +
                                  exposure for exposure in exposures.keys()]
            # get similar cellml using cluster
            similarCellmls = self.__getOtherCellms(cellmlId)
            value['similarCellmls'] = [
                PMR_SERVER + url for url in similarCellmls]
            # get cellml images
            value['cellmlImages'] = self.__getImagePaths(
                cellmlId, similarCellmls)

        return {'candidates': resultPlots, 'length': lenResultPlots}

    def __getOtherCellms(self, cellmlId):
        url = self.sysCellmls.getUrl(id=cellmlId)
        clusterId = self.clusterer['url2Cluster'][url]
        if clusterId == '-1':
            return []
        return self.clusterer['cluster'][clusterId]

    def __getImagePaths(self, cellmlId, similarCellmls):
        cellmlImages = []
        for imageId in self.sysCellmls.getImages(id=cellmlId):
            if self.sysImages.isAvailable(imageId):
                cellmlImages += [os.path.join(os.path.dirname(
                    PMR_SERVER+self.sysCellmls.getUrl(id=cellmlId)), self.sysImages.getPath(imageId))]
        if len(cellmlImages) == 0 and len(self.sysCellmls.getImages(id=cellmlId)) > 0:
            for similarCellml in similarCellmls:
                cellmlId = self.sysCellmls.getId(url=similarCellml)
                for imageId in self.sysCellmls.getImages(id=cellmlId):
                    if self.sysImages.isAvailable(imageId):
                        cellmlImages += [os.path.join(os.path.dirname(
                            PMR_SERVER+similarCellml), self.sysImages.getPath(imageId))]
                if len(cellmlImages) > 0:
                    break
        return cellmlImages
