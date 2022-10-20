import matplotlib.pyplot as plt
from ..general import dumpPickle
from ..general import CURRENT_PATH, WORKSPACE_DIR, RESOURCE_DIR, SEDML_IMG_DIR, SEDML_RSL_DIR
from ..colls.sedml import Sedmls
import opencor as oc
import os
from lxml import etree
from tellurium.utils import omex
import re
from tqdm import tqdm


class Sedmls(Sedmls):
    def __init__(self, *paths):
        super().__init__(*paths)

    def add(self, file, wksDir, wks, workingDir):
        url = wks + '/rawfile/HEAD/' + file[len(wksDir) + 1:]
        if url not in self.data:
            newId = 'sedmlId-'+str(len(self.data))
            self.data[url] = {'id': newId, 'sedml': file[len(
                wksDir) + 1:], 'workspace': wks, 'workingDir': workingDir}
            self.id2Url[newId] = url
        self.data[url]['status'] = self.statusC['validating']

    def validate(self):
        """
            The function if for validating SEDML files whether they can be run
            or not. This function CAN ONLY BE CALLED from OpenCor application
            utilising IPython interfaced.
        """
        print('Start validating %d sedml files' % len(self.data))
        for counter, v in enumerate(self.data.values(), 1):
            if counter % 10 == 0:
                print(counter, end=' ')
            if v['status'] in [self.statusC['validating'], self.statusC['invalid']]:
                path = os.path.join(CURRENT_PATH, WORKSPACE_DIR,
                                    v['workingDir'], v['sedml'])
                isValid, issues = self.__isValid(path)
                if not isValid:
                    v['status'] = self.statusC['invalid']
                else:
                    v['status'] = self.statusC['validating']
                for i in range(len(issues)):
                    if issues[i].startswith('Error: the imports could not be fully instantiated'):
                        issues[i] = re.sub(os.path.join(
                            CURRENT_PATH, WORKSPACE_DIR, v['workingDir'])+'/', '', issues[i])
                v['issues'] = issues
        self.dumpJson()
        print('Finish, sedml files validation')

    def __isValid(self, path):
        # check if omex file
        if path.endswith('omex'):
            defStat = False
            kss = omex.listContents(path)
            for ks in kss:
                for k in ks:
                    if isinstance(k, str):
                        defStat = True if k.endswith('sedml') else defStat
            if not defStat:
                return False, []
        # check all sedml and omex
        try:
            sim = oc.open_simulation(path)
            defStat = True if len(sim.issues()) == 0 and sim.valid() else False
            issues = sim.issues()
            oc.close_simulation(sim)
            return defStat, issues
        except ValueError as e:
            return False, [str(e)]

    def extract(self, sysCellmls, sysVars):
        self.sysVars = sysVars
        self.sysCellmls = sysCellmls
        print('Extracting %d sedml ...' % len(self.data))
        for k, v in tqdm(self.data.items()):
            if v['status'] == self.statusC['validating']:
                self.__getSedmlData(k, v)

        self.dumpJson()
        self.sysVars.dumpJson()
        self.sysCellmls.dumpJson()

    def __getSedmlData(self, k, v):
        path = os.path.join(CURRENT_PATH, WORKSPACE_DIR,
                            v['workingDir'], v['sedml'])
        try:
            # extract simulations, results
            sim = oc.open_simulation(path)
            # simulations
            simulations = sim.data().odeSolverProperties()
            try:
                sim.run()
                rst = sim.results().data_store().voi_and_variables()
                # create result name map, because variable name can be shorter
                resKeyMap = {
                    '/'.join(key.rsplit('/', 2)[-2:]): key for key in rst.keys()}
            except RuntimeError as e:
                print(e, v['id'])
                v['issues'] = [str(e)]
                v['status'] = self.statusC['invalid']
                # don't continue if simulation run failed
                oc.close_simulation(sim)
                return

            # extract from sedml file
            parser = etree.XMLParser(recover=True)
            results = {}

            # Load sedml or omex
            if path.endswith('sedml'):  # sedml file
                root = etree.parse(path, parser).getroot()
            elif path.endswith('omex'):  # omex file
                omex.listContents(path)
                lsts = omex.listContents(path)
                for lst in lsts:
                    for l in lst:
                        if not isinstance(l, str):
                            continue
                        if '<sedml' in l.lower():
                            root = etree.fromstring(l.encode('utf-8'), parser)
                            break
            # Exit if root is not available
            if 'root' not in locals():
                oc.close_simulation(sim)
                return

            # initialisation
            simulations, models, tasks, variables, outputs = {}, {}, {}, {}, {}
            # get simulations, models, variables, and outputs
            for child in root:
                # extract simulations
                if child.tag.endswith('listOfSimulations'):
                    pass

                # extract models
                elif child.tag.endswith('listOfModels'):
                    for model in child:
                        if model.tag.endswith('model'):
                            modelPath = os.path.relpath(os.path.join(
                                os.path.dirname(k), model.attrib['source']))
                            cellmlId = self.sysCellmls.getId(url=modelPath)
                            # the case that the cellml file extension is using xml rather than cellml, then it will not found
                            if cellmlId is None:
                                v['status'] = self.statusC['invalid']
                                return
                            models[model.attrib['id']] = cellmlId
                            cellml = self.sysCellmls.getObjData(id=cellmlId)
                            self.sysCellmls.addSedml(cellmlId, v['id'])
                elif child.tag.endswith('listOfTasks'):
                    pass

                # extract variables from data generators
                elif child.tag.endswith('listOfDataGenerators'):
                    for dataGenerator in child:
                        if dataGenerator.tag.endswith('dataGenerator'):
                            id = dataGenerator.attrib['id']
                            for listOfVariables in dataGenerator:
                                if listOfVariables.tag.endswith('listOfVariables'):
                                    for variable in listOfVariables:
                                        var = variable.attrib['target']
                                        var = var[var.find(
                                            '@name=') + 7:].replace("']/cellml:variable[@name='", '/')[:-2]
                                        compName, varName = var.split('/')
                                        if var in resKeyMap:
                                            varId = self.sysCellmls.getVarId(
                                                cellml, compName, varName)
                                            variables[id] = resKeyMap[var]
                                            results[varId] = rst[resKeyMap[var]].values()

                # extract outputs (curves)
                elif child.tag.endswith('listOfOutputs'):
                    for plot in child:
                        plotType = plot.attrib['id']
                        for listOfCurves in plot:
                            if listOfCurves.tag.endswith('listOfCurves'):
                                for curve in listOfCurves:
                                    if curve.attrib['xDataReference'] in variables and curve.attrib['yDataReference'] in variables:
                                        xVar, yVar = variables[curve.attrib['xDataReference']
                                                               ], variables[curve.attrib['yDataReference']]
                                        xVar = xVar if not xVar.endswith(
                                            '/prime') else xVar[:xVar.rfind('/prime')]
                                        yVar = yVar if not xVar.endswith(
                                            '/prime') else yVar[:xVar.rfind('/prime')]
                                        xCompName, xVarName = xVar.split(
                                            '/')[-2], xVar.split('/')[-1]
                                        yCompName, yVarName = yVar.split(
                                            '/')[-2], yVar.split('/')[-1]
                                        curve.attrib.pop('xDataReference')
                                        curve.attrib.pop('yDataReference')
                                        curve.attrib['x'] = self.sysCellmls.getVarId(
                                            cellml, xCompName, xVarName)
                                        curve.attrib['y'] = self.sysCellmls.getVarId(
                                            cellml, yCompName, yVarName)
                                        outputs[plotType] = [dict(
                                            curve.attrib)] if plotType not in outputs else outputs[plotType] + [dict(curve.attrib)]

            # save simulation result to file
            if len(results) > 0:
                dumpPickle(results, RESOURCE_DIR,
                           'sedmlResults', str(v['id']) + '.gz')

            # modify sedml dictionary
            initVars = {k: v[0] for k, v in results.items()}
            v['status'] = self.statusC['current']
            v['simulations'], v['models'], v['tasks'], v['variables'], v['outputs'] = simulations, models, tasks, initVars, outputs
            # generate plot images:
            self.__generatePlotImages(v['id'], results, outputs)
            oc.close_simulation(sim)
        except:
            pass

    def __generatePlotImages(self, sedmlId, results, outputs):
        for outId, output in outputs.items():
            legen = []
            plotId = sedmlId+'.'+outId
            for plot in output:
                plt.plot(results[plot['x']], results[plot['y']])
                legen += [self.sysVars.getName(plot['y'])]
                # set plot to variable
                self.sysVars.addPlot(plot['x'], plotId)
                self.sysVars.addPlot(plot['y'], plotId)
            plt.legend(legen, loc='upper left')
            imagePdf = os.path.join(
                CURRENT_PATH, RESOURCE_DIR, SEDML_IMG_DIR, plotId + '.pdf')
            imagePng = os.path.join(
                CURRENT_PATH, RESOURCE_DIR, SEDML_IMG_DIR, plotId + '.png')
            plt.savefig(imagePdf)
            plt.savefig(imagePng)
            plt.clf()
