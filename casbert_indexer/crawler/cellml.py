from ..general import getAllFilesInDir, loadPickle, dumpPickle, loadJson, dumpJson as dJson
from ..general import CURRENT_PATH, RESOURCE_DIR, WORKSPACE_DIR
from ..colls.cellml import Cellmls
import opencor as oc
import os
import rdflib
import xmltodict
from lxml import etree
from tellurium.utils import omex
import re
import urllib.parse as urilib
from tqdm import tqdm


class Cellmls(Cellmls):
    def __init__(self, sysWks, sysSedmls, sysImages, sysComps, sysVars, sysUnits, sysMaths, *paths):
        super().__init__(*paths)
        self.sysUnits = sysUnits
        self.sysComps = sysComps
        self.sysVars = sysVars
        self.sysMaths = sysMaths
        self.sysImages = sysImages
        self.sysSedmls = sysSedmls
        self.sysWks = sysWks

    def update(self):
        fileTypes = ['cellml', 'sedml', 'rdf', 'omex']
        # go to all workspaces
        for k, w in self.sysWks.getData().items():
            # only update files in workspaces with validating status
            if w['status'] == self.sysWks.getStatus()['validating']:
                w['status'] == self.sysWks.getStatus()['current']
                wksDir = os.path.join(
                    CURRENT_PATH, WORKSPACE_DIR, w['workingDir'])
                #exclude imported folder to be extracted (from subModels)
                excludeDirs = list(os.path.join(wksDir, p)
                                   for p in w['subModels'].keys())
                allFiles = [file for file in getAllFilesInDir(WORKSPACE_DIR, w['workingDir'])
                            if all(not file.startswith(exclude) for exclude in excludeDirs) and
                            any(file.endswith(ext) for ext in fileTypes)]
                for file in allFiles:
                    fileType = file[file.rfind('.') + 1:]
                    if fileType == 'cellml':
                        self.add(file, wksDir, k, w['workingDir'])
                    elif fileType in ['sedml', 'omex']:
                        self.sysSedmls.add(file, wksDir, k, w['workingDir'])
        self.dumpJson()
        self.sysWks.dumpJson()
        self.sysSedmls.dumpJson()

    def add(self, cellmlFile, wksDir, wks, workingDir):
        url = wks + '/rawfile/HEAD/' + cellmlFile[len(wksDir) + 1:]
        if url not in self.data:
            self.data[url] = {'id': self.getNewId(), 'cellml': cellmlFile[len(
                wksDir) + 1:], 'workspace': wks, 'workingDir': workingDir}
        self.data[url]['status'] = self.statusC['validating']
        self.id2Url[self.data[url]['id']] = url
        self.local2Url[workingDir+'/'+cellmlFile[len(wksDir) + 1:]] = url

    # START: functions to validate cellmls
    def validate(self):
        print('Start validating %d cellml files' % len(self.data))
        try:
            logPos = loadJson(RESOURCE_DIR, 'log_validation.json')['pos']
        except:
            logPos = 0
        print(logPos)
        for e, v in enumerate(tqdm(list(self.data.values())[logPos:])):
            if v['status'] in [self.statusC['validating'], self.statusC['invalid']]:
                path = os.path.join(CURRENT_PATH, WORKSPACE_DIR,
                                    v['workingDir'], v['cellml'])
                isValid, issues = self.__isValid(path)
                if isValid:
                    v['status'] = self.statusC['validating']
                else:
                    v['status'] = self.statusC['invalid']
                for i in range(len(issues)):
                    if issues[i].startswith('Error: the imports could not be fully instantiated'):
                        issues[i] = re.sub(os.path.join(
                            CURRENT_PATH, WORKSPACE_DIR, v['workingDir'])+'/', '', issues[i])
                v['issues'] = issues
            if (logPos+e+1) % 100 == 0:
                self.dumpJson()
                dJson({'pos': logPos+e+1}, RESOURCE_DIR, 'log_validation.json')
        self.dumpJson()
        print('Finish, cellml files validation')

    def __isValid(self, path):
        try:
            sim = oc.open_simulation(path)
            defStat = True if len(sim.issues()) == 0 and sim.valid() else False
            issues = sim.issues()
            oc.close_simulation(sim)
            return defStat, issues
        except ValueError as e:
            return False, [str(e)]
        except:
            return False, []
    # END: functions to validate cellmls

    def extract(self):
        self.rdfGraph = loadPickle(CURRENT_PATH, RESOURCE_DIR, 'rdf.graph')
        print('Extracting %d cellml ...' % len(self.data))

        for cellml in tqdm(self.data.values()):
            if cellml['status'] in [self.statusC['validating'], self.statusC['current']]:
                self.__extractFile(cellml)
                cellml['status'] = self.statusC['current']

        self.dumpJson()
        self.sysUnits.dumpJson()
        self.sysComps.dumpJson()
        self.sysMaths.dumpJson()
        self.sysVars.dumpJson()
        self.sysImages.dumpJson()
        dumpPickle(self.rdfGraph, CURRENT_PATH, RESOURCE_DIR, 'rdf.graph')
        print('# of variables: ', len(self.sysVars.getData()))

    def __extractFile(self, cellml):
        # need to initialise the cellml or reset all data, so there is no rubbish data
        self.__resetOrInitCellml(cellml)  # not yet handle
        # get path
        path = os.path.join(CURRENT_PATH, WORKSPACE_DIR,
                            cellml['workingDir'], cellml['cellml'])
        # print(path)
        # extract from simulation
        self.__getResults(path, cellml)
        # extract from cellml xml
        parser = etree.XMLParser(recover=True, remove_comments=True)
        root = etree.parse(path, parser).getroot()
        cellml['modelMeta'] = {}
        # find cmeta
        self.__setMetaFromElement(root, cellml, cellml['id'])
        # find data from imported cellml
        self.importedElements = self.__getImported(root, path, cellml)
        # extract units
        self.__getUnits(root, cellml)
        # extract components
        self.__getComponents(root, cellml)
        # extract rdfs
        self.__getAndIntegrateRdfData(root, cellml, path)
        # extract documentation
        self.__getDocumentation(root, cellml)

    def __resetOrInitCellml(self, cellml):
        pass

    def __getResults(self, path, cellml):
        sim = oc.open_simulation(path)
        try:
            sim.run()
        except:
            pass  # error in running still may have results

        # get variables and value using opencor
        results = sim.results()
        variables = results.dataStore().voi_and_variables()
        compsT2Id, modelVars = {}, []
        varKeys = list(variables.keys())
        varKeys.sort(key=lambda x: x.count('/'))

        # init type of variables voi
        algebraic, constants, states, rates, voi = [], [], [], [], []

        # iterate for each variable
        for variable in varKeys:
            compId = self.sysComps.add(cellml['id'], variable, compsT2Id)
            if compId != None:
                varId = self.sysVars.getNewId()
                self.sysComps.addVariable(compId, varId)
                modelVars += [varId]
                varValue = results.dataStore().voi_and_variables()[
                    variable].values()[0]
                if variable in results.algebraic().keys():
                    varType = 'algebraic'
                    algebraic += [varId]
                elif variable in results.constants().keys():
                    varType = 'constant'
                    constants += [varId]
                elif variable in results.states().keys():
                    varType = 'state'
                    states += [varId]
                elif variable in results.rates().keys():
                    varType = 'rate'
                    rates += [varId]
                else:
                    varType = 'voi'
                    voi += [varId]
                self.sysVars.add(varId, varValue, variable, compId, varType)

        oc.close_simulation(sim)
        cellml['components'] = list(compsT2Id.values())
        cellml['variables'] = modelVars
        cellml['algebraic'] = algebraic
        cellml['constants'] = constants
        cellml['states'] = states
        cellml['rates'] = rates
        cellml['voi'] = voi

    def __getImported(self, root, path, cellml):
        """
        Get the imported units and component element. The imported component
        element is incorrporated with rdf.
        The result is organised into dictionary:
            {'units':{nameU1:elementU1, ...},
             'component':{nameC1:elementC1, ...}
             'rdf':rdfGraph}
        Return imported components, units, and rdf from imported cellmls.
        Return format is dict, {compName, element}, {unitName, element},
        {rdfMeta, element}. Consider that rdfMeta maybe different with
        """
        types = ['component', 'units']

        def __getImportedElements(path, element, rdfGraph):
            """function to get imported elements and nested imported elements"""
            dir = os.path.dirname(path)
            parser = etree.XMLParser(recover=True, remove_comments=True)
            tree = etree.parse(path, parser)
            type = element.tag[element.tag.rfind('}')+1:]
            nodeRef = element.attrib[type+'_ref']
            nodes = tree.xpath(
                "//*[local-name() = '"+type+"'][@name='"+nodeRef+"']")

            # get rdf and put it on rdfGraph
            for rdfElement in tree.iter():
                if 'rdf' in rdfElement.nsmap:
                    ns = '{' + rdfElement.nsmap['rdf'] + '}'
                    if rdfElement.tag == ns + 'RDF':
                        try:
                            rdfGraph.parse(data=etree.tostring(rdfElement))
                        except:
                            pass

            if len(nodes) > 0:
                if type+'_ref' not in nodes[0].attrib:
                    return nodes[0]
                else:
                    parentNode = nodes[0].getparent()
                    link = parentNode.attrib['{' +
                                             parentNode.nsmap['xlink'] + '}'+'href']
                    importPath = os.path.join(dir, link)
                    return __getImportedElements(importPath, nodes[0], rdfGraph)

        # get all import statements
        dir = os.path.dirname(path)
        allImports = root.findall('{' + root.nsmap[None] + '}' + 'import')
        importGroup = {type: {} for type in types}
        rdfGraph = rdflib.Graph()
        for importEls in allImports:
            importPath = os.path.join(
                dir, importEls.attrib['{' + importEls.nsmap['xlink'] + '}'+'href'])
            for child in importEls:
                type = child.tag[child.tag.rfind('}')+1:]
                if type in types:
                    importedElements = __getImportedElements(
                        importPath, child, rdfGraph)
                    importGroup[type][child.attrib['name']] = importedElements
        importGroup['rdf'] = rdfGraph
        return importGroup

    def __getUnits(self, root, cellml):
        ns = '{' + root.nsmap[None] + '}'
        listUnits = root.findall(ns + 'units')
        # extract units from cellml
        unitsN2Id = self.sysUnits.add(listUnits)
        # extracted units from imported cellml
        for name, units in self.importedElements['units'].items():
            if units != None:
                units.attrib['name'] = name
        unitsN2Id = {**unitsN2Id, **
                     self.sysUnits.add(self.importedElements['units'].values())}
        cellml['units'] = unitsN2Id

    # START: a group of functions to extract rdf from cellml
    def __getAndIntegrateRdfData(self, root, cellml, path):
        rdfPath = 'file://' + urilib.quote(path)
        # merge all RDF tag in a cellml to self.rdfGraph
        rdfElements = root.xpath(".//*[local-name()='RDF']")
        for rdfElement in rdfElements:
            for desc in rdfElement.xpath(".//*[local-name()='Description'][@*[local-name()='about']]"):
                if any(x in ['rdf', 'RDF'] for x in desc.nsmap):
                    if 'rdf' in desc.nsmap:
                        att = '{'+desc.nsmap['rdf']+'}about'
                    elif 'RDF' in desc.nsmap:
                        att = '{'+desc.nsmap['RDF']+'}about'
                    if desc.attrib[att].startswith('#'):
                        desc.attrib[att] = rdfPath+desc.attrib[att]
            try:
                self.rdfGraph.parse(data=etree.tostring(rdfElement))
            except:
                pass

        # assign triples and leave to an object (cellml, component, variable)
        for meta, obj in cellml['modelMeta'].items():
            meta = rdfPath+meta
            triples, leave = self.__getTriplesOfMeta(rdflib.URIRef(meta))
            # check if there is no triples, probably there are triples in imported Graph
            if len(triples) == 0:
                triples, leave = self.__getTriplesOfMeta(rdflib.URIRef(meta))
            # store the rdf triples and leave
            if obj['type'] == 'model':
                url = self.getUrl(id=obj['id'])
                self.addRdf(url, triples, leave, meta)
            elif obj['type'] == 'component':
                self.sysComps.addRdf(obj['id'], triples, leave, meta)
            elif obj['type'] == 'variable':
                self.sysVars.addRdf(obj['id'], triples, leave, meta)

    def __getTriplesOfMeta(self, metaId):
        triples = list(self.rdfGraph.triples((metaId, None, None)))
        leaves = []
        if len(triples) > 0:
            for s, p, o in triples:
                result = self.__getTriplesOfMeta(o)
                triples += result[0]
                leaves += result[1]
            return list(set(triples)), list(set(leaves))
        else:
            return triples, [metaId]

    def __setMetaFromElement(self, element, cellml, objId):
        if 'cmeta' in element.nsmap:
            key = '{' + element.nsmap['cmeta'] + '}id'
            if key in element.attrib:
                cmeta = element.attrib[key] if element.attrib[key].startswith(
                    '#') else '#'+element.attrib[key]
                cellml['modelMeta'][cmeta] = {
                    'id': objId, 'type': element.tag[element.tag.find('}') + 1:]}
    # END: a group of functions to extract rdf from cellml

    # START: a group of functions for components extraction
    def __getComponents(self, root, cellml):
        ns = '{' + root.nsmap[None] + '}'
        for compId in cellml['components']:
            compName = self.sysComps.getName(compId)
            elements = root.xpath("//ns:component[@name='"+compName+"']", namespaces={'ns': root.nsmap[None]})
            if len(elements) == 0: continue
            element = elements[0]
            compVarState = {}
            for varId in self.sysComps.getVariables(compId):
                # get and set the units of variable
                self.sysVars.setUnit(varId, self.__getVariableUnit(cellml, varId, element))
                self.__setVarMathAndDependencies(cellml, varId, element)
                # get and set rdf metadata
                varName = self.sysVars.getName(varId, short=True)
                # metadata is just for variables algebraic, constants, and state
                if self.sysVars.getType(varId) != 'rate':
                    varElement = element.xpath("//ns:variable[@name='"+varName+"']", namespaces={'ns': root.nsmap[None]})[0]
                    self.__setMetaFromElement(varElement, cellml, varId)

                # to record rate into state
                if self.sysVars.getType(varId) == 'state':
                    if compId+varName not in compVarState:
                        compVarState[compId+varName] = {'id': varId}
                    else:
                        compVarState[compId+varName]['id'] = varId
                elif self.sysVars.getType(varId) == 'rate':
                    if compId+varName not in compVarState:
                        compVarState[compId +
                                     varName] = {'rate': self.sysVars.getInit(varId)}
                    else:
                        compVarState[compId +
                                     varName]['rate'] = self.sysVars.getInit(varId)
            # add rate information into variables
            for k, v in compVarState.items():
                try:
                    self.sysVars.setRate(v['id'], v['rate'])
                except:
                    pass

            self.sysComps.setCellml(compId, cellml['id'])
            self.__setMetaFromElement(element, cellml, compId)

    def __getVariableUnit(self, cellml, varId, element):
        varName = self.sysVars.getName(varId, short=True)
        units = element.find(
            './/ns:variable[@name="'+varName+'"]', namespaces={'ns': element.nsmap[None]})
        if units is None:
            return None
        unitsName = units.attrib['units']
        if unitsName in cellml['units']:
            unitId = cellml['units'][unitsName]
        else:
            unitId = self.sysUnits.addNewText(unitsName, unitsName)
            cellml['units'][unitsName] = unitId
        if(self.sysVars.getType(varId) == 'rate'):
            unitText = self.sysUnits.getText(
                unitId)+'units:second,exponent:-1.0;'
            unitsName += '_per_s'
            unitId = self.sysUnits.addNewText(unitsName, unitText)
            cellml['units'][unitsName] = unitId
        return unitId

    def __setVarMathAndDependencies(self, cellml, varId, element):
        compName = element.attrib['name']
        compId = self.sysVars.getCompId(varId)
        varName = self.sysVars.getName(varId, short=True)
        varType = self.sysVars.getType(varId)
        # if the varType is 'constant' then do not need to get maths
        # looking for mathml element
        maths = self.__getMathElement(varName, varType, element)
        # stop if the variable type is constant or voi
        if varType == 'constant':
            return
        # found at current component element
        if len(maths) > 0:
            for math in maths:
                mathText = re.sub(
                    r'\s\s+', ' ', etree.tostring(math).decode('utf-8'))
                self.sysVars.addMath(varId, mathText)
                # set up dependent variables
                rheVars = [right.text.strip()
                           for right in math[2].iter() if right.tag.endswith('ci')]
                compVarsT2Id = self.sysVars.getT2Id(
                    self.sysComps.getVariables(compId), short=True)
                for rheVar in rheVars:
                    if rheVar in compVarsT2Id:
                        self.sysVars.addDependents(
                            varId, compVarsT2Id[rheVar], rheVar)
                    else:
                        varDepId = self.__getVarDep(cellml, rheVar, element)
                        self.sysVars.addDependents(varId, varDepId, rheVar)
        # math is not found in current component element
        elif len(maths) == 0:
            isVarRef = self.__setVarRef(cellml, varName, varType, element)
            # possibly the variable in math is on the right hand
            if not isVarRef:
                maths = element.xpath(
                    ".//*[local-name() = 'apply'][*[local-name() = 'eq']]")
                compVarsT2Id = self.sysVars.getT2Id(
                    self.sysComps.getVariables(compId), short=True)
                for math in maths:
                    mathText = re.sub(
                        r'\s\s+', ' ', etree.tostring(math).decode('utf-8'))
                    allVars = [right.text.strip()
                               for right in math.iter() if right.tag.endswith('ci')]
                    if varName in allVars:
                        self.sysVars.addMath(varId, mathText)
                        allVars.remove(varName)
                        for var in allVars:
                            if var in compVarsT2Id:
                                self.sysVars.addDependents(
                                    varId, compVarsT2Id[var], var)
                            else:
                                varDepId = self.__getVarDep(
                                    cellml, var, element)
                                self.sysVars.addDependents(
                                    varId, varDepId, var)

        # looking for math for integration, i.e. 1 = A + B + C + D
        maths = element.xpath(
            ".//*[local-name() = 'math']/*[local-name() = 'apply'][*[local-name() = 'cn']]")
        for math in maths:
            mathText = re.sub(
                r'\s\s+', ' ', etree.tostring(math).decode('utf-8'))
            rheVars = [right.text.strip()
                       for right in math[2].iter() if right.tag.endswith('ci')]
            compVarsT2Id = self.sysVars.getT2Id(
                self.sysComps.getVariables(compId), short=True)
            if varName in rheVars:
                for rheVar in rheVars:
                    if rheVar in compVarsT2Id:
                        varRheId = self.getVarId(cellml, compName, rheVar)
                        self.sysVars.addMath(varRheId, mathText)
                    else:
                        varRheId = self.__getVarDep(cellml, rheVar, element)
                        self.sysVars.addMath(varRheId, mathText)

    def __getMathElement(self, varName, varType, element):
        # looking for mathml element
        if varType == 'algebraic':
            maths = element.xpath(
                ".//*[local-name() = 'apply'][*[local-name() = 'eq']][*[local-name() = 'ci'][normalize-space()='"+varName+"']]")
        elif varType in ['state', 'rate']:
            maths = element.xpath(
                ".//*[local-name() = 'apply'][*[local-name() = 'eq']][*[local-name() = 'apply'][*[local-name() = 'ci'][normalize-space()='"+varName+"']]]")
        else:
            return []
        return maths

    def __setVarRef(self, cellml, varName, varType, element):
        """
        Set variable reference. In a case that a variable (state or algebraic)
        in a component has its value but do not have its own mathematical
        declaration in its component element, so it can be traced to refered
        variable through connection element.
        Input: cellml result, variable name, variable type, component element
        Return: - True if variable reference or imported variable are found
                - False if variable reference or imported are found
        """
        compName = element.attrib['name']
        varId = self.getVarId(cellml, compName, varName)
        compId = self.sysVars.getCompId(varId)
        while True:
            varRef, elementRef, isImported = self.__getVariableMapping(
                cellml, varName, element)
            # print(compName, varName, varType, varRef, elementRef, isImported)
            # check if varRef and elementRef are None, that mean the variable is at right side
            if varRef == None and elementRef == None:
                return False
            compRef = elementRef.attrib['name']
            # get maths if varType is state or algebraic
            maths = self.__getMathElement(varRef, varType, elementRef)
            # when reference variable in the same cellml
            if len(maths) > 0 and not isImported:
                compRef = elementRef.attrib['name']
                varRefId = self.getVarId(cellml, compRef, varRef)
                self.sysVars.setMap(varId, varRefId)
                return True
            # when the reference variable in imported cellml
            elif isImported:
                # set cmeta from imported element if available
                self.__setMetaFromElement(elementRef, cellml, compId)
                varRefElement = elementRef.xpath(
                    "//ns:variable[@name='"+varRef+"']", namespaces={'ns': elementRef.nsmap[None]})[0]
                self.__setMetaFromElement(varRefElement, cellml, varId)
                # store maths to variable from imported element
                for math in maths:
                    mathText = re.sub(
                        r'\s\s+', ' ', etree.tostring(math).decode('utf-8'))
                    self.sysVars.addMath(varId, mathText)
                    # set up dependent variables
                    rheVars = [right.text.strip()
                               for right in math[2].iter() if right.tag.endswith('ci')]
                    compVarsT2Id = self.sysVars.getT2Id(
                        self.sysComps.getVariables(compId), short=True)
                    for rheVar in rheVars:
                        if rheVar in compVarsT2Id:
                            self.sysVars.addDependents(
                                varId, compVarsT2Id[rheVar], rheVar)
                return True
            if compName == compRef and varName == varRef:
                return True
            varName = varRef
            compName = compRef

    def __getVariableMapping(self, cellml, varName, element):
        """
        Get variable mapping, including its containing component element
        and status wheter the element from imported cellml or not
        return: varRef, comp element, isImported
        """
        compName = element.attrib['name']
        root = element.getparent()
        # input types
        varPublics = root.xpath('.//ns:component[@name="'+compName+'"]/ns:variable[@name="' +
                                varName+'"]/@public_interface', namespaces={'ns': root.nsmap[None]})
        varPrivates = root.xpath('.//ns:component[@name="'+compName+'"]/ns:variable[@name="' +
                                 varName+'"]/@private_interface', namespaces={'ns': root.nsmap[None]})
        # get mapping / connection
        connections = root.xpath('.//ns:connection[ns:map_components[@component_2="'+compName +
                                 '"]][ns:map_variables[@variable_2="'+varName+'"]]', namespaces={'ns': root.nsmap[None]})
        connections += root.xpath('.//ns:connection[ns:map_components[@component_1="'+compName +
                                  '"]][ns:map_variables[@variable_1="'+varName+'"]]', namespaces={'ns': root.nsmap[None]})
        for connection in connections:
            map_comp = connection.xpath(
                './/ns:map_components', namespaces={'ns': connection.nsmap[None]})
            if map_comp[0].attrib['component_1'] == compName:
                compRef = map_comp[0].attrib['component_2']
                map_var = connection.xpath(
                    './/ns:map_variables[@variable_1="'+varName+'"]', namespaces={'ns': connection.nsmap[None]})
                varRef = map_var[0].attrib['variable_2']
            else:
                compRef = map_comp[0].attrib['component_1']
                map_var = connection.xpath(
                    './/ns:map_variables[@variable_2="'+varName+'"]', namespaces={'ns': connection.nsmap[None]})
                varRef = map_var[0].attrib['variable_1']
            # get connected component and variable
            elementRef = root.xpath(
                './/ns:component[@name="'+compRef+'"]', namespaces={'ns': root.nsmap[None]})
            varRefPublics = root.xpath(
                './/ns:component[@name="'+compRef+'"]/ns:variable[@name="'+varRef+'"]/@public_interface', namespaces={'ns': root.nsmap[None]})
            varRefPrivates = root.xpath(
                './/ns:component[@name="'+compRef+'"]/ns:variable[@name="'+varRef+'"]/@private_interface', namespaces={'ns': root.nsmap[None]})
            parents = root.xpath('.//ns:group[ns:relationship_ref[@relationship="encapsulation"]]/ns:component_ref[@component="' +
                                 compRef+'"]/ns:component_ref[@component="'+compName+'"]', namespaces={'ns': root.nsmap[None]})
            isParent = True if len(parents) > 0 else False
            children = root.xpath('.//ns:group[ns:relationship_ref[@relationship="encapsulation"]]/ns:component_ref[@component="' +
                                  compName+'"]/ns:component_ref[@component="'+compRef+'"]', namespaces={'ns': root.nsmap[None]})
            isChild = True if len(children) > 0 else False
            siblings = root.xpath('.//ns:group[ns:relationship_ref[@relationship="encapsulation"]]/ns:component_ref[ns:component_ref[@component="' +
                                  compRef+'"] and ns:component_ref[@component="'+compName+'"]]', namespaces={'ns': root.nsmap[None]})
            isSibling = True if len(siblings) > 0 else False
            # print(compName, varName, compRef, varRef, len(elementRef))

            # check whether elementRef[0] is imported or note
            parentElement = elementRef[0].getparent()
            if parentElement.tag.endswith('import'):
                if compRef in self.importedElements['component']:
                    return varRef, self.importedElements['component'][compRef], True

            # check other relationship condition caused by encapsulation
            if isParent:
                if 'in' in varPublics and 'out' in varRefPrivates:
                    return varRef, elementRef[0], False
            elif isSibling:
                if 'in' in varPublics and 'out' in varRefPublics:
                    return varRef, elementRef[0], False
            elif isChild:
                if 'in' in varRefPrivates and 'out' in varPublics:
                    return varRef, elementRef[0], False
            elif len(elementRef) > 0:
                return varRef, elementRef[0], False
        return None, None, False

    def __getVarDep(self, cellml, varName, element):
        compName = element.attrib['name']
        root = element.getparent()
        usedComps = []
        while True:
            # get mapping / connection
            connections = root.xpath('.//ns:connection[ns:map_components[@component_2="'+compName +
                                     '"]][ns:map_variables[@variable_2="'+varName+'"]]', namespaces={'ns': root.nsmap[None]})
            connections += root.xpath('.//ns:connection[ns:map_components[@component_1="'+compName +
                                      '"]][ns:map_variables[@variable_1="'+varName+'"]]', namespaces={'ns': root.nsmap[None]})
            # remove map that already checked
            for i in range(len(connections)-1, -1, -1):
                map_comp = connections[i].xpath(
                    './/ns:map_components', namespaces={'ns': connections[i].nsmap[None]})
                checkCompName1 = map_comp[0].attrib['component_1']
                checkCompName2 = map_comp[0].attrib['component_2']
                if checkCompName1 in usedComps or checkCompName2 in usedComps:
                    connections.pop(i)
            # check connection
            if len(connections) > 0:
                map_comp = connections[0].xpath(
                    './/ns:map_components', namespaces={'ns': connections[0].nsmap[None]})
                if map_comp[0].attrib['component_1'] == compName:
                    compRef = map_comp[0].attrib['component_2']
                    map_var = connections[0].xpath(
                        './/ns:map_variables[@variable_1="'+varName+'"]', namespaces={'ns': connections[0].nsmap[None]})
                    varRef = map_var[0].attrib['variable_2']
                else:
                    compRef = map_comp[0].attrib['component_1']
                    map_var = connections[0].xpath(
                        './/ns:map_variables[@variable_2="'+varName+'"]', namespaces={'ns': connections[0].nsmap[None]})
                    varRef = map_var[0].attrib['variable_1']

                varRefId = self.getVarId(cellml, compRef, varRef)
                if varRefId != None:
                    return varRefId
            else:
                return None
            usedComps += [compName]
            compName, varName = compRef, varRef

    def getVarId(self, cellml, compName, varName):
        compsT2Id = self.sysComps.getT2Id(ids=cellml['components'])
        if compName in compsT2Id:
            varIds = self.sysComps.getVariables(compsT2Id[compName])
            varsT2Id = self.sysVars.getT2Id(ids=varIds, short=True)
            if varName in varsT2Id:
                return varsT2Id[varName]
            else:
                return None
        else:
            return None
    # END: groups functions for components extraction

    def __getDocumentation(self, root, cellml):
        # find version
        cellml['version'] = root.nsmap[None].rsplit('/', 2)[-1]
        # find documentation
        for element in root:
            try:  # prevent cyton object error
                if element.tag.endswith('documentation'):
                    ns = '{'+element.nsmap[None] + '}'
                    # get and manage model information
                    info = element.findall('.//' + ns + 'articleinfo')
                    for child in element.find('.//' + ns + 'article'):
                        if child.tag == ns + 'articleinfo':
                            cellml['modelInfo'] = list(xmltodict.parse(re.sub(
                                r'\s\s+', ' ', etree.tostring(child).decode('utf-8'))).values())[0]
                        elif 'id' in child.attrib:
                            if child.attrib['id'] == 'sec_status':
                                cellml['modelStatus'] = list(xmltodict.parse(re.sub(
                                    r'\s\s+', ' ', etree.tostring(child).decode('utf-8'))).values())[0]
                            elif child.attrib['id'] == 'sec_structure':
                                cellml['articleRef'] = list(xmltodict.parse(re.sub(
                                    r'\s\s+', ' ', etree.tostring(child).decode('utf-8'))).values())[0]
                    # get and manage figures
                    figures = element.findall('.//' + ns + 'informalfigure')
                    cellml['images'] = self.sysImages.add(figures, cellml)
            except Exception as e:
                print(e)

    def addSedml(self, id, sedmlId):
        url = self.getUrl(id=id)
        if 'sedml' in self.data[url]:
            if sedmlId not in self.data[url]['sedml']:
                self.data[url]['sedml'] += [sedmlId]
        else:
            self.data[url]['sedml'] = [sedmlId]


def showStatistic(self):
    indicator = ['ma', 'chebi', 'pr', 'go', 'opb', 'fma', 'cl', 'uberon']
    # check rdf files
    rdfFileData = {}
    self.rdfGraph = loadPickle(CURRENT_PATH, RESOURCE_DIR, 'rdf.graph')

    # merge all RDF tag in a cellml to self.rdfGraph
    for k, v in self.data.items():
        rdfFileData[k] = {'runable': 0, 'isRdf': 0,
                          'isAnnotated': 0, 'isSemgen': 0}
        rdfFileData[k]['runable'] = 0 if v['status'] == self.statusC['invalid'] else 1
        path = os.path.join(CURRENT_PATH, WORKSPACE_DIR,
                            v['workingDir'], v['cellml'])
        rdfPath = 'file://' + urilib.quote(path)
        parser = etree.XMLParser(recover=True, remove_comments=True)
        root = etree.parse(path, parser).getroot()
        if 'cmeta' in root.nsmap:
            nsmeta = root.nsmap['cmeta']
        else:
            continue
        rdfElements = root.xpath(".//*[local-name()='RDF']")
        for rdfElement in rdfElements:
            for desc in rdfElement.xpath(".//*[local-name()='Description'][@*[local-name()='about']]"):
                if any(x in ['rdf', 'RDF'] for x in desc.nsmap):
                    if 'rdf' in desc.nsmap:
                        att = '{'+desc.nsmap['rdf']+'}about'
                    elif 'RDF' in desc.nsmap:
                        att = '{'+desc.nsmap['RDF']+'}about'
                    if desc.attrib[att].startswith('#'):
                        desc.attrib[att] = rdfPath+desc.attrib[att]
            try:
                self.rdfGraph.parse(data=etree.tostring(rdfElement))
            except:
                pass
        # now check the cellml:
        metas = root.xpath(".//@cmeta:id", namespaces={'cmeta': nsmeta})
        if len(metas) > 0:
            rdfFileData[k]['isRdf'] = 1
            rdfFileData[k]['isSemgen'] = 0 if 'semsim' not in root.nsmap else 1
        for meta in metas:
            sbj = rdfPath+'#'+meta
            triples, leaves = self.__getTriplesOfMeta(rdflib.URIRef(sbj))
            # print(meta, leaves)
            for leave in leaves:
                leave = str(leave).lower()
                if leave.startswith('http') and any(onto in leave for onto in indicator):
                    rdfFileData[k]['isAnnotated'] = 1
                    break
                else:
                    rdfFileData[k]['isAnnotated'] = 0
            if rdfFileData[k]['isAnnotated'] == 1:
                break
    # summarise the statistic:
    summary = {0: {'tot': 0, 'isRdf': 0, 'isAnnotated': 0, 'isSemgen': 0}, 1: {
        'tot': 0, 'isRdf': 0, 'isAnnotated': 0, 'isSemgen': 0}}
    for k, v in rdfFileData.items():
        summary[v['runable']]['tot'] += 1
        summary[v['runable']]['isRdf'] += v['isRdf']
        summary[v['runable']]['isAnnotated'] += v['isAnnotated']
        summary[v['runable']]['isSemgen'] += v['isSemgen']
    print(summary)
    print(summary)
