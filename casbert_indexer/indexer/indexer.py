from ..general import RESOURCE_DIR, CURRENT_PATH, ONTOLOGY_DIR
from ..general import RS_VARIABLE, RS_CELLML, RS_CLUSTERER, RS_ONTOLOGY
from ..general import loadJson, dumpJson, saveToFlatFile, getAllFilesInDir
from ..general import loadFromFlatFile, saveBinaryInteger, loadBinaryInteger
from ..general import loadPickle, dumpPickle, regexTokeniser, getTokens
from ..colls.variable import Variables
from ..colls.cellml import Cellmls
from ..colls.equation import Maths

from .clusterer import CellmlClusterer

import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


class Indexer:
    FEATURE_DOCUMENT = 0
    FEATURE_RDF = 1
    FEATURE_ONTO = 2
    FEATURE_DOI = 3

    #LIST OF METHODS
    MTD_BAG_OF_WORDS = 0

    def __init__(self):
        """load ontologies"""
        self.__loadOntologies()

        """load all required data"""
        self.__loadData()

        """generate cluster of cellmls"""
        self.createClusterer()

    def createBoWIndex(self, destFile, lower=False, stem=None, lemma=False):
        self.invIdxVar = {}
        self.metaVar = {'general': {'totalTerms': 0,
                                    'totalData': len(self.vars['data'])}, 'data': {}}
        for varId, value in self.vars['data'].items():
            text = ''
            """index from RDF"""
            # try:
            if 'rdfLeaves' in value:
                for leaf in value['rdfLeaves']:
                    leaf = str(leaf).strip()
                    if leaf.startswith('http://'):
                        text += self.__getOntoClassText(leaf) + ' '
                    elif leaf.startswith('file://'):
                        pass
                    else:
                        text += leaf + ' '
            # modify text to multiple setting and get tokens
            tokens = getTokens(text, lower=lower, stem=stem, lemma=lemma)
            self.__generateTermsIndex(varId, tokens)
            self.settings = {'lower': lower, 'stem': stem, 'lemma': lemma}
            # save variable local and general metadata
            self.metaVar['data'][varId] = {'len': len(tokens)}
            self.metaVar['general']['totalTerms'] += len(tokens)

        dumpJson({'setting': self.settings, 'index': self.invIdxVar},
                 RESOURCE_DIR, destFile)
        dumpJson(self.metaVar, RESOURCE_DIR, destFile+'_map')
        print(len(self.ontologies.index))

    def __getOntoClassText(self, classId):
        classId = self.__parseOntoClassId(classId)
        df = self.ontologies
        if classId in df.index:
            # temporarily, extract all textual information
            text = classId + '. '
            dfClassId = df.loc[classId]
            # print(classId, dfClassId.index)
            for key in dfClassId.index:
                if type(dfClassId[key]) == str:
                    text += key + ' : ' + dfClassId[key] + '. '
            return text
        return ''

    def __parseOntoClassId(self, classId):
        idPart = classId[classId.rfind('/')+1:]
        idPart = idPart.replace('_', ':')
        if len(idPart.split(':')) >= 2:
            return idPart

    def __generateTermsIndex(self, varId, tokens):
        for token in tokens:
            if token not in self.invIdxVar:
                self.invIdxVar[token] = {}
            if varId not in self.invIdxVar[token]:
                self.invIdxVar[token][varId] = 0
            self.invIdxVar[token][varId] += 1

    def getFeatures(self, *featureTypes):
        for featureType in featureTypes:
            if featureType in Indexer.FEATURE_DOCUMENT:
                self.__getFeatureDocument()
            elif featureType in Indexer.FEATURE_RDF:
                self.__getFeatureRdf()
            elif featureType in Indexer.FEATURE_ONTO:
                self.__getFeatureOnto()
            elif featureType in Indexer.FEATURE_DOI:
                self.__getFeatureDoi()

    def __getFeatureDocument(self):
        pass

    def __getFeatureRdf(self):
        pass

    def __getFeatureOnto(self):
        pass

    def __getFeatureDoi(self):
        pass

    def buildMap(self):
        pass

    def __loadOntologies(self):
        print('Loading ontologies ...')
        listData, dfCsv = [], pd.DataFrame()
        allFiles = getAllFilesInDir(ONTOLOGY_DIR)
        if any(RS_ONTOLOGY in file for file in allFiles):
            self.ontologies = loadPickle(ONTOLOGY_DIR, RS_ONTOLOGY)
            self.ontoName = {idx.split(
                ':')[0] for idx in self.ontologies.index if idx[0].isupper() and ':' in idx}
            return
        for ontoFile in allFiles:
            if ontoFile.endswith('.obo'):
                with open(ontoFile) as fp:
                    while True:
                        line = fp.readline()
                        if not line:
                            break
                        line = line.strip()

                        if line == '[Term]':
                            data = {}
                            while True:
                                line = fp.readline().strip()
                                keyVals = line.split(': ', 1)
                                if len(keyVals) == 2:
                                    if keyVals[0] not in data:
                                        data[keyVals[0]] = keyVals[1]
                                    else:
                                        data[keyVals[0]] += '|'+keyVals[1]
                                if not line:
                                    listData += [data]
                                    break
            elif '.csv' in ontoFile:
                df = pd.read_csv(ontoFile, sep=',', header=0,
                                 index_col='Class ID')
                dfCsv = dfCsv.append(df, sort=False)
        # tranform dfCsv
        dfCsv = dfCsv.dropna(axis='columns', how='all')
        dfCsv = dfCsv.rename(index=lambda s: s[s.rfind('/')+1:])
        dfCsv = dfCsv.rename(
            index=lambda s: s[s.rfind('#')+1:].replace('_', ':'))
        dfCsv['synonym'] = dfCsv['synonyms'].fillna(
            '') + ('|'+dfCsv['synonym']).fillna('')
        dfCsv = dfCsv.drop(columns=['synonyms', 'definition', 'preferred label', 'alternative label',
                                    'http://www.w3.org/2000/01/rdf-schema#label',
                                    'http://data.bioontology.org/metadata/prefixIRI'])
        dfCsv = dfCsv.rename(columns={'http://www.w3.org/2000/01/rdf-schema#comment': 'comment',
                                      'http://purl.org/dc/elements/1.1/creator': 'created_by',
                                      'http://bhi.washington.edu/OPB#discussion': 'discussion',
                                      'http://bhi.washington.edu/OPB#classTerm': 'classTerm',
                                      'Definitions': 'def', 'Preferred Label': 'name'})
        # transform df
        df = df.dropna(axis='columns', how='all')
        df = pd.DataFrame(listData)
        df = df.set_index('id')
        df = df.append(dfCsv, sort=False)
        df = df.groupby(df.index).first()  # delete duplicate
        dumpPickle(df, ONTOLOGY_DIR, RS_ONTOLOGY)
        self.ontologies = df
        self.ontoName = {
            idx.split(':')[0] for idx in df.index if idx[0].isupper() and ':' in idx}

    def __loadData(self):
        print('Loading required data, e.g. cellml, sedml, variable, workspaces, etc ... ')
        self.vars = loadJson(RESOURCE_DIR, RS_VARIABLE)
        self.cellmls = loadJson(RESOURCE_DIR, RS_CELLML)

    def createClusterer(self):
        print('Create clusterer with XPath and structure features using HDBSCAN')
        clusterer = CellmlClusterer(cellmls=self.cellmls)
        dumpJson(clusterer.getDict(), RESOURCE_DIR, RS_CLUSTERER)

    def close(self):
        self.__closeOntologies()

    def __closeOntologies(self):
        del self.ontologies
        import gc
        gc.collect()

    """
    Embedding Indexing Part
    """

    def createCasbertIndex(self, destFolder, alpha=0.22):
        if not os.path.exists(destFolder):
            print('Error: The destination folder is not exist')
            return
        self.entityEmbedding = {}
        self.__loadRequiredEmbeddings()
        self.__createVariableEmbeddings(alpha)
        self.__createComponentEmbeddings()
        self.__createCellmlEmbeddings()
        file = os.path.join(CURRENT_PATH, RESOURCE_DIR, 'casbert_pmr.pt')
        torch.save(self.entityEmbedding, file)
        Indexer.__saveCasbertIndex(destFolder)

    def __loadRequiredEmbeddings(self):
        BERTModel = 'multi-qa-MiniLM-L6-cos-v1'
        from sentence_transformers import SentenceTransformer
        self.sbert = SentenceTransformer(BERTModel)

        def change2embedding(txt):
            return self.sbert.encode(txt, convert_to_tensor=True)

        def from_synonym(syn):
            try:
                tmp = []
                for txt in syn.split('|'):
                    if len(txt) == 0:
                        continue
                    if (txt.find('"') < txt.rfind('"')):
                        tmp += [txt[txt.find('"')+1:txt.rfind('"')]]
                    elif (txt.find("'") < txt.rfind("'")):
                        tmp += [txt[txt.find("'")+1:txt.rfind("'")]]
                    else:
                        tmp += [txt]
                return torch.mean(change2embedding(tmp), 0)
            except:
                return torch.zeros(embedShape)

        # get classIds and predicates
        classIds, predicates = [], []
        for varId, value in self.vars['data'].items():
            if 'rdfLeaves' in value:
                for leaf in value['rdfLeaves']:
                    leaf = str(leaf).strip()
                    # get ontology class para
                    classId = None
                    if leaf.startswith('http'):
                        classId = leaf.rsplit(
                            '/', 1)[-1].rsplit('#', 1)[-1].replace('_', ':')
                    elif leaf.startswith('urn:miriam'):
                        classId = leaf.rsplit(':', 1)[-1].replace('%3A', ':')
                    if classId is not None and classId in self.ontologies.index:
                        classIds += [classId]
            if 'rdf' in value:
                for triple in value['rdf']:
                    predicates += [triple[1]]
        classIds = list(set(classIds))
        predicates = list(set(predicates))

        # convert ontolo classes to embedding
        self.ontoEmbedding = {'classId': classIds, 'name': [], 'synonym': [], 'def': [],
                              'name_synonym_def': [], 'name_synonym_def': []}
        embedShape = 384
        for classId in tqdm(classIds):
            ocls = self.ontologies.loc[classId]
            self.ontoEmbedding['name'] += [change2embedding(ocls['name'])]
            self.ontoEmbedding['synonym'] += [
                from_synonym(ocls['synonym'])]
            if isinstance(ocls['def'], str):
                self.ontoEmbedding['def'] += [
                    change2embedding(ocls['def'])]
            else:
                self.ontoEmbedding['def'] += [torch.zeros(embedShape)]
        self.ontoEmbedding['name'] = torch.stack(
            self.ontoEmbedding['name'], dim=0)
        self.ontoEmbedding['synonym'] = torch.stack(
            self.ontoEmbedding['synonym'], dim=0)
        self.ontoEmbedding['def'] = torch.stack(
            self.ontoEmbedding['def'], dim=0)
        # name_synonym
        tensor = torch.stack(
            [self.ontoEmbedding['name'], self.ontoEmbedding['synonym']], dim=0)
        self.ontoEmbedding['name_synonym'] = torch.div(torch.nansum(
            tensor, dim=0), (~torch.isnan(tensor)).count_nonzero(dim=0))
        # name_synonym_def
        tensor = torch.stack(
            [self.ontoEmbedding['name'], self.ontoEmbedding['synonym'], self.ontoEmbedding['def']], dim=0)
        self.ontoEmbedding['name_synonym_def'] = torch.div(torch.nansum(
            tensor, dim=0), (~torch.isnan(tensor)).count_nonzero(dim=0))

        # convert predicates to embeddings
        from re import finditer

        def _camelCaseSplitter(identifier):
            matches = finditer(
                '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
            return ' '.join([m.group(0) for m in matches])

        self.predicateEmbedding = {}
        for p in predicates:
            predicate = _camelCaseSplitter(
                p.split('#')[-1].split('/')[-1]) if p.startswith('http') else p
            if len(predicate.split()) > 0 and len(predicate) > 2:
                self.predicateEmbedding[p] = self.sbert.encode(
                    predicate, convert_to_tensor=True)

        # save ontology embeddings and predicate embeddings

    def __createVariableEmbeddings(self, alpha):
        # classId to content (ontology classes and predicate paths)
        varKeys = {}
        varIds = []
        varEmbeddings = {'embedding': {'class': [], 'class_predicate': []}}

        # ontology class - predicate to varIds
        classPredicateMap = {'multi': {}, 'single': {}}
        # the embedding of ontology class - predicate
        classPredicateEmbedding = {'multi': [], 'single': []}

        def getPredicatePaths(obj=None, prevPred=[], arrRdf=None):
            if obj != None:
                rows = arrRdf[np.where(arrRdf[:, 2] == obj)]
            if len(rows) > 0:
                finalPreds = []
                for row in rows:
                    finalPreds += getPredicatePaths(
                        obj=row[0], prevPred=[row[1]]+prevPred, arrRdf=arrRdf)
                return list(finalPreds)
            return [prevPred]

        def getClassText(classId, features=['name']):
            """
            features: name, synonym, parent, def
            """
            corpus = {}
            if classId in self.ontoEmbedding['classId']:
                for feature in features:
                    corpus[feature] = self.ontologies.loc[classId][feature]
            return corpus

        def getClassEmbedding(classId, feature='name_synonym'):
            """
            classId: an id of a class such as 'CHEBI:29101'
            feature: name, name_synonym, name_synonym_def, name_synonym_def, name_synonym_def_parent
            """
            if 'http' in classId:
                classId = classId.rsplit(
                    '/')[-1].split('#')[-1].replace('_', ':')
            if classId in self.ontoEmbedding['classId']:
                return self.ontoEmbedding[feature][self.ontoEmbedding['classId'].index(classId)]
            return None

        def getPredicateEmbedding(predicates):
            """
            predicates: a string or a list of predicate. if it is a list, the mean of embedding is returned
            """
            if type(predicates) == str:
                predicates = [predicates]
            embeddings = []
            for predicate in predicates:
                if predicate in self.predicateEmbedding:
                    embeddings += [self.predicateEmbedding[predicate]]
            if len(embeddings) == 0:
                return None
            return torch.mean(torch.stack(embeddings), dim=0)

        for varId, value in tqdm(self.vars['data'].items()):
            embeddings = []
            pathClassEmbeddings = []
            terms = {}
            candQuery = []
            if 'rdfLeaves' in value:
                for leaf in value['rdfLeaves']:
                    leaf = str(leaf).strip()
                    # get ontology class para
                    if leaf.startswith('http'):
                        classId = leaf.rsplit(
                            '/', 1)[-1].rsplit('#', 1)[-1].replace('_', ':')
                    elif leaf.startswith('urn:miriam'):
                        classId = leaf.rsplit(':', 1)[-1].replace('%3A', ':')
                    if leaf.startswith('http') or leaf.startswith('urn:miriam'):
                        embedding = getClassEmbedding(classId)
                        # get ontology class embedding
                        if embedding != None:
                            embeddings += [embedding]
                            # get predicates path embedding
                            import numpy as np
                            arrRdf = np.array(value['rdf'])
                            paths = getPredicatePaths(obj=leaf, arrRdf=arrRdf)
                            pathTexts = [
                                [p.rsplit('/')[-1].rsplit('#')[-1] for p in path] for path in paths]
                            pathEmbeddings = [getPredicateEmbedding(
                                path) for path in paths if getPredicateEmbedding(path) != None]
                            if len(pathEmbeddings) > 0:
                                pathEmbedding = alpha * \
                                    torch.mean(torch.stack(
                                        pathEmbeddings, dim=0), dim=0)
                                pathClassEmbedding = torch.sum(torch.stack(
                                    [embedding, pathEmbedding], dim=0), dim=0)/(1+alpha)
                                pathClassEmbeddings += [pathClassEmbedding]
                            else:
                                pathClassEmbeddings += [embedding]

                            # setting for the use of unique classPredicateEmbedding
                            for path in paths:
                                key = leaf + \
                                    '-'.join([p.rsplit('/')[-1].rsplit('#')[-1]
                                              for p in path])
                                #index pair of path and ontology class
                                pEmbedding = getPredicateEmbedding(path)
                                if pEmbedding != None:
                                    tmpEmbedding = torch.mean(torch.stack(
                                        [embedding, pEmbedding], dim=0), dim=0)
                                else:
                                    tmpEmbedding = embedding
                                if key not in classPredicateMap['multi']:
                                    classPredicateEmbedding['multi'] += [
                                        tmpEmbedding]
                                    classPredicateMap['multi'][key] = [varId]
                                else:
                                    classPredicateMap['multi'][key] += [varId]

                                # index single predicate and ontology class
                                for predicate in path:
                                    key = leaf + predicate
                                    pEmbedding = getPredicateEmbedding(
                                        [predicate])
                                    if pEmbedding != None:
                                        tmpEmbedding = torch.mean(torch.stack(
                                            [embedding, pEmbedding], dim=0), dim=0)
                                    else:
                                        tmpEmbedding = embedding
                                    if key not in classPredicateMap['single']:
                                        classPredicateEmbedding['single'] += [
                                            tmpEmbedding]
                                        classPredicateMap['single'][key] = [
                                            varId]
                                    else:
                                        classPredicateMap['single'][key] += [varId]

                            # set term
                            terms[classId] = {'name': getClassText(classId, features=['name'])[
                                'name'], 'path': pathTexts}
                    elif not leaf.startswith('file://'):
                        paths = getPredicatePaths(obj=leaf, arrRdf=arrRdf)
                        for path in paths:
                            if 'description' in ''.join(path):
                                candQuery += [leaf]
                                break

                if len(embeddings) > 0:
                    varKeys[varId] = {'pos': len(
                        varKeys), 'classes': terms}
                    varEmbeddings['embedding']['class'] += [torch.mean(
                        torch.stack(embeddings, dim=0), dim=0)]
                    varEmbeddings['embedding']['class_predicate'] += [torch.mean(
                        torch.stack(pathClassEmbeddings, dim=0), dim=0)]
                    varIds += [varId]

        for k, v in varEmbeddings['embedding'].items():
            varEmbeddings['embedding'][k] = torch.stack(v, dim=0)
        varEmbeddings['id'] = varIds
        varEmbeddings['class'] = varKeys
        self.entityEmbedding['variable'] = varEmbeddings

    def __createComponentEmbeddings(self):
        from ..general import RS_COMPONENT
        components = loadJson(RESOURCE_DIR, RS_COMPONENT)
        varIds = self.entityEmbedding['variable']['id']
        varClasses = self.entityEmbedding['variable']['class']
        varEmbeddings = self.entityEmbedding['variable']['embedding']
        compIds = []
        compEmbedding = {'embedding': {'class': [], 'class_predicate': []}}
        compClasses = {}
        for compId, comp in tqdm(components['data'].items()):
            buffEmbClass = []
            buffEmbClassPred = []
            buffClasses = {}
            if 'variables' not in comp:
                continue
            for varId in comp['variables']:
                if varId in varIds:
                    buffEmbClass += [varEmbeddings['class']
                                     [varIds.index(varId)]]
                    buffEmbClassPred += [varEmbeddings['class_predicate']
                                         [varIds.index(varId)]]
                    buffClasses = {**buffClasses, **
                                   varClasses[varId]['classes']}
            if len(buffEmbClass) > 0:
                compIds += [compId]
                compEmbedding['embedding']['class'] += [
                    torch.mean(torch.stack(buffEmbClass, dim=0), dim=0)]
                compEmbedding['embedding']['class_predicate'] += [
                    torch.mean(torch.stack(buffEmbClassPred, dim=0), dim=0)]
                compClasses[compId] = {'pos': len(
                    compClasses), 'classes': buffClasses}

        for k, v in compEmbedding['embedding'].items():
            compEmbedding['embedding'][k] = torch.stack(v, dim=0)
        compEmbedding['id'] = compIds
        compEmbedding['class'] = compClasses
        self.entityEmbedding['component'] = compEmbedding

    def __createCellmlEmbeddings(self):
        from ..general import RS_IMAGE
        images = loadJson(RESOURCE_DIR, RS_IMAGE)['data']
        varEmbeddings = self.entityEmbedding['variable']['embedding']
        varIds = self.entityEmbedding['variable']['id']

        import en_core_sci_scibert
        nlp = en_core_sci_scibert.load()

        cellmlUrls = []
        cellmlEmbeddings = []
        sedmlIds = []
        sedmlEmbeddings = []
        imageIds = []
        imageEmbeddings = []

        for cellmlUrl, cellml in tqdm(self.cellmls['data'].items()):
            cellmlEmbds = []
            # 1. file name
            if 'cellml' in cellml:
                text = cellml['cellml'].split('.')[0]
                cellmlEmbds += [self.sbert.encode(text,
                                                  convert_to_tensor=True)]

            # 2. variable embeddings
            if 'variables' in cellml:
                for varId in cellml['variables']:
                    if varId in varIds:
                        cellmlEmbds += [varEmbeddings['class']
                                        [varIds.index(varId)]]

            # 3. metadata
            if 'modelMeta' in cellml:
                for k in cellml['modelMeta'].keys():
                    cellmlEmbds += [self.sbert.encode(k,
                                                      convert_to_tensor=True)]

            # basically is the title, so directly converted to embedding
            if 'modelInfo' in cellml:
                if 'title' in cellml['modelInfo']:
                    if isinstance(cellml['modelInfo']['title'], str):
                        cellmlEmbds += [self.sbert.encode(cellml['modelInfo']
                                                          ['title'], convert_to_tensor=True)]
                    elif '#text' in cellml['modelInfo']['title']:
                        cellmlEmbds += [self.sbert.encode(cellml['modelInfo']
                                                          ['title']['#text'], convert_to_tensor=True)]

            # para is complete information but it semisupervised
            if 'articleRef' in cellml:
                data = []
                if 'para' in cellml['articleRef']:
                    if isinstance(cellml['articleRef']['para'], str):
                        data += [cellml['articleRef']['para']]
                    else:
                        for t in cellml['articleRef']['para']:
                            if isinstance(t, str):
                                if len(t.split(" ")) > 25:
                                    data += [t]
                            elif t == None:
                                pass
                            else:
                                if '#text' in t:
                                    data += [t['#text']]

                for d in data:
                    if len(d.split(" ")) < 50:
                        cellmlEmbds += [self.sbert.encode(d,
                                                          convert_to_tensor=True)]
                    else:  # break abstract
                        doc = nlp(d)
                        for ent in doc.ents:
                            cellmlEmbds += [self.sbert.encode(ent.text,
                                                              convert_to_tensor=True)]

            if len(cellmlEmbds) > 0:
                cellmlUrls += [cellmlUrl]
                cellmlEmbedding = torch.mean(
                    torch.stack(cellmlEmbds, dim=0), dim=0)
                cellmlEmbeddings += [cellmlEmbedding]
                if 'sedml' in cellml:
                    for sedmlId in cellml['sedml']:
                        sedmlIds += [sedmlId]
                        sedmlEmbeddings += [cellmlEmbedding]
                if 'images' in cellml:
                    for imageId in cellml['images']:
                        imageIds += [imageId]
                        if 'caption' in images[imageId]:
                            doc = nlp(images[imageId]['caption'])
                            for ent in doc.ents:
                                cellmlEmbds += [self.sbert.encode(
                                    ent.text, convert_to_tensor=True)]
                        imageEmbeddings += [torch.mean(
                            torch.stack(cellmlEmbds, dim=0), dim=0)]

        torchCellmls = torch.stack(cellmlEmbeddings, dim=0)
        torchSedmls = torch.stack(sedmlEmbeddings, dim=0)
        torchImages = torch.stack(imageEmbeddings, dim=0)
        self.entityEmbedding['cellml'] = {'id': cellmlUrls, 'class': None, 'embedding': {
            'class': torchCellmls, 'class_predicate': None}}
        self.entityEmbedding['sedml'] = {'id': sedmlIds, 'class': None, 'embedding': {
            'class': torchSedmls, 'class_predicate': None}}
        self.entityEmbedding['image'] = {'id': imageIds, 'class': None, 'embedding': {
            'class': torchImages, 'class_predicate': None}}

    def __saveCasbertIndex(destFolder):
        root = os.path.join(CURRENT_PATH, RESOURCE_DIR)
        fileSave = os.path.join(destFolder, 'casbert_data.zip')

        import zipfile

        def zipdir(path, ziph):
            # ziph is zipfile handle
            for root, dirs, files in os.walk(path):
                for file in files:
                    # temporarily do not include experiment results
                    if not file.startswith(('.', '_', 'inv', 'rdf')) and not file.endswith(('gz', 'pdf', 'zip', 'checkpoint.json')):
                        ziph.write(os.path.join(root, file),
                                   os.path.relpath(os.path.join(root, file),
                                                   os.path.join(path, '..')))

        with zipfile.ZipFile(fileSave, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir(root, zipf)

    def getCopyCasbertIndex(destFolder='.'):
        Indexer.__saveCasbertIndex(destFolder)
