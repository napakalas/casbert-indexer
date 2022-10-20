from ..general import dumpJson, loadJson, getTokens
from ..general import RESOURCE_DIR, ALG_BOOL, ALG_BM25

import math

class PmrIndex:

    def __init__(self, fileIndex):
        jsonIndex = loadJson(RESOURCE_DIR, fileIndex)
        self.invIndex = jsonIndex['index']
        self.settings = jsonIndex['setting']
        self.meta = loadJson(RESOURCE_DIR, fileIndex+'_map')

    def getResults(self, query, top=-1, page=1, algorithm=ALG_BM25):
        candidates = {}
        # get candidate based on the algorithm
        if algorithm == ALG_BM25:
            candidates = self.__getResultsBM25(query)
        candidateLen = len(candidates)
        # return candidate as stated settings
        if top < 0:
            candidates = dict(sorted(candidates.items(), key=lambda item: item[1], reverse=True))
            return {'candidates':candidates, 'length':candidateLen}
        if page in [0,1]:
            left, right = 0, top
        else:
            left, right = (page-1)*top, page*top
        if len(candidates) <= left:
            candidates = {}
        elif len(candidates) < right:
            candidates = {k: v for k, v in list(sorted(candidates.items(), key=lambda item: item[1], reverse=True))[left:]}
        else:
            candidates = {k: v for k, v in list(sorted(candidates.items(), key=lambda item: item[1], reverse=True))[left:right]}
        print([(key,candidates[key]) for key in list(candidates.keys())])
        return {'candidates':candidates, 'length':candidateLen}

    def __getResultsBM25(self, query):
        tokens = getTokens(query,**self.settings)
        candidates = {}
        for token in tokens:
            if token in self.invIndex:
                # calculate IDF
                idf = self.__getIdf(self.meta['general']['totalData'], len(self.invIndex[token]))
                # print(token, self.invIndex[token])
                for candidate in self.invIndex[token]:
                    if candidate not in candidates:
                        candidates[candidate] = 0
                    freqTermDoc =  self.invIndex[token][candidate]
                    docLength = self.meta['data'][candidate]['len']
                    avgDataLength = self.meta['general']['totalTerms'] / self.meta['general']['totalData']
                    candidates[candidate] += idf * self.__getTf(freqTermDoc, docLength, avgDataLength)
                    if candidate == 'VarId-117001':
                        print(candidate, idf, self.__getTf(freqTermDoc, docLength, avgDataLength), freqTermDoc, docLength, avgDataLength, token)
                    if candidate == 'VarId-34920':
                        print(candidate, idf, self.__getTf(freqTermDoc, docLength, avgDataLength), freqTermDoc, docLength, avgDataLength, token)
                    if candidate == 'VarId-24548':
                        print(candidate, idf, self.__getTf(freqTermDoc, docLength, avgDataLength), freqTermDoc, docLength, avgDataLength, token)
        return candidates

    def __getIdf(self, totData, totTermData):
        idf = math.log((totData - totTermData + 0.5) / (totTermData + 0.5) + 1)
        return idf

    def __getTf(self, freqTermDoc, docLength, avgDataLength, k1=1.2, b=0.75):
        tf = (freqTermDoc * (k1 + 1)) / (freqTermDoc + k1 * (1 - b + b * docLength / avgDataLength))
        return tf
