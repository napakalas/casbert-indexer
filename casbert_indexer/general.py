import json
import os
import pickle
import gzip
import io
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
from nltk import pos_tag, regexp_tokenize
from lxml import etree

"""init global variable"""
PREFF = ''  # this is useful to differentiate PMR from different resources.
# if the resource from 'models' than leave it blank
CURRENT_PATH = os.path.dirname(
    os.path.realpath(__file__))  # use this on real python
# interchangeable to models, teaching, or staging
PMR_SERVER = 'https://models.physiomeproject.org/'
# PMR_SEVER = {'models':'https://models.physiomeproject.org/', \
#              'teaching':'https://teaching.physiomeproject.org/',
#              'staging':'https://staging.physiomeproject.org/'}
# WORKSPACE_DIR = {'models':'models_workspace', \
#              'teaching':'teaching_workspace',
#              'staging':'staging_workspace'}
# RESOURCE_DIR = {'models':'models_resources', \
#              'teaching':'teaching_resources',
#              'staging':'staging_resources'}
WORKSPACE_DIR = PREFF + 'workspaces'
RESOURCE_DIR = PREFF + 'resources'
ONTOLOGY_DIR = 'ontologies'
SEDML_IMG_DIR = 'sedmlImages'
SEDML_RSL_DIR = 'sedmlResults'

# RESOURCE FILE COLLECTION
RS_CATEGORY = 'listOfCategory.json'
RS_CELLML = 'listOfCellml.json'
RS_COMPONENT = 'listOfComponent.json'
RS_EXPOSURE = 'listOfExposure.json'
RS_IMAGE = 'listOfImage.json'
RS_MATH = 'listOfMath.json'
RS_SEDML = 'listOfSedml.json'
RS_UNIT = 'listOfUnit.json'
RS_VARIABLE = 'listOfVariable.json'
RS_VIEW = 'listOfView.json'
RS_WORKSPACE = 'listOfWorkspace.json'
RS_XSL = 'ctopff.xsl'
RS_CLUSTERER = 'cellmlClusterer.json'
RS_ONTOLOGY = 'ontoDf.gz'

IMG_EXT = '.png'

# INDEXING SETTING
STEM_PORTER = 0
STEM_LANCASTER = 1
IS_LOWER = True
IS_LEMMA = True

# RETRIVAL ALGORITHMS
ALG_BOOL = 0
ALG_BM25 = 1

# PRESENTATION DESTINATION
TO_WEB = 0
TO_JUPYTER = 1


def loadJson(*paths):
    file = os.path.join(CURRENT_PATH, *paths)
    isExist = os.path.exists(file)
    if isExist:
        with open(file, 'r') as fp:
            data = json.load(fp)
        fp.close()
        return data
    else:
        return {}


def dumpJson(data, *paths):
    file = os.path.join(CURRENT_PATH, *paths)
    with open(file, 'w') as fp:
        json.dump(data, fp)
    fp.close()


def saveToFlatFile(data, *paths):
    file = os.path.join(CURRENT_PATH, *paths)
    f = open(file, 'w+')
    for datum in data:
        f.write(str(datum).replace('\n', ' ').replace('\r', ' ') + '\n')
    f.close()


def loadFromFlatFile(*paths):
    file = os.path.join(CURRENT_PATH, *paths)
    try:
        f = open(file, 'r')
        lines = f.readlines()
        f.close()
        return lines
    except:
        return []


def saveBinaryInteger(data, *paths):
    import struct
    file = os.path.join(CURRENT_PATH, *paths)
    with open(file, "wb") as f:
        for x in data:
            f.write(struct.pack('i', x))  # 4bytes
    f.close()


def loadBinaryInteger(*paths):
    import struct
    file = os.path.join(CURRENT_PATH, *paths)
    with open(file, 'rb') as f:
        bdata = []
        while True:
            bytes = f.read(4)
            if bytes == b'':
                break
            else:
                bdata.append(struct.unpack('i', bytes)[0])  # 4bytes
    f.close()
    return bdata


def dumpPickle(data, *paths):
    filename = os.path.join(CURRENT_PATH, *paths)
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def loadPickle(*paths):
    filename = os.path.join(CURRENT_PATH, *paths)
    file = gzip.GzipFile(filename, 'rb')
    data = pickle.load(file)
    file.close()
    return data


def getAllFilesInDir(*paths):
    drc = os.path.join(CURRENT_PATH, *paths)
    lst = []
    for path, subdirs, files in os.walk(drc):
        for name in files:
            lst += [os.path.join(path, name)]
    return lst

# get list of URLs inside a particulat URL in the PMR


def getUrlFromPmr(url):
    import requests
    r = requests.get(
        url, headers={"Accept": "application/vnd.physiome.pmr2.json.1"})
    urls = [link['href'] for link in r.json()['collection']['links']]
    return urls

# get json from PMR based on URL address


def getJsonFromPmr(url):
    import requests
    r = requests.get(
        url, headers={"Accept": "application/vnd.physiome.pmr2.json.1"})
    try:
        return r.json()['collection']
    except:
        return {}


greek_code2name = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

greek_name2code = {v: k for k, v in greek_code2name.items()}


def m_c2p(math_c, destination=TO_WEB):
    preff = '{http://www.w3.org/1998/Math/MathML}'
    if '<math ' not in math_c:
        math_c = '<math xmlns="http://www.w3.org/1998/Math/MathML">' + math_c + '</math>'
    mml_dom = etree.fromstring(math_c)
    xslPath = os.path.join(CURRENT_PATH, RESOURCE_DIR, RS_XSL)
    xslt = etree.parse(xslPath)
    transform = etree.XSLT(xslt)
    mmldom = transform(mml_dom)
    root = mmldom.getroot()
    for name in root.iter(preff + 'mi'):
        comps = name.text.split('_')
        if len(comps) > 1:
            if destination == TO_WEB:
                comps = [
                    '&' + x + ';' if x in greek_name2code else x for x in comps]
            elif destination == TO_JUPYTER:
                comps = [greek_name2code[x]
                         if x in greek_name2code else x for x in comps]
            name.tag = preff + 'msub'
            name.text = ''
            name.attrib.pop('mathvariant')
            rightEl = etree.Element(
                preff + 'mn' if comps[len(comps) - 1].isnumeric() else 'mi', mathvariant='italic')
            rightEl.text = comps[len(comps) - 1]
            leftEl = etree.Element(
                preff + 'mn' if comps[len(comps) - 2].isnumeric() else 'mi', mathvariant='italic')
            leftEl.text = comps[len(comps) - 2]
            if len(comps) > 2:
                subEl = etree.Element(preff + 'msub')
                subEl.append(leftEl)
                subEl.append(rightEl)
                for i in range(len(comps) - 3, -1, -1):
                    rightEl = subEl
                    leftEl = etree.Element(
                        preff + 'mn' if comps[i].isnumeric() else 'mi', mathvariant='italic')
                    leftEl.text = comps[i]
                    subEl = etree.Element(preff + 'msub')
                    subEl.append(leftEl)
                    subEl.append(rightEl)
            name.append(leftEl)
            name.append(rightEl)
    for elem in root.iter('*'):
        if elem.text != None:
            elem.text = elem.text.strip()
    return str(mmldom).replace('·', '&#xB7;').replace('−', '-').replace('<?xml version="1.0"?>', '<?xml version="1.0" encoding="UTF-8"?>')


def regexTokeniser(text):
    pattern = r'''(?x)    # set flag to allow verbose regexps
        (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:[:,-]\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        # | \.\.\.            # ellipsis
        # | \.\.            # ellipsis
        # | \.            # ellipsis
        # | [][.,;"'?():-_`]  # these are separate tokens; includes [ and ]
        '''
    return regexp_tokenize(text, pattern)


def getTokens(text, **settings):
    if settings['lower'] == IS_LOWER:
        text = text.lower()
    if settings['stem'] == STEM_PORTER:
        stemmer = PorterStemmer()
        text = stemmer.stem(text)
    elif settings['stem'] == STEM_LANCASTER:
        stemmer = LancasterStemmer()
        text = stemmer.stem(text)
    if settings['lemma'] == IS_LEMMA:
        wnl = WordNetLemmatizer()
        tokens = [wnl.lemmatize(i, j[0].lower()) if j[0].lower() in [
            'a', 'n', 'v'] else wnl.lemmatize(i) for i, j in pos_tag(regexTokeniser(text))]
    if 'tokens' not in locals():
        tokens = regexTokeniser(text)
    return tokens


def zipResources():
    root = os.path.join(CURRENT_PATH, RESOURCE_DIR)
    fileSave = os.path.join(root, 'casbert_index_data.zip')

    import zipfile

    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                # temporarily do not include experiment results
                if not file.startswith(('.', '_')) and not file.endswith(('.gz', '.pdf', 'zip', 'checkpoint.json')):
                    ziph.write(os.path.join(root, file),
                               os.path.relpath(os.path.join(root, file),
                                               os.path.join(path, '..')))

    with zipfile.ZipFile(fileSave, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(root, zipf)
