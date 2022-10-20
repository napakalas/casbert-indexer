# casbert-indexer

This module supports [CASBERT](https://github.com/napakalas/casbert.git) to search for entities present in the biosimulation model in the [PMR](https://models.physiomeproject.org/welcome). Processes that can be done with this module include:
- Data crawling and data organization
- Index building
- Searching for entities (<span style="color:red"> partially moved to [CASBERT](https://github.com/napakalas/casbert.git)  to provide flexibility for general users </span>)


## Installation
```
pip install git+https://github.com/napakalas/casbert-indexer.git
```
This module is recommended for use with the OpenCOR kernel. As for the details of the installation that worked well:
1. Download [OpenCOR](https://opencor.ws/)
2. Run the application
3. Using menu bar, View > Windows > Python Console
4. Install casbert_indexer:
    ```python 
    !pip install git+https://github.com/napakalas/casbert-indexer.git  
    ```
    
## Data crawling and data organization
  * download/update workspaces  
  * extract data/results from the new workspaces
  * organised the generated data/results
  
This process can only be run with the python console in the [OpenCOR](https://opencor.ws/) application. Using the Jupyter Lab/Notebook console with the OpenCOR kernel actually works, but sometimes it's unstable causing the module to crash. With this module, we validate the model, run the simulation, and save the results in the form of data and images. Files extracted include CellML, RDF, html, SEDML, OMEX, images (e.g. jpeg, png). Meanwhile, the identified entities include categories, models, components, equations, exposures, images, simulations, units, variables, and workspaces.
### Instructions
```python
# import Crawler
from casbert_indexer import Crawler
# create a Crawler object
cw = Crawler()

# download/update models
cw.update()

# validate new/updated models
cw.validate()

# extraction
cw.extract()
```

## Indexing
Indexing is used to create indexes with bag-of-words and embedding methods. Indexes with the bag-of-words method can be used directly in this module via the Searcher submodule, while those using embedding can be used with the [CASBERT](https://github.com/napakalas/casbert.git) module. In addition, biosimulation models are also grouped based on similarity of documentation, structure, and depth and breadth of elements using HDBScan. This grouping is used to find models that are similar but not annotated with an ontology or RDF.

### Instructions
#### Creating indexes (Bag of Words and CASBERT)
```python
# import Indexer
from casbert_indexer import Indexer

# create an Indexer object
# this also automatically decides the custer of new/updated models
idx = Indexer()

# create indexes for bag-of-word method
indexFile = 'indexBoW' 
idx.createBoWIndex(indexFile, lower=False, stem=None, lemma=False)
  ## indexFile, (the file name to save indexes)
  ## lower, (True -> convert text to lower case, False -> no conversion)
  ## stem, (STEM_PORTER, STEM_LANCASTER -> imported from casbert_indexer.general) 
  ## lemma, (True -> apply lemmatisation, False -> no lemmatisation)

# create index for embedding method, resulting casbert_data.zip file
destFolder = 'Documents/'
idx.createCasbertIndex(destFolder, alpha=0.22)
  ## destFolder, (folder name to store casbert_data.zip file,
  ##              later, the file is used in casbert package)
  ## alpha, ((optional) the weight of triples' predicates in embeddings,
  ##         should be between 0 and 1)

# get copy of casbert index, returning casbert_data.zip
Indexer.getCopyCasbertIndex(destFolder)
```

#### Merge CASBERT index with CASBERT
```python
file = 'Documents/casbert_data.zip'
updateIndexes(file)
# file, (path and file of casbert_data.zip)
```

## Searcher
The search here only accommodates the bag-of-words method.

### Instructions

```python
# import Searcher
from casbert_indexer import Searcher

# create a Searcher object
sc = Searcher()
# alternative if want to use a new generated index
indexFile = 'indexBoW'
sc = Searcher(idxVarFile=indexFile)

# get variables (in JSON format)
sc.searchVariables(query='concentration of triose phosphate in astrocytes', top=10, page=1)

# get SEDML (in JSON format)
sc.searchSedmls(query='concentration of triose phosphate in astrocytes', top=10, page=1)
```
