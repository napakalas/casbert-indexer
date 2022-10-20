import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='casbert_indexer',
    version='0.0.1',
    author="Yuda Munarko",
    author_email="yuda.munarko@gmail.com",
    description="An interface to crawl, index, and generate results of biosimulation models. All the collected and generated data are indexed for biosimulation model retrieval purpose.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/napakalas/casbert-indexer.git",
    packages=setuptools.find_packages(),
    # packages=setuptools.find_namespace_packages(include=['casbert_indexer.*']),
    # namespace_packages=['casbert_indexer'],
    # package_dir={'': 'casbert_indexer'},
    install_requires=[
        'matplotlib',
        'lxml',
        'requests',
        'GitPython',
        'tellurium',
        'pandas',
        'numpy==1.20.3',
        'rdflib==4.2.2',
        'xmltodict',
        'urllib3',
        'beautifulsoup4',
        'scikit-learn==0.24.2',
        'hdbscan',
        'nltk',
        'torch',
        'sentence-transformers',
        'tqdm',
        'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_scibert-0.5.1.tar.gz'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    package_data={'': ['*resources/*', 'sedmlImages/*']},
)
