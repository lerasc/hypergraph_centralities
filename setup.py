from setuptools import setup,find_packages

setup(
    name            =  'hypergraph_centralities',
    version         =  '1.0.0',
    packages        = ['hypergraph_centralities'], # can also write: find_packages()
    url             =  '',
    license         =  '',
    author          ='Sandro C. Lera',
    author_email    ='sandrolera@gmail.com',
    description     ='implementation of hypergraph centralities',
    python_requires ='>3.8.0',
    install_requires=[
                        "numpy>=1.21.5",
                        "pandas>=1.4.1",
                        "scipy>=1.8.0",
                        "networkx>=2.7.1",
                        "xarray>=2022.3.0", 
                    ]
)

