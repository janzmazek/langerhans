from setuptools import setup, find_packages

setup(
      name = "langerhans",
      version = "1.7.0",
      description = "Analyzer for the calcium signals of Islets of Langerhans.",
      author = "Jan Zmazek",
      url = "https://github.com/janzmazek/langerhans",
      license = "MIT License",
      packages = find_packages(exclude=['*test']),
      python_requires='>3.6.0',
      install_requires = ['numpy', 'scipy', 'matplotlib', 'pandas', 'networkx', 'pyyaml', 'python-louvain', 'openpyxl']
)
