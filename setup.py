from setuptools import setup, find_packages

setup(
      name = "langerhans",
      version = "1.1.0",
      description = "Analyzer for the calcium signals of Islets of Langerhans.",
      author = "Jan Zmazek",
      url = "https://github.com/janzmazek/cell-networks",
      license = "MIT License",
      packages = find_packages(exclude=['*test']),
      install_requires = ['numpy', 'scipy', 'matplotlib', 'networkx', 'pyyaml']
)
