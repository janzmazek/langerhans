# cell-networks
Analysis of cell signals in a network.

# Usage
The default structure is:
```bash
├── cell_networks
├── DATA
│   ├── 001
│   │   ├── filtered
│   │   ├── distributions
│   │   ├── binarized
│   │   ├── analysis
│   │   ├── series001.dat
│   │   ├── positions001.dat
│   │   ├── settings.yaml
```

Run ```python cell_networks``` outside the package. Enter series number (e.g. ```001```). To change settings, edit ```settings.yaml``` file (sample settings file is in "sample" directory of the package).

# Data structure

 ```seriesXXX.dat``` should be a numpy 2D array with cell indices as rows and time series as columns. Data should be separated by whitespace.
  ```positionsXXX.dat ``` should be a numpy 2D array with cell indices as rows and x & y coordinates as columns.
