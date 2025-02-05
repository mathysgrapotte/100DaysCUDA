## Rewrote naive kernel from scratch 

Ideal is to rewrite and understand the indexing with pen&paper to get the logic in, rewrote the naive kernel from the top of my head using only a pen and paper. From now on, will use the anthropic blog [here](https://siboehm.com/articles/22/CUDA-MMM) as reference. 

## Setup Nsight on Colab 

To setup Nsight on Colab run this :
```
!apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         apt-transport-https \
         ca-certificates \
         gnupg \
         wget && \
     rm -rf /var/lib/apt/lists/*
!wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add - && \
     echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
     apt-get update -y && \
     DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
         nsight-systems-2022.1.1 nsight-compute-2022.1.1 && \
     rm -rf /var/lib/apt/lists/*

# setting environment variable path
import os
os.environ["PATH"] = "/usr/local/bin" + os.pathsep + os.getenv("PATH")
```

Then check if it is installed using this command:

```
!nsys --version
!echo "---"
!ncu --version
```

We can then run profiling using : 

```
ncu -o my_report --set full ./myapp
```


To download files from Nsight, we need to use google colab librairy : 

```
from google.colab import files
files.download(`content/myreport.(nvidia-report-extension)`)
```

This will prompt a download in the browser, you then need nsight installed locally to view the reported file

