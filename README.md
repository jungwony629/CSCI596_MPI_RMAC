# CSCI596_MPI_RMAC
## Team members

Jungwon Yoon (jungwony@usc.edu)

Wonhyuk Jang (wonhyukj@usc.edu)

Wookyum Kim (wookyumk@usc.edu)

### 1. Goal: What’s the “big” problem? 
  &nbsp;&nbsp;&nbsp;&nbsp; Machine learning for image recognition requires tons of image data to train the computer. However, processing and extracting valid data from the images takes too much time if there are tons of images. We are tackling this well-known problem with parallel computing techniques we have learned through the class. We have one million images which are extracted from tons of videos. Our goal is to extract 512-dimensional vector data of each image which can utilize finding similar images using RMAC. RMAC is a machine learning model that is an image feature extractor to generate a vector dataset from a query image. As mentioned above, RMAC, one of image recognition modules, is taking a lot of time, to illustrate, one image spends an average of 63 seconds (only CPU), and then the total time consumption will be around one million minutes which is around 700 days. We are trying to implement MPI for parallelizing RMAC procedures and improve the efficiency of all processing by using multiple CPUs and GPUs.
  
### 2. Specific objectives: Step-by-step path to the goal 
- Set RMAC environment on local machines:
  - RMAC: https://github.com/noagarcia/keras_rmac 
  - Install prerequisites of RMAC
- Install MPI Python libraries
- MPI for Python: https://mpi4py.readthedocs.io/en/stable/:
  - Modify RMAC code for MPI
- Deploy RMAC on the server
- Run RMAC on the server:
  - GPU only
  - CPU only (if GPU is not available)

### 3. Current state of the knowledge/previous work.
  &nbsp;&nbsp;&nbsp;&nbsp; Currently, we have succeeded in running RMAC on the server. We have tried to use GPUs, however, there have occurred some technical issues with MPI + GPUs. First, RMAC occupies a lot of memories. For instance, each RMAC procedure takes around 1 GB of memories, which means that if we execute 10 threads of MPI, it would spend 10 GB of GPU memory. However, the server doesn’t allow using 10GB of GPUs only on the CARC server, therefore, we decided only to use CPUs. Another problem is that the Theano version of RMAC is too low to implement multiple GPUs, but since there are a lot of dependency problems for changing the version to other than Theano, we have decided to stick to Theano.

### 4. Techniques to be used: How to solve it? Big idea? Well-planned detail? 
  &nbsp;&nbsp;&nbsp;&nbsp; We already have the list of all image paths, this list can help in distribution of works among the threads. For example, if there are 5 threads, the program will divide the image paths into 5 image subsets. Each subset contains around 200 thousands images, the threads gather the results and then store the results into a JSON file. After all RMAC procedures have been finished, all datasets, built by the RMAC procedures, will be stored in PostreSQL. As a result, the program can vary the number of threads from the user query and then it will automatically split jobs into each thread.

### 5. Expected results: Research full of surprises but needs hypothesis/test;
  &nbsp;&nbsp;&nbsp;&nbsp; We can increase the efficiency of RMAC procedures if we use 21 threads with MPI. The expected time elapse of processing one million images becomes 1 month. Comparing with 23 months of using one thread to work on the same amount of images, we can see the great improvement of efficiency which results in a great amount of time saving. Our work on enhancing RMAC with using multiple CPUs and MPI can save 95.8% of the total time consumption of plain usage of RMAC on the single CPU. Now, we need to analyze and figure out how to implement RMAC on multiple threads with GPUs. Upto this point, we can’t change from the version of Theano, but if we changed the version, the program would process the data and gather the result much faster. The CPU is taking a minute for an image but the GPU is taking around 14 seconds for the same image. In order to improve the performance, we definitely need to use GPUs, however, as mentioned above, there are a lot of technical issues, like dependency problems, that we cannot solve yet. If we can switch to GPUs and change the version, then we can further improve the efficiency at least up to 4 times faster, that is, the program will be able to complete the train with 1 million images in around a week, not a month. To summarize, we can process and gather all results of images and improve the efficiency of RMAC by implementing MPI, but there are still more possibilities to upgrade our program further by switching to GPU and changing the RMAC version.

### Command to run this code:
```bash 
"mpiexec -np x python MPI_RMAC.py"
```
