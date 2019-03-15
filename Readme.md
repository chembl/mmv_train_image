# MMV model training image

## Basic install and config

- Install docker. Install git if you're under windows as well.

  - https://docs.docker.com/install/
  - https://git-scm.com/downloads


You may need to configure Docker to let it use all available system memory if you're running this under Mac or Windows. 

On Mac:

<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/mac1.png?raw=true" width="300" ></a>
<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/mac2.png?raw=true" width="300" ></a>


On Windows (click on the whale icon on the windows tray icons area):

<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/win0.png?raw=true" width="150" ></a>
<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/win1.png?raw=true" width="300" ></a>
<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/win2.png?raw=true" width="300" ></a>


On Windows, you will need to explicitly configure volume sharing across your host and the container:

<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/win3.png?raw=true" width="300" ></a>



## Build the image and train the model

Run the following commands to set the docker image up and train the model. Use a terminal in either Linux or Mac and Windows Power Shell on Windows.


- Clone the repository, cd into the directory and [build the docker image](https://www.docker.com/resources/what-container). Be sure you have write rights on the directory (ex: avoid clonning to c:\WINDOWS\system32):

  ```
  git clone https://github.com/chembl/mmv_train_image.git
  cd mmv_train_image
  docker build -t chemblgroup/mmv_train .
  ```

- In case you experience any kind of difficulties building the image behind a proxy, you can skip the 'docker build' command and instead of it run:


   ```
   docker pull chemblgroup/mmv_train
   ```

- Training dataset have to be named training_set.csv. It must copied in model_train folder and formatted like [model_train/training_set_sample.csv](https://github.com/chembl/mmv_train_image/blob/master/model_train/training_set_sample.csv?raw=true).

- Configure a model by editing the configuration file([model_train/model_configs.json](https://github.com/chembl/mmv_train_image/blob/master/model_train/model_configs.json)) if needed. Available descriptors are:

    - Fingerprints: Select between fcfp or ecfp. Set the radius.
    - Physicochemical: alogp, mw, hba, hbd, rtb, n_h_atoms

- Run the [Docker container](https://www.docker.com/resources/what-container) using the previously generated image. The container will map your host's model_train directory to the container's /model_train directory allowing the collection of all output files.

  ```
  docker run -v /full/path/to/model_train:/model_train chemblgroup/mmv_train
  
  ex mac: docker run -v /Users/efelix/projects/mmv_train_image/model_train:/model_train chemblgroup/mmv_train
  ex linux: docker run -v /home/efelix/projects/mmv_train_image/model_train:/model_train chemblgroup/mmv_train
  ex windows: docker run -v C:\Users\efelix\projects\mmv_train_image\model_train:/model_train chemblgroup/mmv_train
  ```

The container will generate 5 files([description](https://github.com/chembl/mmv_train_image/blob/master/model_train/output_files_description.md)) for each model in model_train/outputs folder:

- modelX.json: the dump of the model.
- coverage_values_modelX.json: json file with a list of coverage values for each molecule in eMolecules dataset. Boxplots like the ones from the original publication will be generated with them.
- eMolecules_predictions_modelX.csv: eMolecules dataset predictions.
- internal_validation_report_modelX.json: classification metrics with training data using a 5 k-fold cross-validation.
- external_validation_report_modelX.json: classification metrics report with eMolecules predictions.

Send the required files to the EBI.
