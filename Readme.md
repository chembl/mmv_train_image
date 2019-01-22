# MMV model training image


- Install docker.

  https://docs.docker.com/install/

You may need to configure Docker to let it use all available system Memory if you're under Mac or Windows.

In mac:

<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/mac1.png" width="300" ></a>
<a><img src="https://github.com/chembl/mmv_train_image/blob/master/images/mac2.png" width="300" ></a>


- Clone the repository, cd into the directory and build the docker image:

  ```
  git clone https://github.com/chembl/mmv_train_image.git
  cd mmv_train_image
  docker build -t mmv_train .
  ```

- Copy train(training_set.csv) and coverage(coverage_set.csv) datasets in model_train folder. Datasets should be formatted like model_train/training_set_sample.csv and model_train/coverage_set_sample.csv files.

- Configure a model in the configuration file(model_train/config.json) if needed. Available descriptors are:

    - Fingerprints: Select between fcfp or ecfp.
    - Physicochemical: alogp, mw, hba, hbd, rtb, n_h_atoms

- Run the container using the previously generated image.

  ```
  docker run -v /full/path/to/model_train:/model_train mmv_train
  
  ex: docker run -v /Users/efelix/projects/mmv_train_image/model_train:/model_train mmv_train
  ```

The container will generate 4 files for each model:

- modelX.json: the dump of the model.
- coverage_values_modelX.json: file with coverage values for each molecule in the coverage_set file.
- predictions_modelX.csv: eMolecules dataset predictions.
- report_modelX.json: classification metrics report using eMolecules predictions.

Send the model and coverage files to the EBI.
