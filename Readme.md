# MMV model training image


- Install docker.

  https://docs.docker.com/install/

You may need to configure Docker to let it use all available system Memory.

- Clone the repository, cd into the directory and build the docker image:

  ```
  git clone https://github.com/chembl/mmv_train_image.git
  cd mmv_train_image
  docker build -t mmv_train .
  ```

- Copy train(training_set.csv) and coverage(coverage_set.csv) datasets in model_train folder. Datasets should be formatted like model_train/training_set_sample.csv and model_train/coverage_set_sample.csv files.

- Set the descriptors to train the model in model_train/config.json. Available descriptors are:

    - Fingerprints: Select between fcfp or ecfp.
    - Physicochemical: alogp, mw, hba, hbd, rtb, n_h_atoms

- Run the container using the previously generated image.

  ```
  docker run -v /full/path/to/model_train:/model_train mmv_train
  
  ex: docker run -v /Users/efelix/projects/mmv_train_image/model_train:/model_train mmv_train
  ```

Send model_train/model.json and model_train/coverage.json files to the EBI.
