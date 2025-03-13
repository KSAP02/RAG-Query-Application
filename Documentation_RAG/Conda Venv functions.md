
Note:
Creating an environment file:
Type the following command in terminal in the project folder to save conda venv dependencies in an enviroment.yml file which can be used later to deploy the conda environment in a new folder/ directory for working on the project.

*conda env export --no-builds > environment.yml*

We need to do this because we can't push the conda virtual env into the github because of the amount of dependencies. So add the venv folder to the .gitignore and push the remaining project directory to github.


For creating virtual environment after cloning repo.

Inside the terminal of working folder, type:

*git clone "repo-url"*

*cd "project-folder"*

*conda env create -f environment.yml*

*conda activate virtual_env*