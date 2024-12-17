# Final Combined Project

This project is the final combined submission for D400 and D100.

## Project Structure

- **final_combined/**: Main package with all code modules.
- **tests/**: Unit tests for the project.
- **setup.cfg**: Configuration file for installation.
- **environment.yml**: Conda environment configuration file.
- **.pre-commit-config.yaml**: Pre-commit hook configuration.

### Installation 

conda env create -f environment.yml

conda activate final

pip install .

pre-commit install

pre-commit run --all-files
