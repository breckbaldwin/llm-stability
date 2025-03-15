### Models

The models being evaluated should reside in this directory for access by the run_evaluation.py script. Look at the existing implementations for functions they need to implement. We are not using classes to keep the implementation as simple as possible. 

All our models require endpoints and credentials to be identified via environment variables.An easy way to set variables is to have a `.env` file in your home directory for `dotenv` to find. DO NOT PUT `.env` in repo or you risk putting credentials in the repo.

There are unit tests in `tests/test_models` that show how the models are 
invoked. 