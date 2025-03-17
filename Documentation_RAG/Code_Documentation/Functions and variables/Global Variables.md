![[../Images/global_variables.png]]

*load_dotenv()*
* Loads the .env file that is inside the working directory

Then we assign our api key that we got from hugging face by os.getenv('HF_API_KEY')
HF_API_KEY => is the variable in which the api key is stored in the .env file.

Next we decide our embedding model, llm model and the directory in which we want to extract information from pdfs.
We can change the embedding model and llm model as we wish as long as both versions are compatible with each other.