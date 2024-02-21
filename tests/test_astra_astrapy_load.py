import unittest
import os
import shutil
import pickle
from vector_db_external.vectordb.astra_astrapy import AstraDBClient

os.environ["ASTRA_TOKEN"]=""
os.environ["ASTRA_DB_ID"]=""
os.environ["ASTRA_API_ENDPOINT"]=""


class TestAstraClient(unittest.TestCase):

    def test_insert_embeddings(self):
        # Create an instance of AstraDB
        client = AstraDBClient(token=os.environ["ASTRA_TOKEN"], api_url=os.environ["ASTRA_API_ENDPOINT"], vector_dimension=1536, collection_name="vector_random_dataset")

        source_file_path = r'../data/random_dataset.pkl'
        obj = {}
        with open(source_file_path, 'rb') as file1:         
            obj=pickle.load(file1)
        
        # Call insert_embeddings
        try:
            client.insert_embeddings(
                ids=obj['ids'], embeddings=obj['embeddings'], metadata=obj['metadata'])
        except Exception as e:
            # If an exception occurs, fail the test with an informative message
            self.fail(f"Unexpected exception: {e}")

        # If no exception occurs, the test passes
        self.assertTrue(True)

   
if __name__ == '__main__':
    unittest.main()
