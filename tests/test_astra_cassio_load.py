import unittest
import os
from vector_db_external.vectordb.astra_cassio import AstraDBClient
import pickle

os.environ["ASTRA_TOKEN"] = ""
os.environ["ASTRA_DB_ID"] = ""


class TestAstraClient(unittest.TestCase):

    def test_insert_embeddings(self):
        # Create an instance of AstraDB
        client = AstraDBClient(token=os.environ["ASTRA_TOKEN"], db_id=os.environ["ASTRA_DB_ID"],
                               vector_dimension=1536, collection_name="load_test_random_cassio")

        source_file_path = r'../data/random_dataset.pkl'
        obj = {}
        with open(source_file_path, 'rb') as file1:         
            obj=pickle.load(file1)
        
        # Call insert_embeddings
        try:
            client.insert_embeddings(
                ids=obj['ids'], embeddings=obj['embeddings'], metadata=obj['metadata'], batch_size=256)
        except Exception as e:
            # If an exception occurs, fail the test with an informative message
            self.fail(f"Unexpected exception: {e}")

        # If no exception occurs, the test passes
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
