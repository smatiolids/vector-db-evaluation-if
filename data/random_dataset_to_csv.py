import pandas as pd
import csv

def vector_to_string(vector):
    return  '[' + ', '.join(map(str, vector)) +']'

source_file_path = r'/Users/samuel.matioli/work/customers/ifood/VectorDBs/vector-db-external-main/data/random_dataset.pkl'
obj = pd.read_pickle(source_file_path)
df = pd.DataFrame(zip(obj['ids'],obj['metadata'],obj['vector']),columns=['row_id','metadata_s','vector'])

df["embedding"] = df["vector"].apply(vector_to_string)

df.to_csv(source_file_path.replace('.pkl','.csv'),
          sep=";",
          columns=['row_id','metadata_s','embedding'],
          index=False, 
          quoting=csv.QUOTE_ALL)