CREATE TABLE default_keyspace.load_test_random_dsbulk (
    row_id text PRIMARY KEY,
    attributes_blob text,
    body_blob text,
    metadata_s map<text, text>,
    embedding vector<float, 1536>
);

CREATE CUSTOM INDEX eidx_metadata_s_load_test_random_dsbulk ON default_keyspace.load_test_random_dsbulk (entries(metadata_s)) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';
CREATE CUSTOM INDEX idx_vector_load_test_random_dsbulk ON default_keyspace.load_test_random_dsbulk (embedding) USING 'org.apache.cassandra.index.sai.StorageAttachedIndex';
