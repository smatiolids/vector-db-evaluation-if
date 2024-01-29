# Vector Store Benchmarks

This project contains the description of load and quality tests to be performed using the vector databases of interest so we can share with vendors.


## Vector DB Clients

If possible, we would like support from vector database vendors in creating new Python clients using our API definition. To make it simple we are including our Vector DB API class.


New clients would have to extend from the `VectorDB` class (vector_db_external/vectordb/vectordb_api.py) and implement the methods to `insert_embeddings` and `search_embeddings`.

**Requirements**

* Python 3.8+
* Poetry


**Setup**

```sh
$ poetry install
```


**Implementation**


We have an implementation example using ChromaDB, you can check it at `vector_db_external/vectordb/chroma.py`.


We also created a test for it, you may run it as follows:

```sh 
$ poetry run python tests/test_chromadb.py
```

It would also be great if the vendors could implement this same test so we have a common ground.


## Infrastructure

The tests will be ran using an AWS EC2 instance: t3.2xlarge (8 vCPUs, 32.0 GiB) 


## Metrics

For each vector database are the metrics we will collect:

**Load Test metrics:**

* Query per second (QPS) (Throughtput)

* Latency (average, p99, p95, p50)

* CPU and memory consumption of vector store machine (when applicable)

* *Test scenarios*:

  * Write 
  
  * Embedding search

  * Filtered search

  * Embbedings with large text documents

**Retrieval quality metrics:**

* Context recall

* Context precision

* F1-Score


## Load

Initially, we want to scale up until 8K QPS


## Datasets


### Load Test

For load tests, we are using two different datasets. The goal is two represent scenarios that are suitable for applications at iFood.


#### Embeddings dataset

This dataset serves as a benchmark for evaluating the performance of Vector databases in querying embeddings and conducting filtered searches.



| Parameter           | Value                 |
|---------------------|-----------------------|
| **Dataset Size**    | 100,000               |
| **Vector Dimension** | 1536                 |
| **Metadata Fields** | "a" and "b"           |
| **Metadata Values**  | 10 distinct values : keyword_1 to keyword_10 |
| **Example Metadata**| `{"a": "keyword_1", "b": "keyword_6"}`        |
| **Text**            | n/a                   |


Dataset location:
https://github.com/vinicius-pirees/rag-load-test-datasets/tree/main/data/random_dataset

#### Embeddings + Documents dataset
In order to verify the Vector databases performance when large documents are returned, we used the Simple English Wikipedia dataset provided by OpenAI [available here](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip). We have made transformations so it is easily integrated with our load tests.


| Parameter           | Value                 |
|---------------------|-----------------------|
| **Dataset Size**    | 25,000                |
| **Vector Dimension** | 1536                 |
| **Metadata Fields** | n/a                   |
| **Text**            | Wikipedia articles    |




Dataset location:
https://github.com/vinicius-pirees/rag-load-test-datasets/tree/main/data/wikipedia_articles




### Quality Test

We will be using a dataset containing internal information about iFood, thus it cannot be shared. The dataset size is around 40K rows and we generated the embeddings using OpenAI' ada model.



## Qualitative evaluation features

Below is a list of the qualitative features we will take into consideration when evaluating vector databases. This will not be subject to load/quality tests, but we rather want to know about all the features and limitations of each vector database.

**Scalability**

Scalability is a key factor in evaluating a vector database, as it determines the system's ability to grow and handle increasing amounts of data and user requests. We will explore the different aspects of scalability, including horizontal and vertical scaling capabilities, distributed architecture, and how the database manages sharding and partitioning.

Distributed Architecture: Does it support a distributed architecture that allows for growth in data volume and query load? (Horizontal vs. Vertical)

Sharding and Partitioning: How does the database handle data partitioning and sharding to maintain performance as data grows?


**Data Types Support**

The types of data that a vector database can store and index are important to consider, especially when dealing with high-dimensional data. We will review the supported data types and the database's ability to handle them effectively.

Supported Data Types: What kinds of data types can be stored and indexed (e.g., floating-point vectors, binary vectors)?

High-Dimensional Data: How well does the database handle high-dimensional data? (Image, audio, etc)

**Pricing**

Cost is always a consideration when selecting a database. We will cover the licensing costs, operational expenses, and how scaling affects the overall cost of using the database.

Operational Costs: What are the costs associated with running the database (e.g., hardware, cloud services)?

Scaling Costs: How do costs change as the database scales?

Licensing Costs: Is the database open-source, or does it require a license?

**Ease of Use**

A user-friendly vector database can significantly reduce the learning curve and improve productivity. We will evaluate the user interface, documentation quality, and the level of community and support available to users.

Query Language: What kind of query language does it support, and how expressive is it?

User Interface: Does the database offer a user-friendly interface for administration and querying?

Documentation: Is there comprehensive documentation available?

Community and Support: Is there a strong community or support system in place for users?

**Managed Service**

Managed services can simplify database operations by providing cloud hosting and handling tasks such as backups, updates, and scaling. We will assess the availability and features of managed services for the vector database.

Cloud Hosting: Is there a managed service available on popular cloud platforms?

**Reliability**

The reliability of a vector database ensures that data remains consistent and available even in the face of failures. We will explore the database's fault tolerance, replication strategies, and disaster recovery mechanisms.

Fault Tolerance: How does the database handle failures and ensure data integrity?

Replication: Does it support data replication across nodes or data centers?

Disaster Recovery: What mechanisms are in place for disaster recovery?

**Data Management**

Effective data management is crucial for maintaining database performance over time. We will look at how the database handles batch operations, updates, deletions, and backups.

Batch Operations: Can the database efficiently handle batch inserts, updates, and deletions?

Updates: How are vector updates managed, and what is the impact on performance?

Deletions: How does the database handle deletions, and does it support soft deletes?

How data is grouped? Databases, tables, collections, etc

Access management: authentication, authorization (RBAC) 

**Monitoring**

Monitoring tools help maintain the health and performance of a database. We will discuss the availability of performance metrics, alerting systems, and logging capabilities in the vector database.

Performance Metrics: What kind of monitoring tools are available to track database performance?

Datadog integration: Datadog is our standard monitoring tool. Can we use it on top of the candidate solution?

Logging: Does the database provide comprehensive logs for troubleshooting?

**Crossing Information**

Join Operations: Can the database perform join operations between indexes?

Data Integration: How well does the database integrate with other data sources and systems?

**Filtering**

Filtering can improve search efficiency by narrowing down the search space. We will consider the database's support for pre-filtering and post-filtering operations.


**Ranking**

Ranking search results is essential for delivering relevant outcomes to users. We will review the options for custom ranking functions and the algorithms used for ranking in the vector database.

Custom Ranking: Can you define custom ranking functions based on vector similarity or other criteria?

Ranking Algorithms: What algorithms are available for ranking search results?


**Data storage**

Data storage format, which should be optimized for space efficiency.

Data retrieval performance and indexing mechanisms for efficient querying.

**Search**

The underlying search engine is a fundamental component of a vector database, as it dictates the efficiency and accuracy of the similarity search operations. Different search engines employ various algorithms and data structures to index and retrieve high-dimensional vector data. 

Indexing Algorithms: HNSW, Flat, DiskANN, etc 

Similarity measures (e.g., cosine similarity, Euclidean distance, or more advanced tree-based search algorithms) to compare vectors.

Search types: keyword, vector, or hybrid.