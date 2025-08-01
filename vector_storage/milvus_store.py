from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType


class CVVectorStore:
    def __init__(self):
        self.uri = "https://in03-ba90052e3130a37.serverless.gcp-us-west1.cloud.zilliz.com"
        self.token = "e48d006636a4c89f8f8a3db1dc94a8e05eb42f66b4616baf2dfc3376e0b7d0fee5693cc0f894269a01902d7bbf63fcc9a87792d9"
        self.collection_name = "cv_embeddings"

        connections.connect(
            uri=self.uri,
            token=self.token
        )

    def create_collection(self, dim=384):
      if utility.has_collection(self.collection_name):
          collection = Collection(self.collection_name)
      else:
          fields = [
              FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
              FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
              FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
              FieldSchema(name="candidate_name", dtype=DataType.VARCHAR, max_length=100)
          ]

          schema = CollectionSchema(fields=fields, primary_field="id")
          collection = Collection(self.collection_name, schema)

      # Create index if it doesn't exist
      index_params = {
          "index_type": "IVF_FLAT",
          "metric_type": "L2",
          "params": {"nlist": 128}
      }

      if not collection.has_index():
          collection.create_index(field_name="embedding", index_params=index_params)
          print("Index created successfully.")

      collection.load()  # Load after index creation
      return collection


    def insert_cv_chunks(self, vector_chunks):
      collection = self.create_collection()
      collection.load()  # Load collection before inserting

      entities = [
          [chunk["id"] for chunk in vector_chunks],
          [chunk["text"] for chunk in vector_chunks],
          [chunk["embedding"] for chunk in vector_chunks],
          [chunk["metadata"]["candidate_name"] for chunk in vector_chunks]
      ]

      insert_result = collection.insert(entities)
      collection.flush()

      print(f"Inserted {len(vector_chunks)} records.")
      print(f"Total records after insert: {collection.num_entities}")
      return insert_result

