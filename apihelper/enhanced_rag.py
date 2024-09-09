import pdb
import redis
import numpy as np



class EnhancedRag:
    def __init__(self):
        self.im_search = Top50(db=0)
        self.text_search = Top50(db=1)
        print("EnhancedRag initialized")

    def transform_key(self, key):
        return "/".join(key.split("/")[:-1])

    def search(self, query_image_embedding,query_text_embedding,query_key):
        im_keys, im_scores = self.im_search.search(query_image_embedding,query_key)
        text_keys, text_scores = self.text_search.search(query_text_embedding,self.transform_key(query_key))
        # print("query_key",query_key)
        # print("text_query_key",self.transform_key(query_key))
        print("im_keys",len(im_keys))
        print("text_keys",len(text_keys))
        results = []
        for i,key in enumerate(im_keys):
            match_key = self.transform_key(key)
            if match_key in text_keys:
                idx = text_keys.index(match_key)
                record = {'key': key,
                          'im_score': im_scores[i],
                          'text_score': text_scores[idx],
                          'score': (0.6*im_scores[i]) + (0.4*text_scores[idx])}
                results.append(record)
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        return results

class Top50:
    def __init__(self,db=1):
        redis_conn = redis.Redis("localhost", 6379, db=db)
        # get embeddings
        self.embeddings = []
        self.keys = []
        for key in redis_conn.scan_iter():
            self.keys.append(key.decode('utf-8'))
            embedding_bytes = redis_conn.get(key)
            embedding = np.frombuffer(embedding_bytes, dtype='float32')
            self.embeddings.append(embedding)
        self.embeddings = np.array(self.embeddings)
    
    def get_embedding(self,key):
        return self.embeddings[self.keys.index(key)]

    def search(self, query_embedding,query_key):
        distances = np.dot(self.embeddings,query_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        top5 = np.argsort(distances)[::-1][:5001]
        top5_keys, top5_scores = [], []
        for i in top5:
            if query_key != self.keys[i]:
                top5_keys.append(self.keys[i])
                top5_scores.append(distances[i])
        return top5_keys, top5_scores



if __name__ == '__main__':
    # top50 = Top50()
    # print(top50.keys[0])
    # print(top50.embeddings.shape)
    # top50_keys, top50_scores = top50.search(top50.embeddings[0],top50.keys[0])
    # print(top50_keys)
    # print(top50_scores)
    # print(len(top50_keys))
    rag = EnhancedRag()
    query_image_embedding = rag.im_search.embeddings[0]
    query_key = rag.im_search.keys[0]
    query_text_embedding = rag.text_search.get_embedding(rag.transform_key(query_key))
    results = rag.search(query_image_embedding,query_text_embedding,query_key)
    print(len(results))
    for item in results:
        print(item)
