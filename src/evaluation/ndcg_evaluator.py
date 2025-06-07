import math
import numpy as np

class NDCGEvaluator:
    """nDCG evaluation for retrieval systems"""

    @staticmethod
    def dcg_at_k(relevance_scores, k):
        """Calculate DCG@k"""
        relevance_scores = relevance_scores[:k]
        dcg = relevance_scores[0] if len(relevance_scores) > 0 else 0

        for i in range(1, len(relevance_scores)):
            dcg += relevance_scores[i] / math.log2(i + 1)

        return dcg

    @staticmethod
    def ndcg_at_k(predicted_relevance, ideal_relevance, k):
        """Calculate nDCG@k"""
        dcg = NDCGEvaluator.dcg_at_k(predicted_relevance, k)
        idcg = NDCGEvaluator.dcg_at_k(sorted(ideal_relevance, reverse=True), k)
        return dcg / idcg if idcg > 0 else 0

    @staticmethod
    def evaluate_retrieval(golden_dataset, retrieval_results, k=10):
        """Evaluate retrieval system using nDCG"""
        ndcg_scores = []

        for query_data in golden_dataset:
            query = query_data['query']
            expected_results = {item['episode_id']: item['relevance']
                                for item in query_data['expected_results']}

            if query in retrieval_results:
                predicted_ranking = retrieval_results[query]

                # Extract relevance scores for predicted ranking
                predicted_relevance = []
                for doc_id, _ in predicted_ranking[:k]:
                    relevance = expected_results.get(doc_id, 0)
                    predicted_relevance.append(relevance)

                # Get ideal relevance scores
                ideal_relevance = list(expected_results.values())

                # Calculate nDCG
                ndcg = NDCGEvaluator.ndcg_at_k(predicted_relevance, ideal_relevance, k)
                ndcg_scores.append(ndcg)

        return np.mean(ndcg_scores) if ndcg_scores else 0