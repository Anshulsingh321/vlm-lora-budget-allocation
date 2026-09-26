import torch


def recall_at_k(
    similarity_matrix: torch.Tensor,
    ground_truth,
    ks=(1, 5, 10),
):
    """
    Compute Recall@K.

    similarity_matrix:
        [num_queries, num_candidates]

    ground_truth:
        ground_truth[i] = set/list of correct candidate indices
        for query i.
    """
    num_queries = similarity_matrix.size(0)

    max_k = max(ks)

    topk_indices = torch.topk(
        similarity_matrix,
        k=max_k,
        dim=1,
        largest=True,
    ).indices

    results = {}

    for k in ks:
        correct = 0

        for query_idx in range(num_queries):
            retrieved = set(
                topk_indices[query_idx, :k]
                .tolist()
            )

            targets = set(
                ground_truth[query_idx]
            )

            if retrieved.intersection(targets):
                correct += 1

        results[f"R@{k}"] = correct / num_queries

    return results