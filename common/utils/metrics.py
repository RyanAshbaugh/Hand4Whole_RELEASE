import pdb
import numpy as np


def DIR_FAR(score_mat, label_mat, ranks=[1], FARs=[1.0], get_false_indices=False,
            get_equal_error_rate=False, verbose=False):
    '''
    Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    Closed/Open-set Identification.
        A general case of Cummulative Match Characteristic (CMC)
        where thresholding is allowed for open-set identification.
    args:
        score_mat:            a P x G matrix, P is number of probes, G is size of gallery
        label_mat:            a P x G matrix, bool
        ranks:                a list of integers
        FARs:                 false alarm rates, if 1.0, closed-set identification (CMC)
        get_false_indices:    not implemented yet
    return:
        DIRs:                 an F x R matrix, F is the number of FARs, R is the number of ranks,
                              flatten into a vector if F=1 or R=1.
        FARs:                 an vector of length = F.
        thredholds:           an vector of length = F.
    '''
    score_mat = np.asarray(score_mat)
    label_mat = np.asarray(label_mat)
    assert score_mat.shape==label_mat.shape
    # assert np.all(label_mat.astype(np.float32).sum(axis=1) <=1 )
    # Split the matrix for match probes and non-match probes
    # subfix _m: match, _nm: non-match
    # For closed set, we only use the match probes
    match_indices = label_mat.astype(np.bool).any(axis=1)
    score_mat_m = score_mat[match_indices,:]
    label_mat_m = label_mat[match_indices,:]
    score_mat_nm = score_mat[np.logical_not(match_indices),:]
    label_mat_nm = label_mat[np.logical_not(match_indices),:]

    if verbose:
        print('mate probes: %d, non mate probes: %d' %
              (score_mat_m.shape[0], score_mat_nm.shape[0]))

    # Find the thresholds for different FARs
    max_score_nm = np.max(score_mat_nm, axis=1)
    label_temp = np.zeros(max_score_nm.shape, dtype=np.bool)
    if len(FARs) == 1 and FARs[0] >= 1.0:
        # If only testing closed-set identification, use the minimum score as threshold
        # in case there is no non-mate probes
        thresholds = [np.min(score_mat) - 1e-10]
        openset = False
    else:
        # If there is open-set identification, find the thresholds by FARs.
        assert score_mat_nm.shape[0] > 0, "For open-set identification (FAR<1.0), there should be at least one non-mate probe!"
        thresholds = find_thresholds_by_FAR(max_score_nm, label_temp, FARs=FARs)
        openset = True

    # Sort the labels row by row according to scores
    sort_idx_mat_m = np.argsort(score_mat_m, axis=1)
    sorted_label_mat_m = np.ndarray(label_mat_m.shape, dtype=np.bool)
    for row in range(label_mat_m.shape[0]):
        sort_idx = (sort_idx_mat_m[row, :])[::-1]
        sorted_label_mat_m[row,:] = label_mat_m[row, sort_idx]

    # Calculate DIRs for different FARs and ranks
    if openset:
        gt_score_m = score_mat_m[label_mat_m]
        assert gt_score_m.size == score_mat_m.shape[0]

    DIRs = np.zeros([len(FARs), len(ranks)], dtype=np.float32)
    FARs = np.zeros([len(FARs)], dtype=np.float32)
    if get_false_indices:
        false_retrieval = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_reject = np.zeros([len(FARs), len(ranks), score_mat_m.shape[0]], dtype=np.bool)
        false_accept = np.zeros([len(FARs), len(ranks), score_mat_nm.shape[0]], dtype=np.bool)
    for i, threshold in enumerate(thresholds):
        for j, rank  in enumerate(ranks):
            success_retrieval = sorted_label_mat_m[:,0:rank].any(axis=1)
            if openset:
                success_threshold = gt_score_m >= threshold
                DIRs[i,j] = (success_threshold & success_retrieval).astype(np.float32).mean()
            else:
                DIRs[i,j] = success_retrieval.astype(np.float32).mean()
            if get_false_indices:
                false_retrieval[i,j] = ~success_retrieval
                false_accept[i,j] = score_mat_nm.max(1) >= threshold
                if openset:
                    false_reject[i,j] = ~success_threshold
        if score_mat_nm.shape[0] > 0:
            FARs[i] = (max_score_nm >= threshold).astype(np.float32).mean()

    if DIRs.shape[0] == 1 or DIRs.shape[1] == 1:
        DIRs = DIRs.flatten()

    if get_false_indices:
        return DIRs, FARs, thresholds, match_indices, false_retrieval, false_reject, false_accept, sort_idx_mat_m
    if get_equal_error_rate:
        eer, _, _ = EER_FARs_FRRs(score_mat.flatten(), label_mat.flatten())
        return DIRs, FARs, thresholds, eer
    else:
        return DIRs, FARs, thresholds


# Find thresholds given FARs
# but the real FARs using these thresholds could be different
# the exact FARs need to recomputed using calcROC
def find_thresholds_by_FAR(score_vec, label_vec, FARs=None, epsilon=1e-5):
    #     Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool
    score_neg = score_vec[~label_vec]
    score_neg[::-1].sort()
    # score_neg = np.sort(score_neg)[::-1] # score from high to low
    num_neg = len(score_neg)

    assert num_neg >= 1

    if FARs is None:
        thresholds = np.unique(score_neg)
        thresholds = np.insert(thresholds, 0, thresholds[0]+epsilon)
        thresholds = np.insert(thresholds, thresholds.size, thresholds[-1]-epsilon)
    else:
        FARs = np.array(FARs)
        num_false_alarms = np.round(num_neg * FARs).astype(np.int32)

        thresholds = []
        for num_false_alarm in num_false_alarms:
            if num_false_alarm==0:
                threshold = score_neg[0] + epsilon
            else:
                threshold = score_neg[num_false_alarm-1]
            thresholds.append(threshold)
        thresholds = np.array(thresholds)

    return thresholds


def EER_FARs_FRRs(score_vec, label_vec, num_thresholds=int(1e4)):
    # Code borrowed from https://github.com/seasonSH/Probabilistic-Face-Embeddings

    assert len(score_vec.shape)==1
    assert score_vec.shape == label_vec.shape
    assert label_vec.dtype == np.bool

    sorted_indices = np.argsort(score_vec)[::-1]
    sorted_scores = score_vec[sorted_indices]
    sorted_labels = label_vec[sorted_indices]

    cumulative_TP = np.cumsum(sorted_labels)
    cumulative_FP = np.cumsum(~sorted_labels)

    FARs = cumulative_FP / cumulative_FP[-1]
    FRRs = 1 - cumulative_TP / cumulative_TP[-1]

    eer_index = np.argmin(np.abs(FARs - FRRs))
    EER = (FARs[eer_index] + FRRs[eer_index]) / 2

    return EER, FARs, FRRs
