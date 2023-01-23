from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind


def get_pearson_r(x, y):
    """Calculates the Pearson r of the two lists"""
    return pearsonr(x, y)


def get_spearman_r(x, y=None, axis=0):
    """Calculates the Spearman r of the two lists"""
    return spearmanr(x, y, axis)


def get_kendall_tau(x, y):
    """Calculates the Kendall Tau of the two lists"""
    return kendalltau(x, y)


def get_t_test(x, y):
    return ttest_ind(x, y)
