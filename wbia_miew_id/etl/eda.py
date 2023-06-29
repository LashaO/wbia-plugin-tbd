
import numpy as np

def print_intersect_stats(df_a, df_b, individual_key="name"):
    print("** cross-set stats **")
    print()
    print(' - Counts: ')
    names_a = df_a[individual_key].unique()
    names_b = df_b[individual_key].unique()
    
    
    print("     number of individuals in train: ", len(names_a))
    print("     number of annotations in train: ", len(df_a))
    print()
    print("     number of individuals in test: ", len(names_b))
    print("     number of annotations in test: ", len(df_b))
    print()
    avg_ratio_train = len(df_a) / len(names_b)
    avg_ratio_test = len(df_b) / len(names_b)
    print(f"    average number of annotations per individual in train: {avg_ratio_train:.2f}")
    print(f"    average number of annotations per individual in test: {avg_ratio_test:.2f}")
    print()

    print(' - New individuals: ')
    # NOTE setdiff1d is not symmetric
    names_diff = np.setdiff1d(names_b, names_a)
    print("     number of new (unseen) individuals in test:", len(names_diff))
    ratio_diff = len(names_diff) / len(names_b)
    print(f"    ratio of new names to all individuals in test: {ratio_diff:.2f}")
    print()
    print(" - Individuals in both sets: ")
    len_intersect = len(np.intersect1d(names_a, names_b))
    print("     number of overlapping individuals in train & test:", len_intersect)
    ratio_a = len_intersect / len(names_a)
    ratio_b = len_intersect / len(names_b)
    print(f"    ratio of overlapping names to total individuals in train: {ratio_a:.2f}")
    print(f"    ratio of overlapping names to total individuals in test: {ratio_b:.2f}")
    print()

def print_min_max_counts(df, key, min_thresh, max_thresh = None):
    vc = df[key].value_counts()
    if not max_thresh:
        max_thresh = vc.max()
    vc_thresh = vc[(vc <= max_thresh)&(vc >= min_thresh)]
    num_ids = len(vc_thresh)
    num_annos = len(df[df[key].apply(lambda x: x in list(vc_thresh.index))])
    print(f"min-max {min_thresh}-{max_thresh}, ids {num_ids}, samples {num_annos}")
    
def print_min_max_stats(df, key, threshold_list = [
    (0,2), (0, 3), (10, None), (3,10), (0, 10), (3,None), (0, None), (2,10)]
                     ):
    print("** min-max stats **")
    for min_thresh, max_thresh in threshold_list:
        print_min_max_counts(df, key, min_thresh, max_thresh)