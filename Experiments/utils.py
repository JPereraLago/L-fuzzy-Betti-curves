import numpy as np
from gudhi import SimplexTree
from pyfzz import pyfzz
from collections import Counter
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle

########################################################
##### AVERAGE DISTANCE BETWEEN TWO COLORS          ##### 
########################################################

def avg_dist(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return 0.0
    total_distance = 0.0
    for i in range(P.shape[0]):
        dists = np.linalg.norm(Q - P[i], axis=1)
        closest_dist = np.min(dists)
        total_distance += closest_dist
    return total_distance / len(P)

########################################################
##### L-FUZZY BETTI CURVE AND EULER CHARACTERISTIC ##### 
########################################################

def FuzzyBettiCurve(
    simplex_tree: SimplexTree,
    fuzzy_values: list,
    predefined_grid: list[float] | np.ndarray,
    bins: int = 10):
    #Validating the fuzzy values:
    if sum(1 for simplex, _ in simplex_tree.get_filtration() if len(simplex) == 1) != len(fuzzy_values):
        raise ValueError("The length of fuzzy_values must be equal to the number of simplices in simplex_tree'.")        
    if all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in fuzzy_values):
        #Code for L=[0,1]
        max_dim = simplex_tree.dimension() + 1
        L = np.linspace(0.,1.,num=bins+1)
        fbcurves = [np.empty(shape=(bins,0)) for dim in range(max_dim)]
        for t in predefined_grid:
            filtered_simplices = [(simplex, filt)
                  for simplex, filt in simplex_tree.get_filtration()
                  if filt <= t]
            new_st = SimplexTree()
            for simplex, _ in filtered_simplices:
                val = -min(fuzzy_values[v] for v in simplex)
                new_st.insert(simplex, filtration=val)
            new_st.compute_persistence(homology_coeff_field=2, persistence_dim_max=True)
            for dim in range(max_dim):
                intervals = new_st.persistence_intervals_in_dimension(dim)
                betti = [-i[0] for i in intervals if np.isinf(i[1])]
                indices = np.digitize(x=betti, bins=L,right=True)
                counts = np.array([np.sum(indices == i) for i in range(1, bins+1)])
                fbcurves[dim] = np.hstack((fbcurves[dim], counts.reshape(-1, 1)))
        return fbcurves
        
    elif all(isinstance(x, str) and x in ('x', 'y') for x in fuzzy_values):
        #Code for L=FDL(x,y)
        max_dim = simplex_tree.dimension() + 1
        fbcurves = [np.empty(shape=(4,0)) for dim in range(max_dim)]
        for t in predefined_grid:
            filtered_simplices = [
                (simplex, 'x' if (labels := {fuzzy_values[i] for i in simplex}) == {'x'}
                else 'y' if labels == {'y'}
                else 'x_and_y')
                for simplex, filt in simplex_tree.get_filtration()
                if filt <= t]
            n = len(filtered_simplices)
            inclusion = sorted(filtered_simplices,
                key=lambda fs: ({'x':0,'y':1,'x_and_y':2}[fs[1]], len(fs[0])))
            deletion = sorted(filtered_simplices,
                key=lambda fs: ({'x_and_y':0,'x':1,'y':2}[fs[1]], -len(fs[0])))
            zigzag_filtration = [('i', simplex) for simplex, _ in inclusion] + \
                [('d', simplex) for simplex, _ in deletion]
            zigzag_labels = [label for _, label in inclusion] + \
                [label for _, label in deletion]
            fzz = pyfzz()
            bars = fzz.compute_zigzag(data=zigzag_filtration)
            for dim in range(max_dim):
                colors = [(zigzag_labels[a-1],zigzag_labels[b]) 
                    for (a,b,d) in bars if d==dim and a<=n and n<=b]
                normalized = [tuple(sorted(pair)) for pair in colors]
                counts = Counter(normalized)
                result = np.array([
                    counts[('x_and_y', 'x_and_y')],
                    counts[('x', 'x')] + counts[('x', 'x_and_y')],
                    counts[('y', 'y')] + counts[('x_and_y', 'y')],
                    counts[('x', 'y')]])
                fbcurves[dim] = np.hstack((fbcurves[dim], result.reshape(-1, 1)))
        return fbcurves
   
    else:
        raise ValueError("All elements in fuzzy_values must be either floats between 0 and 1, or strings equal to 'x' or 'y'.")

def FuzzyEulerCharacteristic(fbcurves: list):
    signs = np.array([(-1)**i for i in range(len(fbcurves))]).reshape(-1, *([1] * fbcurves[0].ndim))
    euler = np.sum(signs * np.stack(fbcurves), axis=0)
    return euler

########################################################
##### BETTI CURVE FROM BARCODE ######################### 
########################################################

def betti_curve_from_barcode(barcode, predefined_grid):
    l = len(barcode)
    births = np.array([bar[0] for bar in barcode])
    deaths = np.array([bar[1] for bar in barcode])
    events = np.concatenate([births,deaths])
    signs = np.array([1]*l+[-1]*l)
    order = np.argsort(events)
    events = events[order]
    signs = signs[order]
    betti_at_events = np.cumsum(signs)
    idx = np.searchsorted(events, predefined_grid, side="right") - 1
    betti = np.zeros(len(predefined_grid), dtype=int)
    valid = idx >= 0
    betti[valid] = betti_at_events[idx[valid]]
    return betti.tolist()

########################################################
##### CLASSIFIERS ###################################### 
########################################################

def f1_dist(X,y,n_iter,test_size,n_trees):
    f1_array = np.empty(n_iter)
    for i in tqdm(range(n_iter)):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=i)
        model = RandomForestClassifier(
            n_estimators=n_trees, random_state=2*i, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_array[i] = f1_score(y_test, y_pred, average="macro")
    return f1_array
            
########################################################
##### STATISTIC TESTS ################################## 
########################################################

def compare_samples(sample1, sample2, alpha=0.1):
    # 0 = Tie, 1 = Sample 1 is greater, 2 = Sample 2 is greater
    pvalue = stats.mannwhitneyu(sample1, sample2, alternative="two-sided").pvalue
    if pvalue >= alpha:
        return 0
    pvalue = stats.mannwhitneyu(sample1, sample2, alternative="greater").pvalue
    return 1 if pvalue < alpha else 2