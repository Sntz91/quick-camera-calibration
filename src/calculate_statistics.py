import cv2
import re
import yaml
import numpy as np
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from lib.reference_points import ReferencePoints


def reference_points_experiment(img_name):
    # Load Config
    with open(f'conf/config_{img_name}.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    fname_val_pts_pv = cfg['perspective_view']['fname_val_pts']
    fname_ref_pts_pv = cfg['perspective_view']['fname_ref_pts']

    fname_val_pts_tv = cfg['top_view']['fname_val_pts']
    fname_ref_pts_tv = cfg['top_view']['fname_ref_pts']
    scaling_factor = cfg['top_view']['scaling_factor']

    selection_ref_pts = cfg['selection']['ref_pts']
    selection_val_pts = cfg['selection']['val_pts']

    val_pts_pv = ReferencePoints.load(fname_val_pts_pv)[selection_val_pts]
    val_pts_tv = ReferencePoints.load(fname_val_pts_tv)[selection_val_pts]

    data = []

    # For every combination
    for comb_len in range(4, len(selection_ref_pts)+1):
        print(f'start with {comb_len} nr of ref pts')
        combs = list(combinations(selection_ref_pts, comb_len))
        print(f'!!nr of experiments: {len(combs)}!!')

        # For every Experiment
        for i, comb in enumerate(combs):
            print(f'exp: {comb_len}, {i}')
            ref_pts_pv = ReferencePoints.load(fname_ref_pts_pv)[comb]
            ref_pts_tv = ReferencePoints.load(fname_ref_pts_tv)[comb]

            h = ReferencePoints.calc_homography_matrix(
                ref_pts_pv,
                ref_pts_tv
            )
            val_pts_pv_transformed = val_pts_pv.transform(h)
            distances, mean_d, var_d, std_d, nr_pts = ReferencePoints.calc_distances(
                val_pts_pv_transformed,
                val_pts_tv,
                scaling_factor
            )

            hull = cv2.convexHull(ref_pts_pv.get_numpy_arr().astype(int).reshape(-1, 1, 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = np.squeeze(hull)
            _hull = np.concatenate((hull, [hull[0, :]]))

            for name, val_pt in val_pts_pv.items():
                quality = get_validation_point_quality(val_pt, hull)
                data.append({
                    'img': img_name,
                    'comb': comb_len,
                    'exp': i,
                    'val_pt': name,
                    'val_pt_x': val_pt.x,
                    'val_pt_y': val_pt.y,
                    'quality': quality,
                    'hull': hull,
                    'hull_len': len(hull),
                    'dist_to_hull': cv2.pointPolygonTest(_hull,
                                                         (val_pt.x, val_pt.y),
                                                         True),
                    'hull_area': cv2.contourArea(_hull),
                    'ref_pts': ref_pts_pv.get_numpy_arr(),
                    'n_ref_pts': len(ref_pts_pv),
                    'error': distances[name]*100
                })
    df = pd.DataFrame(data)
    return df


def get_validation_point_quality(val_pt, hull):
    if cv2.pointPolygonTest(hull, (val_pt.x, val_pt.y), False) == -1:
        return 'bad'
    return 'good'


def create_data(fname):
    df = reference_points_experiment(fname)
    df.to_pickle('out/statistics/DJI_0026_df.pickle')


def read_data(fname):
    df = pd.read_pickle(f'out/statistics/{fname}_df.pickle')
    return df


def filter_bad_hulls(df):
    return df[df.hull_len > 3]


def filter_bad_points(df):
    return df[df.quality == 'good']


def get_n_worst_experiments(df, n_ref, n):
    df = df.loc[df['comb'] == n_ref]
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error') \
               .reset_index()
    return groups.iloc[-n:]


def get_n_worst_experiments_overall(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[-n:]


def get_n_best_experiments_overall(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[:n]


def get_n_best_experiments(df, n_ref, n):
    df = df.loc[df['comb'] == n_ref]
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[:n]


def plot_experiment(df, comb, exp):
    exp_df = df[(df.comb == comb) & (df.exp == exp)]
    mean_error = exp_df.error.mean()
    fname = f'/home/ziegleto/ziegleto/data/5Safe/vup/homography_evaluation/data/perspective_views/{exp_df.img.values[0]}.JPG'
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plotting
    plt.imshow(img)

    # Plot Reference Points
    ref_pts = exp_df.ref_pts.values[0]
    plt.scatter(ref_pts[:, 0], ref_pts[:, 1], color='white', alpha=0.4, s=80)

    # Plot Hull
    hull = exp_df.hull.values[0]
    plot_hull(hull)

    # Plot Validation Points
    plot_validation_points(exp_df)
    t = plt.text(50, 150, f'error: {mean_error:.2f}', color='white')
    t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    plt.axis('off')
    plt.savefig(f'out/{exp_df.img.values[0]}_{comb}_{exp}', dpi=300, bbox_inches='tight')


def plot_experiment_new(ax, df, comb, exp):
    exp_df = df[(df.comb == comb) & (df.exp == exp)]
    mean_error = exp_df.error.mean()
    fname = f'/home/ziegleto/ziegleto/data/5Safe/vup/homography_evaluation/data/perspective_views/{exp_df.img.values[0]}.JPG'
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plotting
    ax.imshow(img)

    # Plot Reference Points
    ref_pts = exp_df.ref_pts.values[0]
    ax.scatter(ref_pts[:, 0], ref_pts[:, 1], color='white', alpha=0.4, s=80)

    # Plot Hull
    hull = exp_df.hull.values[0]
    plot_hull_new(ax, hull)

    # Plot Validation Points
    plot_validation_points_new(ax, exp_df)

    t = ax.text(220, 150, f'error: {mean_error:.2f}', color='white')
    t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    ax.axis('off')
    return ax


def plot_validation_points(exp_df):
    for _, row in exp_df.iterrows():
        color = 'green' if row.quality == 'good' else 'red'
        plt.scatter(row.val_pt_x, row.val_pt_y, c=color, s=80, alpha=0.6)
        plt.text(row.val_pt_x, row.val_pt_y, f'E:{row.error:.2f}cm')


def plot_validation_points_new(ax, exp_df):
    for _, row in exp_df.iterrows():
        color = 'green' if row.quality == 'good' else 'red'
        ax.scatter(row.val_pt_x, row.val_pt_y, c=color, s=80, alpha=0.6)
        ax.text(row.val_pt_x, row.val_pt_y, f'E:{row.error:.2f}cm')


def plot_hull(hull):
    _hull = np.concatenate((hull, [hull[0, :]]))
    plt.plot(_hull[:, 0], _hull[:, 1], color='white', lw=2.0)


def plot_hull_new(ax, hull):
    _hull = np.concatenate((hull, [hull[0, :]]))
    ax.plot(_hull[:, 0], _hull[:, 1], color='white', lw=2.0)


def get_best_of_each_exp(df):
    groups = df.groupby(['comb', 'exp']) \
            .agg(mean_error=('error', 'mean')).reset_index()
    groups = groups.sort_values(by='mean_error')
    bestest = {}
    for i in range(4, 10):
        best_exp = groups[groups.comb == i].iloc[0]
        bestest[int(best_exp['comb'])] = {}
        bestest[int(best_exp['comb'])]['exp'] = int(best_exp['exp'])
        bestest[int(best_exp['comb'])]['mean_error'] = best_exp['mean_error']
    return bestest


def get_result_table(df):
    df_ = df.copy()
    df_['quality'] = 'overall'

    # Get Results per experiment
    groups1 = df_.groupby(['quality', 'comb', 'exp'])['error'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        ("nr val pts", "count")
    ])
    groups2 = df.groupby(['quality', 'comb', 'exp'])['error'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        ("nr val pts", "count")
    ])
    groups = pd.concat([groups1, groups2], axis=0).reset_index()
    # print(groups.groupby(['quality', 'comb'])['count'].max())
    groups = groups.groupby(['quality', 'comb'])['mean'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        ('nr of experiments', 'count')
    ])
    groups = groups.reset_index()
    return groups


def get_result_after_quality(df):
    groups = df.groupby(['comb', 'quality']).agg(mean_error=('error', 'mean'))
    print(groups)


def save_html(df, name, suffix):
    html_string = df.to_html(index=False)
    html = re.sub(r'<tr.*>', '<tr>', df.to_html().replace('border="1" ', ''))
    with open(f'out/{name}_{suffix}_table.html', 'w') as out:
        out.write(html_string)


def save_results(df, name):
    results1 = get_result_table(df)
    results1.to_csv(f"out/{name}_table_unfiltered.csv")
    save_html(results1, name, 'unfiltered')

    df = filter_bad_hulls(df)
    results2 = get_result_table(df)
    results2.to_csv(f"out/{name}_table_filtered.csv")
    save_html(results2, name, 'filtered')


if __name__ == '__main__':
    pd.set_option("display.precision", 2)
    pd.set_option("display.max_rows", None)
    img = 'DJI_0026'
    # create_data('DJI_0026')
    print(f"----- Save results for img {img} -----")
    df = read_data(img)
    #save_results(df, img)

    #plot_experiment(df, 4, 44)
    #plt.show()

    # plot_experiment(df, 4, 1)
    df = filter_bad_hulls(df)
    #df = df.loc[df['quality'] == 'good']
    #plot_experiment(df, 4, 8)
    # that's for the best/worst overall now...
    """
    fig, axs = plt.subplots(6, 8)
    for k in range(4, 10):
        losers = get_n_worst_experiments(df, k, 4)
        winners = get_n_best_experiments(df, k, 4)

        axs[k-4, 0].axis('on')
        axs[k-4, 0].set_ylabel(f'k={k}')
        axs[k-4, 0].axes.xaxis.set_ticklabels([])
        axs[k-4, 0].axes.yaxis.set_ticklabels([])

        i = 0
        for index, row in winners.iterrows():
            axs[k-4, i].set_title(f'best - {i+1}')
            plot_experiment_new(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
        if len(winners) == 1:  # No worst / best
            break
        for index, row in losers.iterrows():
            axs[k-4, i].set_title(f'worst - {i-3}')
            plot_experiment_new(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
    plt.tight_layout()
    plt.show()
    """
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    print(groups)

    # WINNERS OVERALL TODO
    # winners = get_n_best_experiments_overall(df, 10)
    # print(winners)
    # LOSERS OVERALL TODO
    #losers = get_n_worst_experiments_overall(df, 10)
    #print(losers)

    #plt.savefig('out/worst_best', dpi=300, bbox_inches='tight')

    #plt.savefig('out/worst_best', dpi=300, bbox_inches='tight')

    #print(df)
    #plot_experiment(df, 5, 15)
    #df = filter_bad_hulls(df)
    #print(len(df))
    #print(df)

    #get_overall_result(df)

    #get_result_after_quality(df)

    #df = filter_bad_points(df)

    #df = df.groupby(['comb', 'exp']).agg(mean_error=('error', 'mean'))
    #df = df.reset_index()
    #print(df)
    #df = df[df.mean_error <= 6.95]
    #plt.scatter(df.comb, df.mean_error)
    #plt.hlines(6.95, 4, 9, 'black')
    #plt.show()

    # 1. get the expeirments that are overall < 6.95
    # 2. select them from new dataframe
    #df_ = df.groupby(['comb', 'exp']).agg(mean_error=('error', 'mean'))
    #df_ = df_.reset_index()
    #df_ = df_[df_.mean_error <= 6.95]
    #exps = df_['exp'].values.tolist()
    #comps = df_['comb'].values.tolist()
    #print(exps.values)
    #df = df[df['exp'].isin(exps)]
    # comb + exp...
    #df = df.set_index(['exp', 'comb']).loc[zip(exps, comps)].reset_index()
    #print(df)
    #df = df.groupby(['comb', 'exp', 'quality', 'val_pt']).agg(mean_error=('error', 'mean'))
    #df = df.reset_index()
    #df['color'] = 'green' if df['quality'] == 'good' else 'red'
    #df['color'] = np.where(df['quality'] == 'good', 'green', 'red')
    #plt.scatter(df.comb, df.mean_error, c=df.color)
    #plt.hlines(6.95, 4, 9, 'black')
    #plt.show()
   # print(df[df.comb==9])


    #losers = get_n_worst_experiments(df, 5)
    #print(losers)
    # NA SCHAU MAL EINER AN: wenn man nur noch good points nimmt, dann error maximal 25cm.
    # Und die schlechtestens fuenf sind dann die, die so ne komische spitze haben.

    # EXPERIMENT OVERALL ------
    #get_overall_result(df) # Klar je mehr punkte desto besser, ABER
    #df = filter_bad_points(df)
    #get_overall_result(df) # Abstand geringer. 6-9 minimal
    #df = filter_bad_hulls(df)
    #get_overall_result(df) # 4 Holt auf.
    # -----------------

    # EXPERIMENT WORST OF EACH ----
    #losers = get_n_worst_experiments(df, 5) # FOUR, but...
    #print(losers)
    #df = filter_bad_points(df)
    #df = filter_bad_hulls(df)
    #losers = get_n_worst_experiments(df, 15) 
    #plot_experiment(df, 4, 0)
    #print(losers) # 4 - 5
    # --- TODO

    # EXPERIMENT BEST OF EACH --------
    #best_of_each_exp = get_best_of_each_exp(df)
    #for comb, exp in best_of_each_exp.items():
    #    print(f'{comb}, {exp["mean_error"]:.2f}')
        #plt.scatter(comb, exp['mean_error'])
    #    plot_experiment(df, comb, exp['exp'])
    #plt.show()


    # Die Besten Ergebnisse haben wir zwar mit vier Punkten, aber die Varianz ist enorm bei vier Punkten!!!
    # Auch die schlechtesten Ergebnisse natuerlich bei vier punkten. auch wenn wir nur good points und  ..., einbeziehen.
    # TODO Nur wenn len hull == len ref pts?
    # ------
