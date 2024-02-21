import cv2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stat_data import (
    create_data,
    read_data,
    filter_bad_hulls
)


def main():
    img = 'DJI_0029'
    # create_data(img)
    df = read_data(img)
    df = filter_bad_hulls(df)
    # df = df.loc[df['quality'] == 'good']

    # plot_winners_losers(df, 'out/worst_best')
    # for i in range(100):
    plot_experiment(df, 10, 0, save=True)

    # save_results(df, img)
    # results = get_result_table(df)
    # print(results)
    # winners = get_n_best_experiments(df, 10)
    # print(winners)
    # df = df.loc[df['comb']==13]
    # losers = get_n_worst_experiments(df, 10)
    # print(losers)


def get_n_worst_experiments(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[-n:]


def get_n_best_experiments(df, n):
    groups = df.groupby(['comb', 'exp']) \
               .agg(mean_error=('error', 'mean')) \
               .sort_values(by='mean_error').reset_index()
    return groups.iloc[:n]


def plot_experiment(df, comb, exp, save=True):
    exp_df = df[(df.comb == comb) & (df.exp == exp)]
    _, ax = plt.subplots()
    ax = plot_experiment_ax(ax, df, comb, exp)
    if save:
        plt.savefig(
            f'out/{exp_df.img.values[0]}_{comb}_{exp}',
            dpi=300,
            bbox_inches='tight'
        )
    else:
        plt.show()
    plt.clf()


def plot_experiment_ax(ax, df, comb, exp):
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
    plot_hull(ax, hull)

    # Plot Validation Points
    plot_validation_points(ax, exp_df)

    t = ax.text(220, 150, f'error: {mean_error:.2f}', color='white')
    t.set_bbox(dict(facecolor='black', alpha=0.5, edgecolor='black'))
    ax.axis('off')
    return ax


def plot_validation_points(ax, exp_df):
    for _, row in exp_df.iterrows():
        color = 'green' if row.quality == 'good' else 'red'
        ax.scatter(row.val_pt_x, row.val_pt_y, c=color, s=80, alpha=0.6)
        ax.text(row.val_pt_x, row.val_pt_y, f'E:{row.error:.2f}cm')


def plot_hull(ax, hull):
    _hull = np.concatenate((hull, [hull[0, :]]))
    ax.plot(_hull[:, 0], _hull[:, 1], color='white', lw=2.0)


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
        'var',
        ("nr val pts", "count")
    ])
    groups2 = df.groupby(['quality', 'comb', 'exp'])['error'].agg([
        'mean',
        'min',
        'max',
        ("quantile 0.25", lambda x: x.quantile(0.25)),
        ("quantile 0.50", lambda x: x.quantile(0.5)),
        ("quantile 0.75", lambda x: x.quantile(0.75)),
        'var',
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
        'var',
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


def plot_winners_losers(df, filename):
    fig, axs = plt.subplots(6, 8)
    fig.set_size_inches(25, 15)
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
            plot_experiment_ax(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
        if len(winners) == 1:  # No worst / best
            break
        for index, row in losers.iterrows():
            axs[k-4, i].set_title(f'worst - {i-3}')
            plot_experiment_ax(axs[k-4, i], df, row['comb'], row['exp'])
            i += 1
    plt.tight_layout()
    plt.show()
    # plt.savefig(filename, dpi=300)


if __name__ == '__main__':
    plt.style.use('dark_background')
    pd.set_option("display.precision", 2)
    pd.set_option("display.max_rows", None)
    main()
